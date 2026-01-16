import os
import math
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN

import torchvision
import torchvision.transforms as T

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from cnn_test import *

import os
import matplotlib.pyplot as plt

# -----------------------------
# 1) Simple CNN (similar to your structure)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Head
        # CIFAR10 input 32x32:
        # after block1 + pool => 16x16
        # after block2 + pool => 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # <-- good Grad-CAM target
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class CNN5_Extractor(nn.Module):
    """
    CNN5 mid-layer maps + Flatten (no GAP)
      fe_map: (B,64,16,16) -> fe_flat (B,16384)
      fi_map: (B,128,8,8) -> fi_flat (B,8192)
    Return concatenated (B,24576) for baseline.
    Also provide a method to return separated flats for PurLock.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

    def forward_maps(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        fe_map = self.pool(x)                  # (B,64,16,16)

        x = F.relu(self.bn3(self.conv3(fe_map)))
        x = F.relu(self.bn4(self.conv4(x)))
        fi_map = self.pool(x)                  # (B,128,8,8)
        return fe_map, fi_map

    def forward_separate(self, x):
        fe_map, fi_map = self.forward_maps(x)
        fe = fe_map.flatten(1)                 # (B,16384)
        fi = fi_map.flatten(1)                 # (B,8192)
        return fe, fi

    def forward_fe_map(self, x):
        fe_map, _ = self.forward_maps(x)
        return fe_map

    def forward_fi_map(self, x):
        _, fi_map = self.forward_maps(x)
        return fi_map

    def forward(self, x):
        fe, fi = self.forward_separate(x)
        return torch.cat([fe, fi], dim=1)      # (B,24576)


class UnauthClassifierFromFeMap(nn.Module):
    """
    Simple CNN head on fe_map (B,64,16,16) -> logits (B,10)
    Keep it small but convolutional so Grad-CAM makes sense.
    """
    def __init__(self, inC=64, num_classes=10):
        super().__init__()
        self.convA = nn.Conv2d(inC, 128, kernel_size=3, padding=1)
        self.bnA   = nn.BatchNorm2d(128)
        self.convB = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bnB   = nn.BatchNorm2d(128)

        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(128, num_classes)

    def forward(self, fe_map):
        x = F.relu(self.bnA(self.convA(fe_map)))
        x = F.relu(self.bnB(self.convB(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)

class UnauthPipeline(nn.Module):
    def __init__(self, extractor, obf_fe, unauth_head):
        super().__init__()
        self.extractor = extractor
        self.obf_fe = obf_fe
        self.unauth_head = unauth_head

    def forward(self, x):
        fe_map, fi_map = self.extractor.forward_maps(x)   # (B,64,16,16)
        fe_map_t = self.obf_fe(fe_map)               # obfuscated map
        logits = self.unauth_head(fe_map_t)
        return logits


# -----------------------------
# 2) Grad-CAM helper
# -----------------------------
class GradCAM:
    """
    Minimal Grad-CAM:
      - register forward hook to save activations A (B,C,H,W)
      - register backward hook to save gradients dY/dA (B,C,H,W)
      - weight channels by spatial avg gradient -> alpha (B,C,1,1)
      - cam = ReLU(sum_c alpha_c * A_c) -> (B,1,H,W) normalized
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        # full backward hook is safer in newer torch
        self.bwd_handle = target_layer.register_forward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # (B,C,H,W)

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] is gradient w.r.t. module output
        self.gradients = grad_output[0]  # (B,C,H,W)

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # cam: (B,1,H,W)
        B = cam.size(0)
        cam_ = cam.view(B, -1)
        cam_min = cam_.min(dim=1, keepdim=True).values
        cam_max = cam_.max(dim=1, keepdim=True).values
        cam_ = (cam_ - cam_min) / (cam_max - cam_min + eps)
        return cam_.view_as(cam)

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Returns:
          cam_norm: (B,1,h,w) in [0,1] at target layer resolution
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # (B,C)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)  # (B,)

        # pick target logit for each sample
        target = logits.gather(1, class_idx.view(-1, 1)).squeeze(1)  # (B,)
        target.sum().backward(retain_graph=False)

        # activations and gradients must exist
        A = self.activations            # (B,C,H,W)
        dYdA = self.gradients           # (B,C,H,W)

        if A is None or dYdA is None:
            raise RuntimeError("GradCAM: activations/gradients not captured. Check target_layer hook.")

        # If missing batch dim: (C,H,W) -> (1,C,H,W)
        if dYdA.dim() == 3:
            dYdA = dYdA.unsqueeze(0)
        if A.dim() == 3:
            A = A.unsqueeze(0)

        # Compute alpha by averaging over all spatial dims (from dim=2 onward)
        spatial_dims = tuple(range(2, dYdA.dim()))
        alpha = dYdA.mean(dim=spatial_dims, keepdim=True)  # (B,C,1,1)

        cam = (alpha * A).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)
        return cam

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

# class GradCAM:
#     def __init__(self, model: nn.Module, target_layer: nn.Module):
#         self.model = model
#         self.target_layer = target_layer
#         self.activations = None
#         self.gradients = None
#
#         self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
#
#     def _forward_hook(self, module, inp, out):
#         # out is the activation tensor A: (B,C,H,W)
#         self.activations = out
#
#         # register tensor hook to capture gradients dY/dA
#         def _tensor_grad_hook(grad):
#             self.gradients = grad
#
#         out.register_hook(_tensor_grad_hook)
#
#     @torch.no_grad()
#     def _normalize_cam(self, cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#         B = cam.size(0)
#         cam_ = cam.view(B, -1)
#         cam_min = cam_.min(dim=1, keepdim=True).values
#         cam_max = cam_.max(dim=1, keepdim=True).values
#         cam_ = (cam_ - cam_min) / (cam_max - cam_min + eps)
#         return cam_.view_as(cam)
#
#     def __call__(self, x: torch.Tensor, class_idx: torch.Tensor = None) -> torch.Tensor:
#         self.model.zero_grad(set_to_none=True)
#
#         logits = self.model(x)
#         if class_idx is None:
#             class_idx = logits.argmax(dim=1)
#
#         target = logits.gather(1, class_idx.view(-1, 1)).squeeze(1)
#         target.sum().backward(retain_graph=False)
#
#         A = self.activations          # (B,C,H,W)
#         dYdA = self.gradients         # (B,C,H,W)
#
#         alpha = dYdA.mean(dim=(2, 3), keepdim=True)     # (B,C,1,1)
#         cam = (alpha * A).sum(dim=1, keepdim=True)      # (B,1,H,W)
#         cam = F.relu(cam)
#         cam = self._normalize_cam(cam.detach())
#         return cam
#
#     def close(self):
#         self.fwd_handle.remove()


# -----------------------------
# 3) Training / Testing ACC (classification)
# -----------------------------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    total_acc, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        bs = x.size(0)
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_acc / n


# -----------------------------
# 4) Grad-CAM Evaluation (no GT): Deletion / Insertion + confidence drop
# -----------------------------
def upsample_cam(cam: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    # cam: (B,1,h,w) -> (B,1,H,W)
    return F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def apply_mask_topk(x: torch.Tensor, cam01: torch.Tensor, keep_ratio: float, mode: str) -> torch.Tensor:
    """
    x: (B,3,H,W)
    cam01: (B,1,H,W) in [0,1]
    keep_ratio: keep top-k ratio of pixels by CAM value
    mode:
      - "deletion": remove salient regions => mask out top-k (set to 0)
      - "insertion": keep only salient regions => keep top-k, others 0
    """
    B, _, H, W = x.shape
    flat = cam01.view(B, -1)  # (B,HW)

    k = max(1, int(keep_ratio * flat.size(1)))
    # threshold per sample: kth largest
    topk_vals, _ = torch.topk(flat, k=k, dim=1, largest=True, sorted=True)
    thr = topk_vals[:, -1].view(B, 1, 1, 1)  # (B,1,1,1)

    mask = (cam01 >= thr).float()  # 1 for top-k region

    if mode == "deletion":
        # remove salient: top-k -> 0
        return x * (1.0 - mask)
    elif mode == "insertion":
        # keep only salient: top-k -> keep, rest -> 0
        return x * mask
    else:
        raise ValueError("mode must be 'deletion' or 'insertion'")


@torch.no_grad()
def auc_trapz(xs, ys) -> float:
    # xs: increasing in [0,1], ys: values
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    return float(np.trapz(ys, xs))


def gradcam_deletion_insertion_eval(
    model: nn.Module,
    cam_obj: GradCAM,
    loader: DataLoader,
    device,
    steps: int = 10,
    max_batches: int = 20,
):
    model.eval()
    delet_aucs, ins_aucs, conf_drops = [], [], []

    ratios = np.linspace(0.0, 1.0, steps + 1).tolist()

    batch_count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # -------------------------
        # (A) Grad-CAM must have grads
        # -------------------------
        with torch.enable_grad():
            logits = model(x)                 # has grad_fn now
            pred = logits.argmax(dim=1)
            cam = cam_obj(x, class_idx=pred)  # backward happens here
            cam_up = upsample_cam(cam, (x.size(2), x.size(3)))

        # -------------------------
        # (B) Masked forward passes do NOT need grads
        # -------------------------
        with torch.no_grad():
            base_probs = F.softmax(logits, dim=1).gather(1, pred.view(-1, 1)).squeeze(1)

            del_curve, ins_curve = [], []
            for r in ratios:
                x_del = apply_mask_topk(x, cam_up, keep_ratio=r, mode="deletion")
                p_del = F.softmax(model(x_del), dim=1).gather(1, pred.view(-1, 1)).squeeze(1)
                del_curve.append(p_del.mean().item())

                x_ins = apply_mask_topk(x, cam_up, keep_ratio=r, mode="insertion")
                p_ins = F.softmax(model(x_ins), dim=1).gather(1, pred.view(-1, 1)).squeeze(1)
                ins_curve.append(p_ins.mean().item())

            delet_aucs.append(auc_trapz(ratios, del_curve))
            ins_aucs.append(auc_trapz(ratios, ins_curve))

            x_del30 = apply_mask_topk(x, cam_up, keep_ratio=0.30, mode="deletion")
            p_del30 = F.softmax(model(x_del30), dim=1).gather(1, pred.view(-1, 1)).squeeze(1)
            conf_drops.append((base_probs.mean().item() - p_del30.mean().item()))

        batch_count += 1
        if batch_count >= max_batches:
            break

    return {
        "DeletionAUC_mean": float(np.mean(delet_aucs)),
        "InsertionAUC_mean": float(np.mean(ins_aucs)),
        "ConfDrop@30%_mean": float(np.mean(conf_drops)),
    }

def gradcam_deletion_insertion_eval_30(
    model: nn.Module,
    cam_obj: GradCAM,
    loader: DataLoader,
    device,
    max_batches: int = 20,
    r: float = 0.30,
):
    model.eval()
    del30_list, ins30_list = [], []

    batch_count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # -------------------------
        # (A) Grad-CAM must have grads
        # -------------------------
        with torch.enable_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            cam = cam_obj(x, class_idx=pred)  # backward happens here
            cam_up = upsample_cam(cam, (x.size(2), x.size(3)))

        # -------------------------
        # (B) Masked forward passes do NOT need grads
        # -------------------------
        with torch.no_grad():
            # baseline target probs for predicted class
            base_probs = F.softmax(logits, dim=1).gather(1, pred.view(-1, 1)).squeeze(1)

            # Deletion @ r: remove top-r salient pixels
            x_del = apply_mask_topk(x, cam_up, keep_ratio=r, mode="deletion")
            p_del = F.softmax(model(x_del), dim=1).gather(1, pred.view(-1, 1)).squeeze(1)
            del30_list.append((base_probs.mean().item() - p_del.mean().item()))

            # Insertion @ r: keep only top-r salient pixels
            x_ins = apply_mask_topk(x, cam_up, keep_ratio=r, mode="insertion")
            p_ins = F.softmax(model(x_ins), dim=1).gather(1, pred.view(-1, 1)).squeeze(1)
            ins30_list.append(p_ins.mean().item())

        batch_count += 1
        if batch_count >= max_batches:
            break

    return {
        "Deletion@30%_mean": float(np.mean(del30_list)),  # (= ConfDrop@30%)
        "Insertion@30%_mean": float(np.mean(ins30_list)),
    }

def make_orthogonal(C: int, device: str):
    A = torch.randn(C, C, device=device)
    Q, R = torch.linalg.qr(A)
    # fix sign (optional)
    diag = torch.sign(torch.diag(R))
    Q = Q * diag
    return Q  # (C,C)

class ChannelPurLockObfuscator(nn.Module):
    """
    For feature maps f: (B,C,H,W),
    applies per-location channel transform:
        f' = shuffle( (f @ Q^T) * sigma )
    where Q is orthogonal (C,C), sigma is diagonal reweight (C,).
    """
    def __init__(self, C: int, use_shuffle: bool = True, learn_sigma: bool = True):
        super().__init__()
        self.C = C
        self.use_shuffle = use_shuffle

        # secret orthogonal key
        self.register_buffer("Q", torch.empty(C, C))

        # diagonal reweighting
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(C))
        else:
            self.register_buffer("log_sigma", torch.zeros(C))

        if use_shuffle:
            perm = torch.randperm(C)
            inv_perm = torch.argsort(perm)
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)
        else:
            self.perm = None
            self.inv_perm = None

    @torch.no_grad()
    def init_key(self, device=None):
        device = device or self.Q.device
        self.Q.copy_(make_orthogonal(self.C, device=device))

    def forward(self, f_map: torch.Tensor):
        """
        f_map: (B,C,H,W)
        """
        B, C, H, W = f_map.shape
        assert C == self.C

        # (B,C,H,W) -> (B,H,W,C)
        x = f_map.permute(0, 2, 3, 1).contiguous()

        # orthogonal in channel space: (B,H,W,C) @ (C,C)
        x = torch.matmul(x, self.Q.transpose(0, 1))

        # diagonal reweighting (positive)
        sigma = F.softplus(self.log_sigma) + 1e-6  # (C,)
        x = x * sigma.view(1, 1, 1, C)

        # channel shuffle
        if self.use_shuffle:
            x = x[..., self.perm]

        # back to (B,C,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def _cifar10_denormalize(img_chw: np.ndarray,
                         mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)) -> np.ndarray:
    """
    img_chw: (3,H,W) numpy in normalized space
    return:  (H,W,3) numpy in [0,1]
    """
    mean = np.asarray(mean).reshape(3, 1, 1)
    std  = np.asarray(std).reshape(3, 1, 1)
    img = img_chw * std + mean
    img = np.clip(img, 0.0, 1.0)
    return np.transpose(img, (1, 2, 0))


def save_gradcam_example(
    pipeline: torch.nn.Module,
    cam_obj,
    loader,
    device,
    out_path: str = "unauthorized_gradcam_example.pdf",
    max_batches: int = 20,
    alpha: float = 0.5,
    target: str = "pred",                 # "pred" or "label"
    use_opencv: bool = False,             # if True, uses cv2 colormap; else matplotlib colormap
    cmap: str = "jet",
    title_left: str = "Input Image",
    title_right: str = "Grad-CAM",
    dpi: int = 300,
):
    """
    Save a paper-quality figure: [input image | Grad-CAM overlay].

    Args:
        pipeline: UnauthPipeline (x -> logits)
        cam_obj:  GradCAM object with call cam_obj(x, class_idx=...)
        loader:   DataLoader yielding (x,y) where x is normalized CIFAR-10 tensor
        device:   "cuda" or "cpu"
        out_path: output .png path
        max_batches: search up to this many batches to pick a sample
        alpha:    overlay strength (0..1). overlay = (1-alpha)*img + alpha*heatmap
        target:   "pred" uses predicted class; "label" uses ground-truth label
        use_opencv: if True uses cv2 COLORMAP_JET; otherwise matplotlib colormap
        cmap:     matplotlib colormap name (ignored if use_opencv=True)
        dpi:      saved dpi

    Returns:
        dict with { "out_path", "pred", "label" } for the chosen sample.
    """
    assert target in ("pred", "label")

    pipeline.eval()

    chosen = None
    batch_count = 0

    # pick the first sample (or you can extend this to pick "best/worst" by some score)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = pipeline(x)
            pred = logits.argmax(dim=1)

            # pick index 0 from this batch
            i = 12
            chosen = {
                "x1": x[i:i+1],
                "y1": y[i].item(),
                "pred1": pred[i].item(),
            }
            break

            batch_count += 1
            if batch_count >= max_batches:
                break

    if chosen is None:
        raise RuntimeError("No sample found in loader.")

    x1 = chosen["x1"]  # (1,3,32,32)
    y1 = chosen["y1"]
    p1 = chosen["pred1"]

    # choose class index for CAM
    class_idx = torch.tensor([p1], device=device) if target == "pred" else torch.tensor([y1], device=device)

    # Grad-CAM requires grads
    with torch.enable_grad():
        cam_map = cam_obj(x1, class_idx=class_idx)  # (1,1,h,w), expected in [0,1]

    # upsample CAM to input size
    cam_up = F.interpolate(cam_map, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=False)
    heat = cam_up[0, 0].cpu().numpy()  # [0,1]
    thr = np.quantile(heat, 0.70)  # top 15%
    mask = (heat >= thr).astype(np.float32)

    # 只在显著区域显示热图
    heat = heat * mask

    # denormalize image for visualization
    img = x1[0].detach().cpu().numpy()  # (3,32,32)
    img = _cifar10_denormalize(img)     # (32,32,3) in [0,1]

    # convert heatmap to RGB
    if use_opencv:
        import cv2
        heat_u8 = np.uint8(255 * heat)
        hm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        hm = hm.astype(np.float32) / 255.0
    else:
        cm = plt.get_cmap(cmap)
        hm = cm(heat)[..., :3].astype(np.float32)  # (H,W,3)

    overlay = (1 - alpha) * img + alpha * hm
    overlay = np.clip(overlay, 0.0, 1.0)

    # make output folder if needed
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # plot and save
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(title_left)

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title_right)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    # plt.imsave("figs/unauthor_original_DP.pdf", img)
    plt.imsave("figs/unauthor_gradcam_overlay_purlock.pdf", overlay)
    plt.close()

    return {"out_path": out_path, "pred": p1, "label": y1}

def save_ckpt(path, pipeline, optimizer=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "pipeline_state": pipeline.state_dict()
    }
    if optimizer is not None:
        ckpt["optim_state"] = optimizer.state_dict()
    torch.save(ckpt, path)
    print(f"✅ Saved checkpoint to {path})")


def load_ckpt(path, pipeline, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    pipeline.load_state_dict(ckpt["pipeline_state"], strict=True)
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    print(f"✅ Loaded checkpoint from {path} (epoch={ckpt.get('epoch')}, best_score={ckpt.get('best_score')})")
    return ckpt

# -----------------------------
# 5) Main: train + gradcam eval
# -----------------------------
def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # CIFAR-10 transforms
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    # train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    # test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_set = SVHN(root='./data', split='train', download=True, transform=train_tf)
    test_set = SVHN(root='./data', split='test', download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCNN(num_classes=10).to(device)
    extractor = CNN5_Extractor().to(device)

    ckpt_path = "checkpoints/cnn5_pretrained.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        load_by_suffix(extractor, state_dict, verbose=True)
        print(f"✅ loaded pretrained extractor from {ckpt_path}")
    else:
        print(f"⚠️ checkpoint not found: {ckpt_path} (will use random conv/bn)")


    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)  # 自动替换/包裹不兼容模块（尽量保留功能）
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    max_grad_norm = 1.0
    noise_multiplier = 3.0
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    # quick training (increase epochs for better accuracy)


    epochs = 20
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_acc = eval_acc(model, test_loader, device)
        print(f"Epoch {ep:02d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | test acc={te_acc:.4f}")
    # save_ckpt("checkpoints/unauth_cnn_cifar10_DP.pth", model)
    # load_ckpt("checkpoints/unauth_cnn_cifar10_baseline.pth", model)

    if hasattr(model, "disable_hooks"):
        model.disable_hooks()
    # Grad-CAM on conv4

    # extractor.eval()
    # for p in extractor.parameters():
    #     p.requires_grad = False

    # -----------------------------
    # (2) PurLock obfuscation on fe_map (C=64)
    # -----------------------------
    # If you want a baseline run (no obfuscation), replace obf_fe with nn.Identity()
    # obf_fe = ChannelPurLockObfuscator(C=64, use_shuffle=True, learn_sigma=False).to(device)
    # obf_fe.init_key(device=device)

    # freeze obfuscator (attacker cannot learn key/sigma)
    # obf_fe.eval()
    # for p in obf_fe.parameters():
    #     p.requires_grad = False

    # -----------------------------
    # (3) Attacker head (unauthorized classifier)
    # -----------------------------
    # unauth_head = UnauthClassifierFromFeMap(inC=64, num_classes=10).to(device)

    # Build pipeline: x -> extractor(fe_map) -> obf_fe -> unauth_head -> logits
    # pipeline = UnauthPipeline(extractor, obf_fe, unauth_head).to(device)

    # Only train unauth_head parameters
    # optimizer = optim.Adam(pipeline.unauth_head.parameters(), lr=1e-3, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # (4) Train attacker head (sanity: it should still reach some acc)
    # -----------------------------
    # epochs = 10  # you can increase to 20 for stronger attacker
    # for ep in range(1, epochs + 1):
    #     pipeline.train()  # safe: extractor/obf_fe frozen so won't update
    #     total_loss, total, correct = 0.0, 0, 0
    #
    #     for x, y in train_loader:
    #         x, y = x.to(device), y.to(device)
    #
    #         optimizer.zero_grad(set_to_none=True)
    #         logits = pipeline(x)
    #         loss = criterion(logits, y)
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item() * x.size(0)
    #         pred = logits.argmax(dim=1)
    #         correct += (pred == y).sum().item()
    #         total += x.size(0)
    #
    #     train_loss = total_loss / total
    #     train_acc = correct / total
    #
    #     # quick eval acc
    #     pipeline.eval()
    #     with torch.no_grad():
    #         total_t, correct_t = 0, 0
    #         for x, y in test_loader:
    #             x, y = x.to(device), y.to(device)
    #             logits = pipeline(x)
    #             pred = logits.argmax(dim=1)
    #             correct_t += (pred == y).sum().item()
    #             total_t += x.size(0)
    #         test_acc = correct_t / total_t
    #
    #     print(
    #         f"Epoch {ep:02d} | unauth train loss={train_loss:.4f} acc={train_acc:.4f} | unauth test acc={test_acc:.4f}")

    # -----------------------------
    # (5) Grad-CAM on attacker head convB (unauthorized localization)
    # -----------------------------
    # pipeline.eval()

    # cam = GradCAM(pipeline, target_layer=pipeline.unauth_head.convB)
    cam = GradCAM(model, target_layer=model.conv4)

    # Faithfulness eval (no GT required)
    metrics = gradcam_deletion_insertion_eval_30(
        model=model,
        cam_obj=cam,
        loader=test_loader,
        device=device,
        max_batches=20,   # evaluate on first 20 batches for speed
    )
    print("\nGrad-CAM evaluation (faithfulness, no GT):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    cam.close()
    # info = save_gradcam_example(
    #     pipeline=pipeline,
    #     cam_obj=cam,
    #     loader=test_loader,
    #     device=device,
    #     out_path="figs/unauth_gradcam_example_purlock.png",
    #     alpha=0.5,
    #     target="pred",  # or "label"
    # )
    #
    # print("Saved Grad-CAM example:", info)

if __name__ == "__main__":
    main()
