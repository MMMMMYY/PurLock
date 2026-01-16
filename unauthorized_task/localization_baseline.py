import os
import math
import numpy as np
from typing import Tuple, Dict
from torchvision.datasets import SVHN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

import os
import matplotlib.pyplot as plt
from localization_purlock_DP import _cifar10_denormalize


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
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

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

        # alpha_k^c = mean_{i,j} dY/dA_k(i,j)
        alpha = dYdA.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)

        # cam = ReLU(sum_k alpha_k * A_k)
        cam = (alpha * A).sum(dim=1, keepdim=True)   # (B,1,H,W)
        cam = F.relu(cam)

        cam = self._normalize_cam(cam.detach())
        return cam

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


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
    thr = np.quantile(heat, 0.7)  # top 15%
    mask = (heat >= thr).astype(np.float32)

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
    plt.imsave("figs/unauthor_gradcam_overlay_baseline.png", overlay)
    plt.close()

    return {"out_path": out_path, "pred": p1, "label": y1}

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # quick training (increase epochs for better accuracy)
    epochs = 5
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_acc = eval_acc(model, test_loader, device)
        print(f"Epoch {ep:02d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | test acc={te_acc:.4f}")

    # save_ckpt("checkpoints/unauth_cnn_cifar10_baseline_new.pth", model)
    # load_ckpt("checkpoints/unauth_cnn_cifar10_baseline_new.pth", model)
    # Grad-CAM on conv4
    cam = GradCAM(model, target_layer=model.conv4)

    # Faithfulness eval (no GT required)
    metrics = gradcam_deletion_insertion_eval_30(
        model=model,
        cam_obj=cam,
        loader=test_loader,
        device=device,
        max_batches=20,  # evaluate on first 20 batches for speed
    )
    print("\nGrad-CAM evaluation (faithfulness, no GT):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    cam.close()

    # info = save_gradcam_example(
    #     pipeline=model,
    #     cam_obj=cam,
    #     loader=test_loader,
    #     device=device,
    #     out_path="figs/unauth_gradcam_example_baseline_new.png",
    #     alpha=0.5,
    #     target="pred",  # or "label"
    # )
    #
    # print("Saved Grad-CAM example:", info)


if __name__ == "__main__":
    main()
