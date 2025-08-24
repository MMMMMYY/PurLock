# Complete code for PURLocker with both TinyCNN and ResNet18 architectures
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.models import resnet18
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import time
import os


# ---------------------- Low-rank Layers ----------------------
class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.V = nn.Linear(in_dim, rank, bias=False)
        self.U = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x):
        return self.U(self.V(x))


# ---------------------- TinyCNN ----------------------
class TinyCNN_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        fe = self.pool(F.relu(self.conv1(x)))
        fi = self.pool(F.relu(self.conv2(fe)))
        return fe, fi


class TinyCNN_TaskModel(nn.Module):
    def __init__(self, input_dim, num_classes=10, rank=16):
        super().__init__()
        self.L_e = LowRankLinear(input_dim, input_dim, rank)
        self.L_i1 = LowRankLinear(input_dim, 256, rank)
        self.L_i2 = LowRankLinear(256, num_classes, rank)
        self.dropout = nn.Dropout(0.5)

    def forward(self, fe, fi):
        fe = self.L_e(fe)
        x = fe + fi
        x = F.relu(self.L_i1(x))
        x = self.dropout(x)
        return self.L_i2(x)


# ---------------------- ResNet18 ----------------------
class ResNet18_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        self.layer1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer2 = base.layer1
        self.layer3 = base.layer2

    def forward(self, x):
        x = self.layer1(x)
        fe = self.layer2(x)
        fi = self.layer3(fe)
        return fe, fi


class ResNet18_TaskModel(nn.Module):
    def __init__(self, input_dim, num_classes=10, rank=32):
        super().__init__()
        self.L_e = LowRankLinear(input_dim, input_dim, rank)
        self.L_i1 = LowRankLinear(input_dim, 256, rank)
        self.L_i2 = LowRankLinear(256, num_classes, rank)
        self.dropout = nn.Dropout(0.5)

    def forward(self, fe, fi):
        fe = self.L_e(fe)
        x = fe + fi
        x = F.relu(self.L_i1(x))
        x = self.dropout(x)
        return self.L_i2(x)


# ---------------------- Utility Functions ----------------------
def generate_orthogonal_matrix(dim):
    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return q


def dimension_shuffle(feat):
    B, C, H, W = feat.shape
    sigma = torch.std(feat.view(B, C, -1), dim=2)
    scale = sigma.view(B, C, 1, 1)
    return feat * scale, sigma.mean(dim=0)


def diagonal_purification(layer, sigma):

    if sigma.dim() == 2:
        sigma = sigma.mean(dim=0)
    elif sigma.dim() == 0:
        raise ValueError("sigma should be a 1D vector")

    in_dim = layer.V.in_features
    if sigma.size(0) != in_dim:
        raise ValueError(f"sigma dimension {sigma.size(0)} does not match V.in_features {in_dim}")

    diag_inv = torch.diag(1.0 / sigma.to(layer.V.weight.device))

    with torch.no_grad():
        layer.V.weight.data = layer.V.weight.data @ diag_inv

    return layer





def low_rank_purification(layer, Q, rank):
    with torch.no_grad():
        W_eff = layer.U.weight @ layer.V.weight
        W_purified = W_eff @ Q.T
        U, S, Vh = torch.linalg.svd(W_purified, full_matrices=False)
        purified = LowRankLinear(layer.V.in_features, layer.U.out_features, rank)
        purified.U.weight.copy_(U[:, :rank] @ torch.diag(S[:rank]))
        purified.V.weight.copy_(Vh[:rank, :])
    return purified


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for fe, fi, labels in dataloader:
            fe, fi, labels = fe.to(device), fi.to(device), labels.to(device)
            out = model(fe, fi)
            prob = F.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_probs.append(prob.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    f1 = f1_score(labels, preds, average='macro')
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr')
    except:
        auc = 0.0

    cm = confusion_matrix(labels, preds, labels=range(10))
    TP = cm.diagonal()
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    TN = cm.sum() - (FP + FN + TP)
    TPR = (TP / (TP + FN + 1e-10)).mean()
    FPR = (FP / (FP + TN + 1e-10)).mean()
    return acc, f1, auc, TPR, FPR


# ---------------------- Dataset ----------------------
def get_dataloaders(name="cifar10", batch_size=128):
    transform = transforms.ToTensor()
    if name == "cifar10":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name == "svhn":
        trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    return DataLoader(trainset, batch_size=batch_size, shuffle=True), DataLoader(testset, batch_size=batch_size)


# ---------------------- Feature Caching ----------------------
def cache_features(model, loader, device):
    model.eval()
    fe_list, fi_list, labels_list = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            fe, fi = model(x)
            B = fe.size(0)
            fe = fe.view(B, -1)
            fi = fi.view(B, -1)
            Q1 = generate_orthogonal_matrix(fe.size(1)).to(device)
            Q2 = generate_orthogonal_matrix(fi.size(1)).to(device)
            fe = fe @ Q1
            fi = fi @ Q2
            try:
                fe = fe.view(B, 32, 16, 16)
                fe, sigma = dimension_shuffle(fe)
                fe = fe.view(B, -1)
            except:
                sigma = torch.std(fe, dim=0)

            fe_list.append(fe.cpu())
            fi_list.append(fi.cpu())
            labels_list.append(y.cpu())

    return TensorDataset(torch.cat(fe_list), torch.cat(fi_list),
                         torch.cat(labels_list)), sigma.cpu(), Q2.cpu(), fi.size(1)


# ---------------------- Main ----------------------
def main():
    start_time = time.time()
    results = []
    model_type = "resnet18"  # or "resnet18"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 32
    trainloader, testloader = get_dataloaders("cifar10")

    if model_type == "tinycnn":
        extractor = TinyCNN_FeatureExtractor().to(device)
    else:
        extractor = ResNet18_FeatureExtractor().to(device)

    dataset, sigma, Q, input_dim = cache_features(extractor, trainloader, device)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    if model_type == "tinycnn":
        model = TinyCNN_TaskModel(input_dim).to(device)
    else:
        model = ResNet18_TaskModel(input_dim).to(device)

    model.L_e = diagonal_purification(model.L_e, sigma.to(device))
    model.L_i1 = low_rank_purification(model.L_i1, Q.to(device)[:input_dim, :input_dim], rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    os.makedirs("checkpoints", exist_ok=True)
    print(f"⏱️ processing time: {time.time() - start_time:.2f}s")
    for epoch in range(1, 11):
        model.train()
        for fe, fi, y in loader:
            fe, fi, y = fe.to(device), fi.to(device), y.to(device)
            out = model(fe, fi)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc, f1, auc, tpr, fpr = evaluate(model, loader, device)
        results.append({
            "Epoch": epoch,
            "Test Acc": acc,
            "Test F1": f1,
            "Test AUC": auc,
            "TPR": tpr,
            "FPR": fpr
        })
        print(f"Epoch {epoch:02d} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | TPR: {tpr:.4f} | FPR: {fpr:.4f}")
    print(f"\n⏱️ Training time: {time.time() - start_time:.2f}s")
    print("\n📊 Metrics for Visualization:")
    print("acc_list =", [round(r["Test Acc"], 4) for r in results])
    print("f1_list =", [round(r["Test F1"], 4) for r in results])
    print("auc_list =", [round(r["Test AUC"], 4) for r in results])
    print("tpr_list =", [round(r["TPR"], 4) for r in results])
    print("fpr_list =", [round(r["FPR"], 4) for r in results])
    print(f"time = {round(time.time() - start_time, 2)}  # seconds")

if __name__ == "__main__":
    main()
