import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torchvision.datasets import SVHN
import time
from opacus.utils.batch_memory_manager import BatchMemoryManager


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_dataloaders(dataset="cifar10", batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "cifar10":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == "svhn":
        trainset = SVHN(root='./data', split='train', download=True, transform=transform)
        testset = SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average='macro')

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc = 0.0

    # Compute TPR and FPR
    cm = confusion_matrix(all_labels, all_preds, labels=range(10))
    TP = cm.diagonal()
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    TPR = (TP / (TP + FN + 1e-10)).mean()  # Avoid divide-by-zero
    FPR = (FP / (FP + TN + 1e-10)).mean()

    return acc, f1, auc, TPR, FPR


def main():
    start_time = time.time()
    results = []
    dataset = "svhn"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloaders(dataset=dataset)

    model = SimpleCNN()
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=1,
        target_delta=1e-5,
        epochs=20,
        max_grad_norm=1,
    )

    # print(f"DP training: ε={privacy_engine.get_epsilon(1e-5):.2f}, δ=1e-5")
    print(f"⏱️ Processing time: {time.time() - start_time:.2f}s")

    for epoch in range(1, 21):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        # test_acc = evaluate(model, testloader, device)
        epsilon = privacy_engine.get_epsilon(1e-5)
        acc, f1_val, auc, tpr, fpr = evaluate(model, testloader, device)
        results.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Test Acc": acc,
            "Test F1": f1_val,
            "Test AUC": auc,
            "TPR": tpr,
            "FPR": fpr
        })
        print(
            f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1_val:.4f} | AUC: {auc:.4f} | TPR: {tpr:.4f} | FPR: {fpr:.4f}")

    print(f"⏱️ Training time: {time.time() - start_time:.2f}s")
    print("\n📊 Metrics for Visualization:")
    print("loss_list =", [round(r["Train Loss"], 4) for r in results])
    print("acc_list =", [round(r["Test Acc"], 4) for r in results])
    print("f1_list =", [round(r["Test F1"], 4) for r in results])
    print("auc_list =", [round(r["Test AUC"], 4) for r in results])
    print("tpr_list =", [round(r["TPR"], 4) for r in results])
    print("fpr_list =", [round(r["FPR"], 4) for r in results])
    print(f"time = {round(time.time() - start_time, 2)}  # seconds")
if __name__ == "__main__":
    main()
