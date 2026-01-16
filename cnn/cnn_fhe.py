import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import time

from concrete.ml.torch.compile import compile_torch_model


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))     # -> [B, 6, 16, 16]
        x = x.view(x.size(0), -1)               # flatten -> [B, 1536]
        x = F.relu(self.fc1(x))                 # -> [B, 64]
        x = self.fc2(x)                         # -> [B, 10]
        return x


def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].float()


def train_model(model, trainloader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(5):
        total_loss = 0
        for inputs, labels in tqdm(trainloader, desc=f"[Epoch {epoch+1}]"):
            inputs = inputs.to(device)
            targets = one_hot(labels).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"  Avg Loss: {total_loss / len(trainloader):.4f}")


def compile_fhe_model(model, loader):
    print("[🔐] Compiling model into FHE...")
    model.eval()
    model.cpu()

    for x, _ in loader:
        sample_input = x[0].unsqueeze(0).cpu()
        break

    quantized_module = compile_torch_model(model, sample_input, n_bits=3)


    circuit = quantized_module.compile(sample_input.numpy())

    return circuit  # ⬅️ 返回的是 circuit，而不是 quantized_module


def fhe_inference(fhe_circuit, testloader):
    correct = 0
    total = 0
    for x, y in tqdm(testloader, desc="🔐 FHE Inference"):
        x_np = (x.numpy() * 7).astype("uint8")
        y_true = y.item()

        y_pred = fhe_circuit.encrypt_run_decrypt(x_np)

        if int(np.argmax(y_pred[0])) == y_true:
            correct += 1
        total += 1
        if total >= 20:  # quick test
            break
    print(f"✅ FHE Accuracy on {total} samples: {correct/total:.4f}")

def main():
    start_time = time.time()
    device = torch.device("cpu")

    transform = ToTensor()
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    print(f"⏱️ Processing time: {time.time() - start_time:.2f}s")
    model = SimpleCNN().to(device)
    train_model(model, trainloader, device)

    fhe_model = compile_fhe_model(model, trainloader)
    fhe_inference(fhe_model, testloader)


if __name__ == "__main__":
    main()
