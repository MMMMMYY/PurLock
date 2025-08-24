import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
import time
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import numpy as np


def load_and_prepare_dataset(dataset_choice="rotten_tomatoes"):
    print(f"📦 Loading dataset: {dataset_choice}")
    dataset = load_dataset(dataset_choice)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(encoded_dataset["train"], batch_size=16, shuffle=True)
    test_loader = DataLoader(encoded_dataset["test"], batch_size=32)
    return train_loader, test_loader


def add_laplace_noise_to_embeddings(model, scale=0.01):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'embeddings' in name and 'weight' in name:
                noise = torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=param.size())).float().to(param.device)
                param.add_(noise)


def train(model, train_loader, optimizer, device, noise_scale=0.01):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        add_laplace_noise_to_embeddings(model, scale=noise_scale)

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average='macro')

    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])  # binary
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=range(2))
    TP = cm.diagonal()
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    TPR = (TP / (TP + FN + 1e-10)).mean()
    FPR = (FP / (FP + TN + 1e-10)).mean()

    return acc, f1, auc, TPR, FPR


def main(dataset_choice="rotten_tomatoes", noise_scale=0.01):
    train_loader, test_loader = load_and_prepare_dataset(dataset_choice)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    results = []
    start_time = time.time()
    for epoch in range(1, 4):
        train_loss, train_acc = train(model, train_loader, optimizer, device, noise_scale)
        acc, f1_val, auc, tpr, fpr = evaluate(model, test_loader, device)
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

    print(f"\n⏱️ Training time: {time.time() - start_time:.2f}s")
    print("\n📊 Metrics for Visualization:")
    print("loss_list =", [round(r["Train Loss"], 4) for r in results])
    print("acc_list =", [round(r["Test Acc"], 4) for r in results])
    print("f1_list =", [round(r["Test F1"], 4) for r in results])
    print("auc_list =", [round(r["Test AUC"], 4) for r in results])
    print("tpr_list =", [round(r["TPR"], 4) for r in results])
    print("fpr_list =", [round(r["FPR"], 4) for r in results])
    print(f"time = {round(time.time() - start_time, 2)}  # seconds")


if __name__ == "__main__":
    main(dataset_choice="rotten_tomatoes", noise_scale=0.01)
