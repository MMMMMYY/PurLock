import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import pickle
from transformers import BertModel, DistilBertModel, T5EncoderModel

class FeatureExtractor(nn.Module):
    def __init__(self, model_type="bert", fe_layer=-1, fi_layer=-2):
        """
        Args:
            model_type (str):  'bert', 'distilbert', 't5'
            fe_layer (int): f_e
            fi_layer (int): f_i
        """
        super().__init__()
        self.model_type = model_type
        self.fe_layer = fe_layer
        self.fi_layer = fi_layer

        if model_type == "bert":
            self.encoder = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        elif model_type == "distilbert":
            self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
        elif model_type == "t5":
            self.encoder = T5EncoderModel.from_pretrained("t5-small")
        else:
            raise ValueError("Unsupported model type")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.model_type == "t5":
            f_e = outputs.last_hidden_state[:, 0, :]  # token 0
            f_i = outputs.last_hidden_state[:, 1, :]  # token 1
        else:
            hidden = outputs.hidden_states  # List of hidden states from all layers
            f_e = hidden[self.fe_layer][:, 0, :]  # CLS token of fe layer
            f_i = hidden[self.fi_layer][:, 0, :]  # CLS token of fi layer

        return f_e, f_i



# ====================== Step 2: QR-based Perturbation ======================
def generate_orthogonal_matrix(dim):
    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return q

def apply_orthogonal_perturbation(x, Q):
    Q = Q.to(x.device)
    return x @ Q


# ====================== Step 3: Purification Modules ======================
class DiagonalPurification(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.register_buffer('inv_sigma', 1.0 / (sigma + 1e-8))

    def forward(self, x):
        return x * self.inv_sigma

class ProgressiveLowRankPurification(nn.Module):
    def __init__(self, in_dim, rank=128):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, in_dim, bias=False)

    def forward(self, x):
        return self.V(self.U(x))

# ====================== Step 4: Task-specific Module ======================
class TaskModel(nn.Module):
    def __init__(self, f_e_dim=768, f_i_dim=768, num_classes=2):
        super().__init__()
        hidden_dim = f_i_dim
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.purifier_e = None
        self.purifier_i = None

    def set_purifiers(self, purifier_e, purifier_i):
        self.purifier_e = purifier_e
        self.purifier_i = purifier_i

    def forward(self, f_e, f_i):
        if self.purifier_e:
            f_e = self.purifier_e(f_e)
        if self.purifier_i:
            f_i = self.purifier_i(f_i)
        if f_e.shape[1] != f_i.shape[1]:
            raise ValueError(f"Shape mismatch: fe={f_e.shape}, fi={f_i.shape}")
        x = f_e + f_i
        return self.linear(x)


# ====================== Preprocessing Dataset ======================
def preprocess_data(tokenizer, dataset, max_len=128):
    def tokenize(example):
        return tokenizer(example['text'], padding="max_length", truncation=True, max_length=max_len)
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset

# ====================== Training & Evaluation ======================
def train_and_evaluate(dataset_name="imdb", model_type ="t5"):
    device = torch.device("cuda:0")

    # Load data
    dataset = load_dataset(dataset_name)
    tokenizer_name = {
        "bert": "bert-base-uncased",
        "distilbert": "distilbert-base-uncased",
        "t5": "t5-small"
    }[model_type]

    if model_type == "t5":
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = preprocess_data(tokenizer, dataset)

    train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset["test"], batch_size=64)
    start_time = time.time()

    # Initialize modules
    extractor = FeatureExtractor(model_type=model_type).to(device)
    # task_model = TaskModel().to(device)

    # Prepare for feature extraction
    extractor.eval()
    f_e_all, f_i_all, labels_all = [], [], []

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            f_e, f_i = extractor(input_ids, attention_mask)
            f_e_dim = f_e.size(1)
            f_i_dim = f_i.size(1)
            f_e_all.append(f_e)
            f_i_all.append(f_i)
            labels_all.append(labels)

    f_e_all = torch.cat(f_e_all, dim=0)
    f_i_all = torch.cat(f_i_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    task_model = TaskModel(f_e_dim=f_e_dim, f_i_dim=f_i_dim).to(device)

    # Apply QR-based perturbation
    Q = generate_orthogonal_matrix(f_e_all.size(1))
    Q = Q.to(f_e_all.device)
    f_e_perturbed = apply_orthogonal_perturbation(f_e_all, Q)

    Q_i = generate_orthogonal_matrix(f_i_all.size(1))
    Q_i = Q_i.to(f_i_all.device)
    f_i_perturbed = apply_orthogonal_perturbation(f_i_all, Q_i)

    # Compute σ and set purification
    sigma = torch.std(f_e_perturbed, dim=0)
    purifier_e = DiagonalPurification(sigma).to(device)

    purifier_i = ProgressiveLowRankPurification(f_i_perturbed.size(1)).to(device)

    task_model.set_purifiers(purifier_e, purifier_i)
    print(f"\n⏱️ processing time: {time.time() - start_time:.2f}s")
    # Train with purified features
    optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    task_model.train()

    results = []
    for epoch in range(1, 10):
        total_loss = 0
        for i in range(0, len(f_e_perturbed), 64):
            fe = f_e_perturbed[i:i+64].to(device)
            fi = f_i_perturbed[i:i+64].to(device)
            labels = labels_all[i:i+64].to(device)

            optimizer.zero_grad()
            output = task_model(fe, fi)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        task_model.eval()
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)  # ✅ 只使用 test 的 labels

                f_e, f_i = extractor(input_ids, attention_mask)
                f_e = apply_orthogonal_perturbation(f_e.cpu(), Q).to(device)
                f_i = apply_orthogonal_perturbation(f_i.cpu(), Q_i).to(device)

                outputs = task_model(f_e, f_i)
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())  # ✅ 用的是 test 的 label

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='macro')
        try:
            auc = roc_auc_score(trues, np.array(probs)[:, 1])
        except:
            auc = 0.0

        cm = confusion_matrix(trues, preds, labels=[0, 1])
        TP = cm.diagonal()
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        TPR = (TP / (TP + FN + 1e-10)).mean()
        FPR = (FP / (FP + TN + 1e-10)).mean()

        results.append({
            "Epoch": epoch,
            "Train Loss": total_loss / len(f_e_perturbed),
            "Test Acc": acc,
            "Test F1": f1,
            "Test AUC": auc,
            "TPR": TPR,
            "FPR": FPR
        })

        print(f"Epoch {epoch:02d} | Loss: {results[-1]['Train Loss']:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | TPR: {TPR:.4f} | FPR: {FPR:.4f}")
    y_true, y_pred, y_prob = [], [], []
    task_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            f_e, f_i = extractor(input_ids, attention_mask)
            f_e = apply_orthogonal_perturbation(f_e.cpu(), Q).to(device)
            f_i = apply_orthogonal_perturbation(f_i.cpu(), Q_i).to(device)

            outputs = task_model(f_e, f_i)
            prob = F.softmax(outputs, dim=1)

            y_pred.extend(prob.argmax(dim=1).cpu().numpy())  # ✅ 只添加一次
            y_true.extend(labels.cpu().numpy())
            y_prob.append(prob.cpu().numpy())


    y_prob = np.vstack(y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.shape[1] == 2 else 0.0

    print(f"\n✅ ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {round(time.time() - start_time, 2)}s")
    save_path = "bert_ablation_rank128.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob
        }, f)

    print("\n📊 Metrics for Visualization:")
    print("loss_list =", [round(r["Train Loss"], 4) for r in results])
    print("acc_list =", [round(r["Test Acc"], 4) for r in results])
    print("f1_list =", [round(r["Test F1"], 4) for r in results])
    print("auc_list =", [round(r["Test AUC"], 4) for r in results])
    print("tpr_list =", [round(r["TPR"], 4) for r in results])
    print("fpr_list =", [round(r["FPR"], 4) for r in results])
    print(f"time = {round(time.time() - start_time, 2)}  # seconds")
if __name__ == "__main__":
    train_and_evaluate("rotten_tomatoes", model_type="bert")
