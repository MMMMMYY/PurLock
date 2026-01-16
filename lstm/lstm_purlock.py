import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import time


class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size=20):
        self.X, self.y = [], []
        for i in range(len(series) - window_size):
            self.X.append(series[i:i + window_size])
            self.y.append(series[i + window_size])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMExtractor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        f_i = out[:, -1, :]
        f_e = self.proj(out[:, 0, :])
        return f_e, f_i


def generate_orthogonal_matrix(dim):
    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return q

def apply_orthogonal_perturbation(x, Q):
    return x @ Q.to(x.device)

class DiagonalPurification(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.register_buffer('inv_sigma', 1.0 / (sigma + 1e-8))

    def forward(self, x):
        return x * self.inv_sigma

class ProgressiveLowRankPurification(nn.Module):
    def __init__(self, in_dim, rank=64):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, in_dim, bias=False)

    def forward(self, x):
        return self.V(self.U(x))



class RNNExtractor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        f_i = out[:, -1, :]
        f_e = self.proj(out[:, 0, :])
        return f_e, f_i





class TaskModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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
        x = f_e + f_i
        return self.linear(x)


def load_stock_data(ticker="AAPL", start="2010-01-01", end="2022-01-01"):
    df = yf.download(ticker, start=start, end=end)
    return df[["Close"]]


def evaluate_model(extractor, task_model, dataloader, scaler, device, Q_e, Q_i):
    extractor.eval()
    task_model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            f_e, f_i = extractor(X)
            f_e = apply_orthogonal_perturbation(f_e, Q_e)
            f_i = apply_orthogonal_perturbation(f_i, Q_i)
            out = task_model(f_e, f_i).cpu().numpy().reshape(-1, 1)
            y = y.cpu().numpy().reshape(-1, 1)
            preds.append(out)
            actuals.append(y)

    preds = scaler.inverse_transform(np.vstack(preds))
    actuals = scaler.inverse_transform(np.vstack(actuals))
    mse = mean_squared_error(actuals, preds)
    nmse = mse / np.var(actuals)
    return mse, nmse



def main():
    results = []
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_stock_data()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    split = int(0.8 * len(scaled))
    train_data, test_data = scaled[:split], scaled[split:]
    train_set = TimeSeriesDataset(train_data)
    test_set = TimeSeriesDataset(test_data)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    extractor = LSTMExtractor().to(device)

    # extractor = RNNExtractor().to(device)
    task_model = TaskModel(feat_dim=64).to(device)


    extractor.eval()
    f_e_all, f_i_all, y_all = [], [], []
    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(device)
            f_e, f_i = extractor(X)
            f_e_all.append(f_e)
            f_i_all.append(f_i)
            y_all.append(y)
    f_e_all = torch.cat(f_e_all)
    f_i_all = torch.cat(f_i_all)
    y_all = torch.cat(y_all)

    Q_e = generate_orthogonal_matrix(f_e_all.size(1))
    Q_i = generate_orthogonal_matrix(f_i_all.size(1))
    f_e_perturbed = apply_orthogonal_perturbation(f_e_all, Q_e)
    f_i_perturbed = apply_orthogonal_perturbation(f_i_all, Q_i)

    sigma = torch.std(f_e_perturbed, dim=0)
    purifier_e = DiagonalPurification(sigma).to(device)
    purifier_i = ProgressiveLowRankPurification(f_i_perturbed.size(1)).to(device)

    task_model.set_purifiers(purifier_e, purifier_i)


    optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    print(f"Processing time = {round(time.time() - start_time, 2)}  # seconds")
    for epoch in range(30):
        task_model.train()
        total_loss = 0
        for i in range(0, len(f_e_perturbed), 64):
            fe = f_e_perturbed[i:i+64].to(device)
            fi = f_i_perturbed[i:i+64].to(device)
            y = y_all[i:i+64].to(device)

            pred = task_model(fe, fi)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(f_e_perturbed)


        test_mse, test_nmse = evaluate_model(extractor, task_model, test_loader, scaler, device, Q_e, Q_i)
        results.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Test MSE": test_mse,
            "Test NMSE": test_nmse
        })
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f} | Test MSE = {test_mse:.6f} | Test NMSE = {test_nmse:.6f}")

    print("\n📊 Metrics for Visualization:")
    print("train_loss_list =", [round(r["Train Loss"], 4) for r in results])
    print("test_mse_list   =", [round(r["Test MSE"], 4) for r in results])
    print("test_nmse_list   =", [round(r["Test NMSE"], 4) for r in results])
    print(f"time = {round(time.time() - start_time, 2)}  # seconds")
if __name__ == "__main__":
    main()
