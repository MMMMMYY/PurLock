import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import zipfile
import urllib.request
import os
import time



def add_laplace_noise(data, epsilon=1.0, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=data.shape)
    return data + noise



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
        return self.X[idx], self.y[idx].squeeze()  # ✅ 保证 y 是 [1] 而不是 [1,1]



class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])



def load_stock_data(ticker="AAPL", start="2010-01-01", end="2022-01-01"):
    df = yf.download(ticker, start=start, end=end)
    return df[["Close"]]



def load_power_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    zip_path = "household_power_consumption.zip"
    extract_path = "household_power"

    if not os.path.exists("household_power_consumption.txt"):
        print("⬇️ Downloading household dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    df = pd.read_csv(f"{extract_path}/household_power_consumption.txt", sep=';',
                     parse_dates={"datetime": ['Date', 'Time']},
                     infer_datetime_format=True,
                     na_values='?', low_memory=False)

    df.dropna(inplace=True)
    df = df.set_index("datetime")
    df = df[["Global_active_power"]].astype("float32")
    return df


def evaluate_model(model, dataloader, scaler, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            output = model(X).cpu().numpy().reshape(-1, 1)

            if isinstance(y, torch.Tensor):
                y = y.numpy()
            y = y.reshape(-1, 1)

            preds.append(output)
            actuals.append(y)

    preds = scaler.inverse_transform(np.vstack(preds))
    actuals = scaler.inverse_transform(np.vstack(actuals))
    mse = mean_squared_error(actuals, preds)
    return mse




def main(dataset_choice="stock"):
    results = []
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Using dataset: {dataset_choice}")


    if dataset_choice == "stock":
        df = load_stock_data()
    elif dataset_choice == "consumption":
        df = load_power_data()
    else:
        raise ValueError("Invalid dataset_choice. Choose 'stock' or 'consumption'.")

    print(f"✅ Loaded {df.shape[0]} data points.")


    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)


    scaled = add_laplace_noise(scaled, epsilon=1.0, sensitivity=1.0)


    split = int(0.8 * len(scaled))
    train_data, test_data = scaled[:split], scaled[split:]


    window_size = 20
    train_set = TimeSeriesDataset(train_data, window_size)
    test_set = TimeSeriesDataset(test_data, window_size)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


    model = LSTMModel(input_dim=1, hidden_dim=64, num_layers=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    print("🚀 Training...")

    for epoch in range(30):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        train_loss = total_loss / len(train_loader)
        test_mse = evaluate_model(model, test_loader, scaler, device)
        results.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Test MSE": test_mse
        })
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f} | Test MSE = {test_mse:.6f}")

    print("\n📊 Metrics for Visualization:")
    print("train_loss_list =", [round(r["Train Loss"], 4) for r in results])
    print("test_mse_list   =", [round(r["Test MSE"], 4) for r in results])
    print(f"time = {round(time.time() - start_time, 2)}  # seconds")


if __name__ == "__main__":

    main(dataset_choice="stock")
