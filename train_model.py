# train_model.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.model import RecommenderMLP
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"

# 加载数据
users = pd.read_csv("data/users.csv")
items = pd.read_csv("data/items.csv")
inter = pd.read_csv("data/interactions.csv")

# 特征工程：拼接数值型特征并标准化
user_feat_cols = ["age","risk_pref","capital_tier"]
item_feat_cols = ["category","volatility","expected_return"]

user_feats = users[user_feat_cols].values.astype(float)
item_feats = items[item_feat_cols].values.astype(float)

u_scaler = StandardScaler().fit(user_feats)
i_scaler = StandardScaler().fit(item_feats)

user_feats = u_scaler.transform(user_feats)
item_feats = i_scaler.transform(item_feats)

# 保存scaler以供后端推理使用（简单方式：pickle）
import pickle
with open("u_scaler.pkl", "wb") as f:
    pickle.dump(u_scaler, f)
with open("i_scaler.pkl", "wb") as f:
    pickle.dump(i_scaler, f)

# Dataset
class InteractionDataset(Dataset):
    def __init__(self, inter_df):
        self.u = inter_df["user_id"].values.astype(int)
        self.i = inter_df["item_id"].values.astype(int)
        self.y = inter_df["score"].values.astype(float)

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.y[idx]

dataset = InteractionDataset(inter.sample(frac=1.0, random_state=42))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

def collate(batch):
    us, is_, ys = zip(*batch)
    us = torch.tensor(us, dtype=torch.long)
    is_ = torch.tensor(is_, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.float32)
    return us, is_, ys

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False, collate_fn=collate)

# Model
user_feat_dim = user_feats.shape[1]
item_feat_dim = item_feats.shape[1]
model = RecommenderMLP(user_feat_dim, item_feat_dim).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# helper to get features batch by ids
def get_user_item_feats(user_ids, item_ids):
    uf = torch.tensor(user_feats[user_ids], dtype=torch.float32).to(DEVICE)
    itf = torch.tensor(item_feats[item_ids], dtype=torch.float32).to(DEVICE)
    return uf, itf

# training loop
best_val = 1e9
for epoch in range(1, 21):
    model.train()
    total_loss = 0.0
    for us, is_, ys in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        uf, itf = get_user_item_feats(us.numpy(), is_.numpy())
        preds = model(uf, itf)
        loss = loss_fn(preds, ys.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(ys)
    train_loss = total_loss / len(train_loader.dataset)

    # val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for us, is_, ys in val_loader:
            uf, itf = get_user_item_feats(us.numpy(), is_.numpy())
            preds = model(uf, itf)
            val_loss += loss_fn(preds, ys.to(DEVICE)).item() * len(ys)
    val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "model_state": model.state_dict(),
            "u_scaler": None, "i_scaler": None
        }, MODEL_PATH)
        print("Saved best model.")

print("Training finished. Best val:", best_val)
