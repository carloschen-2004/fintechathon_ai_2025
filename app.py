# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import pickle
from models.model import RecommenderMLP

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# 加载数据和模型
users = pd.read_csv("data/users.csv")
items = pd.read_csv("data/items.csv")

# load scalers
with open("u_scaler.pkl", "rb") as f:
    u_scaler = pickle.load(f)
with open("i_scaler.pkl", "rb") as f:
    i_scaler = pickle.load(f)

user_feat_cols = ["age","risk_pref","capital_tier"]
item_feat_cols = ["category","volatility","expected_return"]

user_feats_raw = users[user_feat_cols].values.astype(float)
item_feats_raw = items[item_feat_cols].values.astype(float)
user_feats = u_scaler.transform(user_feats_raw)
item_feats = i_scaler.transform(item_feats_raw)

# construct model and load weights
user_feat_dim = user_feats.shape[1]
item_feat_dim = item_feats.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderMLP(user_feat_dim, item_feat_dim)
checkpoint = torch.load("model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

def score_user_item(user_id, item_id):
    uf = torch.tensor(user_feats[[user_id]], dtype=torch.float32).to(device)
    itf = torch.tensor(item_feats[[item_id]], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(uf, itf).cpu().item()
    return float(pred)

def explain_topk(user_id, topk_items):
    # 简单解释：对 top items 比较 expected_return 与 volatility 与用户 risk_pref
    u_risk = users.loc[users.user_id==user_id, "risk_pref"].values[0]
    explanations = []
    for iid in topk_items:
        row = items.loc[items.item_id==iid].iloc[0]
        score = score_user_item(user_id, iid)
        reason = []
        if row.expected_return > 0.06:
            reason.append("预期回报高")
        if row.volatility < 0.12:
            reason.append("波动率低")
        if (row.category == 2) and u_risk>0.6:
            reason.append("与您的高风险偏好匹配")
        if not reason:
            reason.append("综合特征较为平衡")
        explanations.append({
            "item_id": int(iid),
            "category": int(row.category),
            "expected_return": float(row.expected_return),
            "volatility": float(row.volatility),
            "score": round(score,4),
            "explanation": "; ".join(reason)
        })
    return explanations

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    payload = request.json
    user_id = int(payload.get("user_id", 0))
    topk = int(payload.get("topk", 5))
    # score all items for this user
    scores = []
    for iid in items["item_id"].values:
        scores.append((iid, score_user_item(user_id, int(iid))))
    scores.sort(key=lambda x: x[1], reverse=True)
    topk_items = [int(x[0]) for x in scores[:topk]]
    explanations = explain_topk(user_id, topk_items)
    return jsonify({
        "user_id": user_id,
        "topk": topk,
        "recommendations": explanations
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True,threaded=True)
