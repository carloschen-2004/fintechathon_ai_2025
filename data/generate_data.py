# data/generate_data.py
import numpy as np
import pandas as pd
import os

os.makedirs(os.path.dirname(__file__), exist_ok=True)

np.random.seed(42)

# 参数
N_USERS = 500
N_ITEMS = 200
INTERACTIONS = 10000

# 生成用户特征：年龄(20-70), 风险偏好(0-1), 资金规模(小/中/大 -> 0/1/2)
ages = np.random.randint(20, 70, N_USERS)
risk_pref = np.random.rand(N_USERS)  # 0-1
capital = np.random.choice([0,1,2], size=N_USERS, p=[0.5,0.35,0.15])

users = pd.DataFrame({
    "user_id": np.arange(N_USERS),
    "age": ages,
    "risk_pref": risk_pref,
    "capital_tier": capital
})
users.to_csv("data/users.csv", index=False)

# 生成物品（资产）特征：类别（债券=0, 指数=1, 科技=2）, volatility, expected_return
categories = np.random.choice([0,1,2], size=N_ITEMS, p=[0.3,0.4,0.3])
vol = np.clip(np.random.randn(N_ITEMS)*0.05 + 0.1, 0.01, 0.5)
exp_ret = np.clip(np.random.randn(N_ITEMS)*0.03 + (0.03 + categories*0.03), -0.05, 0.5)

items = pd.DataFrame({
    "item_id": np.arange(N_ITEMS),
    "category": categories,
    "volatility": vol,
    "expected_return": exp_ret
})
items.to_csv("data/items.csv", index=False)

# 生成交互：user-item-score (模拟评分: -1..1 表示不适合->适合)
rows = []
for _ in range(INTERACTIONS):
    u = np.random.randint(0, N_USERS)
    i = np.random.randint(0, N_ITEMS)
    # 基于用户风险和 item 类别产生打分偏好
    u_risk = risk_pref[u]
    cat = categories[i]
    base = exp_ret[i] - vol[i]*0.5
    # 风险匹配：如果用户risk高偏好高风险（cat==2）提升评分
    match = (u_risk * (1 if cat==2 else 0.6 if cat==1 else 0.3))
    noise = np.random.randn() * 0.02
    score = base * 2 + match + noise  # 更高越好
    # 归一到 0-1
    score = (score - (-0.5)) / (1.5 - (-0.5))
    score = np.clip(score, 0.0, 1.0)
    rows.append((u, i, float(score)))

inter = pd.DataFrame(rows, columns=["user_id","item_id","score"])
inter.to_csv("data/interactions.csv", index=False)

print("Generated data: data/users.csv, data/items.csv, data/interactions.csv")
