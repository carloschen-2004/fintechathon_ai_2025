# data/generate_data.py
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

os.makedirs(os.path.dirname(__file__), exist_ok=True)

np.random.seed(42)

# 参数 - 增加样本量
N_CLIENTS = 200000  # 客户数
N_STOCKS = 1000   # 股票数量
TRANSACTIONS = 5000000  # 交易数据量

# 行业类别
INDUSTRIES = [
    '金融', '科技', '医药', '消费', '能源', 
    '材料', '工业', '可选消费', '公用事业', '电信服务'
]

# 生成客户数据和客户分级
def generate_client_data(n_clients):
    # 基础特征
    ages = np.random.randint(20, 70, n_clients)
    risk_pref = np.random.beta(2, 2, n_clients)  # 风险偏好(0-1)，beta分布更符合实际
    capital = np.random.choice([0, 1, 2], size=n_clients, p=[0.5, 0.35, 0.15])  # 资金规模：小/中/大
    
    # 投资经验 (0-5年)
    experience = np.random.choice([0, 1, 2, 3, 4, 5], size=n_clients, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])
    
    # 客户分级 - 基于风险偏好、资金规模和投资经验
    features_for_clustering = np.column_stack([risk_pref, capital, experience/5])
    kmeans = KMeans(n_clusters=5, random_state=42)
    client_levels = kmeans.fit_predict(features_for_clustering)
    
    # 客户等级标签映射
    level_labels = {0: '保守型', 1: '稳健型', 2: '平衡型', 3: '成长型', 4: '进取型'}
    client_level_names = [level_labels[level] for level in client_levels]
    
    clients = pd.DataFrame({
        "client_id": np.arange(n_clients),
        "age": ages,
        "risk_pref": risk_pref,
        "capital_tier": capital,
        "experience_years": experience,
        "client_level": client_levels,
        "client_level_name": client_level_names
    })
    return clients

# 生成股票数据
def generate_stock_data(n_stocks):
    # 股票基本信息
    stock_codes = [f"{100000 + i}" for i in range(n_stocks)]
    stock_names = [f"股票{i}" for i in range(n_stocks)]
    
    # 行业分布
    industries = np.random.choice(range(len(INDUSTRIES)), size=n_stocks, p=[0.15, 0.15, 0.12, 0.12, 0.08, 0.08, 0.1, 0.1, 0.05, 0.05])
    
    # 基本面指标
    pe = np.clip(np.random.lognormal(2, 1, n_stocks), 5, 100)  # PE: 5-100
    pb = np.clip(np.random.lognormal(0, 1, n_stocks), 0.5, 20)  # PB: 0.5-20
    roe = np.clip(np.random.normal(0.1, 0.05, n_stocks), -0.2, 0.5)  # ROE: -20% to 50%
    net_profit_growth = np.clip(np.random.normal(0.2, 0.3, n_stocks), -0.5, 2.0)  # 净利润增长率: -50% to 200%
    
    # 技术指标
    ma5 = np.random.uniform(10, 100, n_stocks)  # 5日均线
    ma20 = ma5 * np.random.uniform(0.9, 1.1, n_stocks)  # 20日均线
    macd = np.random.uniform(-5, 5, n_stocks)  # MACD
    rsi = np.clip(np.random.normal(50, 20, n_stocks), 0, 100)  # RSI: 0-100
    
    # 风险指标
    volatility = np.clip(np.random.lognormal(-3, 0.5, n_stocks), 0.01, 0.5)  # 波动率: 1%-50%
    beta = np.clip(np.random.normal(1, 0.3, n_stocks), 0.3, 2.0)  # 贝塔系数: 0.3-2.0
    
    # 预期收益
    # 根据行业、基本面和风险设置不同的预期收益率
    base_return = 0.05
    industry_premium = np.array([0.02, 0.06, 0.04, 0.03, 0.01, 0.02, 0.02, 0.03, 0.005, 0.01])
    roe_premium = (roe - 0.1) * 0.5
    volatility_premium = volatility * 0.3
    
    expected_return = base_return + industry_premium[industries] + roe_premium + volatility_premium
    expected_return = np.clip(expected_return, -0.1, 0.5)  # 预期收益范围：-10%到50%
    
    stocks = pd.DataFrame({
        "stock_id": np.arange(n_stocks),
        "stock_code": stock_codes,
        "stock_name": stock_names,
        "industry": [INDUSTRIES[i] for i in industries],
        "industry_code": industries,
        "pe": pe,
        "pb": pb,
        "roe": roe,
        "net_profit_growth": net_profit_growth,
        "ma5": ma5,
        "ma20": ma20,
        "macd": macd,
        "rsi": rsi,
        "volatility": volatility,
        "beta": beta,
        "expected_return": expected_return
    })
    return stocks

# 生成交易数据（用于模型训练）
def generate_transaction_data(clients, stocks, n_transactions):
    rows = []
    client_ids = clients['client_id'].values
    stock_ids = stocks['stock_id'].values
    client_risks = clients['risk_pref'].values
    stock_volatilities = stocks['volatility'].values
    stock_expected_returns = stocks['expected_return'].values
    stock_industries = stocks['industry_code'].values
    client_levels = clients['client_level'].values
    
    for _ in range(n_transactions):
        # 随机选择客户和股票
        u_idx = np.random.randint(0, len(client_ids))
        i_idx = np.random.randint(0, len(stock_ids))
        
        u = client_ids[u_idx]
        i = stock_ids[i_idx]
        u_risk = client_risks[u_idx]
        u_level = client_levels[u_idx]
        s_vol = stock_volatilities[i_idx]
        s_ret = stock_expected_returns[i_idx]
        s_industry = stock_industries[i_idx]
        
        # 基础评分 = 预期收益 - 风险惩罚
        base_score = s_ret - s_vol * 0.8
        
        # 根据客户等级和股票特征匹配调整评分
        # 保守型客户偏好低波动性
        if u_level == 0:  # 保守型
            match = max(0, 0.5 - s_vol * 5)
        # 稳健型客户平衡风险和收益
        elif u_level == 1:  # 稳健型
            match = (s_ret - 0.05) * 2 - s_vol * 3
        # 平衡型客户
        elif u_level == 2:  # 平衡型
            match = s_ret * 3 - s_vol * 2
        # 成长型客户偏好成长性行业和较高收益
        elif u_level == 3:  # 成长型
            industry_bonus = 0.2 if s_industry in [1, 2, 7] else 0  # 科技、医药、可选消费
            match = (s_ret * 4 - s_vol) + industry_bonus
        # 进取型客户追求高收益，风险容忍度高
        else:  # 进取型
            industry_bonus = 0.3 if s_industry in [1, 2, 4] else 0  # 科技、医药、能源
            match = s_ret * 5 + industry_bonus
        
        # 添加随机噪声
        noise = np.random.randn() * 0.05
        score = base_score + match + noise
        
        # 归一到0-1范围
        score = (score - (-0.5)) / (1.5 - (-0.5))
        score = np.clip(score, 0.0, 1.0)
        
        rows.append((u, i, float(score)))
    
    transactions = pd.DataFrame(rows, columns=["client_id", "stock_id", "score"])
    return transactions

# 生成数据
clients = generate_client_data(N_CLIENTS)
stocks = generate_stock_data(N_STOCKS)
transactions = generate_transaction_data(clients, stocks, TRANSACTIONS)

# 保存数据
clients.to_csv("data/clients.csv", index=False)
stocks.to_csv("data/stocks.csv", index=False)
transactions.to_csv("data/transactions.csv", index=False)

# 为了兼容原有代码，同时生成users.csv和items.csv（但内容已更新）
users = clients.rename(columns={"client_id": "user_id", "client_level": "level", "client_level_name": "level_name"})
items = stocks.rename(columns={"stock_id": "item_id", "industry_code": "category", "volatility": "volatility", "expected_return": "expected_return"})
users.to_csv("data/users.csv", index=False)
items[['item_id', 'category', 'volatility', 'expected_return']].to_csv("data/items.csv", index=False)

# 兼容原有交互数据格式
interactions = transactions.rename(columns={"client_id": "user_id", "stock_id": "item_id"})
interactions.to_csv("data/interactions.csv", index=False)

print("生成数据完成:")
print("- 客户数据: data/clients.csv (包含客户分级)")
print("- 股票数据: data/stocks.csv (包含基本面和技术面指标)")
print("- 交易数据: data/transactions.csv (用于模型训练)")
print("- 兼容数据: data/users.csv, data/items.csv, data/interactions.csv")
print("\n客户等级分布:")
print(clients['client_level_name'].value_counts().sort_index())
