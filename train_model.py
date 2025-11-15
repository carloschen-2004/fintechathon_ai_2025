# train_model.py - 客户分级导向的量化选股模型训练脚本
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from models.model import StockSelectionModel, MultiFactorStockModel, ClientLevelAwareLoss
from models.model_utils import StockClientDataset
import os
import pickle
from tqdm import tqdm

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/stock_selection_model.pth"
CLIENT_SEGMENTATION_PATH = "models/client_segmentation_model.pth"
SCALERS_PATH = "models/scalers.pkl"

# 确保模型目录存在
os.makedirs("models", exist_ok=True)

# 加载数据
def load_data():
    print("正在加载数据...")
    # 检查是否有新的客户和股票数据，如果没有则使用旧的用户和物品数据
    try:
        # 尝试加载新的数据格式
        clients = pd.read_csv("data/clients.csv")
        stocks = pd.read_csv("data/stocks.csv")
        transactions = pd.read_csv("data/transactions.csv")
        print("成功加载新格式数据")
        return clients, stocks, transactions
    except FileNotFoundError:
        # 如果没有新数据，使用旧的推荐系统数据作为替代
        print("未找到新格式数据，使用旧格式数据进行训练")
        clients = pd.read_csv("data/users.csv")
        # 重命名列以适应新的模型
        if 'user_id' in clients.columns:
            clients = clients.rename(columns={'user_id': 'client_id'})
        # 添加缺失的列
        if 'investment_experience' not in clients.columns:
            clients['investment_experience'] = np.random.randint(0, 20, size=len(clients))
        
        stocks = pd.read_csv("data/items.csv")
        if 'item_id' in stocks.columns:
            stocks = stocks.rename(columns={'item_id': 'stock_id'})
            stocks['ticker'] = [f'STOCK{i:03d}' for i in range(len(stocks))]
            stocks['stock_name'] = [f'股票名称{i}' for i in range(len(stocks))]
            stocks['industry'] = stocks.get('category', np.random.choice(['金融', '科技', '医疗', '消费', '工业'], size=len(stocks)))
            # 添加多因子特征
            stocks['value_factor'] = np.random.normal(0, 1, size=len(stocks))
            stocks['growth_factor'] = np.random.normal(0, 1, size=len(stocks))
            stocks['momentum_factor'] = np.random.normal(0, 1, size=len(stocks))
            stocks['quality_factor'] = np.random.normal(0, 1, size=len(stocks))
            stocks['risk_level'] = np.random.randint(1, 6, size=len(stocks))
        
        transactions = pd.read_csv("data/interactions.csv")
        if 'user_id' in transactions.columns and 'item_id' in transactions.columns:
            transactions = transactions.rename(columns={'user_id': 'client_id', 'item_id': 'stock_id'})
        
        return clients, stocks, transactions

# 特征工程
def prepare_features(clients, stocks):
    print("正在进行特征工程...")
    # 客户特征
    client_feat_cols = ['age', 'risk_pref', 'capital_tier', 'experience_years']
    client_feats = clients[client_feat_cols].values.astype(float)
    
    # 股票特征 - 多因子
    stock_feat_cols = ['volatility', 'expected_return', 'value_factor', 'growth_factor', 
                      'momentum_factor', 'quality_factor', 'risk_level']
    # 确保所有必需的列都存在
    for col in stock_feat_cols:
        if col not in stocks.columns:
            stocks[col] = np.random.normal(0, 1, size=len(stocks))
    
    stock_feats = stocks[stock_feat_cols].values.astype(float)
    
    # 标准化
    client_scaler = StandardScaler().fit(client_feats)
    stock_scaler = StandardScaler().fit(stock_feats)
    
    client_feats_scaled = client_scaler.transform(client_feats)
    stock_feats_scaled = stock_scaler.transform(stock_feats)
    
    return client_feats_scaled, stock_feats_scaled, client_scaler, stock_scaler

# 保存scaler
def save_scalers(client_scaler, stock_scaler):
    with open(SCALERS_PATH, 'wb') as f:
        pickle.dump({
            'client_scaler': client_scaler,
            'stock_scaler': stock_scaler
        }, f)
    print(f"Scalers 已保存到 {SCALERS_PATH}")

# 训练模型
def train_model(clients, stocks, transactions, client_feats_scaled, stock_feats_scaled):
    print("准备训练数据...")
    # 基于交易数据创建匹配的客户-股票特征对
    # 确保transactions中有client_id和stock_id列
    if 'client_id' in transactions.columns and 'stock_id' in transactions.columns:
        # 根据client_id和stock_id从原始缩放特征中获取对应数据
        matched_client_feats = []
        matched_stock_feats = []
        
        # 确保client_id和stock_id在有效范围内
        max_client_id = client_feats_scaled.shape[0] - 1
        max_stock_id = stock_feats_scaled.shape[0] - 1
        
        for _, row in transactions.iterrows():
            client_id = int(row['client_id'])
            stock_id = int(row['stock_id'])
            
            # 确保ID在有效范围内
            if 0 <= client_id <= max_client_id and 0 <= stock_id <= max_stock_id:
                matched_client_feats.append(client_feats_scaled[client_id])
                matched_stock_feats.append(stock_feats_scaled[stock_id])
        
        # 转换为numpy数组
        matched_client_feats = np.array(matched_client_feats)
        matched_stock_feats = np.array(matched_stock_feats)
        
        # 提取目标值和客户等级
        valid_rows = (transactions['client_id'] <= max_client_id) & (transactions['stock_id'] <= max_stock_id)
        filtered_transactions = transactions[valid_rows].copy()
        
        # 使用'score'列作为目标值，这是数据生成中使用的列名
        targets = filtered_transactions['score'].values if 'score' in filtered_transactions.columns else None
        
        # 从clients数据中获取客户等级
        if 'client_id' in filtered_transactions.columns and 'client_id' in clients.columns and 'client_level' in clients.columns:
            # 创建client_id到client_level的映射
            client_level_map = dict(zip(clients['client_id'], clients['client_level']))
            # 为每个交易记录添加对应的客户等级
            filtered_transactions['client_level'] = filtered_transactions['client_id'].map(client_level_map)
            # 移除可能的NaN值
            valid_level_rows = filtered_transactions['client_level'].notna()
            if not valid_level_rows.all():
                print(f"移除 {len(filtered_transactions) - valid_level_rows.sum()} 条没有对应客户等级的记录")
                filtered_transactions = filtered_transactions[valid_level_rows]
                # 相应地更新特征
                matched_client_feats = np.array([matched_client_feats[i] for i, valid in enumerate(valid_level_rows) if valid])
                matched_stock_feats = np.array([matched_stock_feats[i] for i, valid in enumerate(valid_level_rows) if valid])
                if targets is not None:
                    targets = targets[valid_level_rows.values]
            
            client_levels = filtered_transactions['client_level'].values
        else:
            client_levels = None
        
        print(f"匹配到 {len(matched_client_feats)} 对有效客户-股票数据")
        
        # 创建数据集
        dataset = StockClientDataset(matched_client_feats, matched_stock_feats, targets=targets, client_levels=client_levels)
    else:
        # 简单处理：如果没有client_id和stock_id列，创建相同长度的随机样本
        min_len = min(len(client_feats_scaled), len(stock_feats_scaled))
        dataset = StockClientDataset(
            client_feats_scaled[:min_len], 
            stock_feats_scaled[:min_len], 
            targets=None, 
            client_levels=None
        )
        print(f"使用最小长度 {min_len} 创建数据集")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], 
                                                     generator=torch.Generator().manual_seed(42))
    
    # DataLoader
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    
    print("初始化模型...")
    # 初始化模型
    client_feat_dim = client_feats_scaled.shape[1]
    stock_feat_dim = stock_feats_scaled.shape[1]
    
    # 使用我们的多因子选股模型
    model = StockSelectionModel(
        client_feat_dim=client_feat_dim,
        stock_feat_dim=stock_feat_dim,
        hidden_layers=[128, 64]
    ).to(DEVICE)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 使用客户等级感知损失函数
    loss_fn = ClientLevelAwareLoss()
    
    # 训练配置
    num_epochs = 10
    patience = 8
    best_val_loss = float('inf')
    counter = 0
    
    print(f"开始训练，使用 {DEVICE}")
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - 训练"):
            if len(batch) == 4:  # 包含client_levels和target的情况
                client_feats, stock_feats, target, client_levels = batch
            elif len(batch) == 3:  # 只有target的情况
                client_feats, stock_feats, target = batch
                client_levels = None
            elif len(batch) == 2:  # 没有target的情况
                client_feats, stock_feats = batch
                target = None
                client_levels = None
                
            # 将数据移动到设备
            client_feats = client_feats.to(DEVICE).float()
            stock_feats = stock_feats.to(DEVICE).float()
            if target is not None:
                target = target.to(DEVICE).float()
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(client_feats, stock_feats, client_levels=client_levels)
            # 模型返回的是元组：(scores, attention_weights, factor_contributions)
            scores = outputs[0]  # 只取第一个元素
            # 只有当target不为None时计算损失
            if target is not None:
                loss = loss_fn(scores, target, client_levels)
                
                # 反向传播和优化
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * len(target)
            else:
                # 如果没有target，只进行前向传播
                train_loss += 0
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - 验证"):
                if len(batch) == 4:  # 包含client_levels和target的情况
                    client_feats, stock_feats, target, client_levels = batch
                elif len(batch) == 3:  # 只有target的情况
                    client_feats, stock_feats, target = batch
                    client_levels = None
                elif len(batch) == 2:  # 没有target的情况
                    client_feats, stock_feats = batch
                    target = None
                    client_levels = None
                    
            # 将数据移动到设备
            client_feats = client_feats.to(DEVICE).float()
            stock_feats = stock_feats.to(DEVICE).float()
            if target is not None:
                target = target.to(DEVICE).float()
            
            # 前向传播
            outputs = model(client_feats, stock_feats, client_levels=client_levels)
            # 模型返回的是元组：(scores, attention_weights, factor_contributions)
            scores = outputs[0]  # 只取第一个元素
            
            # 只有当target不为None时计算损失
            if target is not None:
                loss = loss_fn(scores, target, client_levels)
                val_loss += loss.item() * len(target)
            else:
                val_loss += 0
        
        val_loss /= len(val_loader.dataset)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{num_epochs} - 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        
        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, MODEL_PATH)
            print(f"保存最佳模型到 {MODEL_PATH}")
        else:
            counter += 1
            if counter >= patience:
                print(f"早停: {patience}个epoch没有改进")
                break
    
    # 尝试加载简化版本的模型作为备选
    print("训练简化版多因子模型作为备选...")
    simple_model = MultiFactorStockModel(
        client_feat_dim=client_feat_dim,
        stock_feat_dim=stock_feat_dim
    ).to(DEVICE)
    
    simple_optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
    simple_loss_fn = nn.MSELoss()
    
    for epoch in range(1, 21):
        simple_model.train()
        train_loss = 0.0
        for batch in train_loader:
            # 处理不同结构的批次数据
            if len(batch) == 4:  # 包含client_levels和target的情况
                client_feats, stock_feats, target, _ = batch
            elif len(batch) == 3:  # 只有target的情况
                client_feats, stock_feats, target = batch
            elif len(batch) == 2:  # 没有target的情况
                client_feats, stock_feats = batch
                target = None
            else:
                continue  # 跳过无法处理的批次
                
            # 将数据移动到设备
            client_feats = client_feats.to(DEVICE).float()
            stock_feats = stock_feats.to(DEVICE).float()
            
            if target is not None:
                target = target.to(DEVICE).float()
                
                simple_optimizer.zero_grad()
                outputs = simple_model(client_feats, stock_feats)
                loss = simple_loss_fn(outputs.squeeze(), target)
                loss.backward()
                simple_optimizer.step()
                
                train_loss += loss.item() * len(target)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        simple_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # 处理不同结构的批次数据
                if len(batch) == 4:  # 包含client_levels和target的情况
                    client_feats, stock_feats, target, _ = batch
                elif len(batch) == 3:  # 只有target的情况
                    client_feats, stock_feats, target = batch
                elif len(batch) == 2:  # 没有target的情况
                    client_feats, stock_feats = batch
                    target = None
                else:
                    continue  # 跳过无法处理的批次
                    
                # 将数据移动到设备
                client_feats = client_feats.to(DEVICE).float()
                stock_feats = stock_feats.to(DEVICE).float()
                
                if target is not None:
                    target = target.to(DEVICE).float()
                    outputs = simple_model(client_feats, stock_feats)
                    loss = simple_loss_fn(outputs.squeeze(), target)
                    val_loss += loss.item() * len(target)
        
        val_loss /= len(val_loader.dataset)
        print(f"简化模型 Epoch {epoch}/20 - 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        
    # 保存简化模型
    simple_model_path = "models/multi_factor_model.pth"
    torch.save({
        'model_state_dict': simple_model.state_dict(),
    }, simple_model_path)
    print(f"简化版模型已保存到 {simple_model_path}")
    
    return best_val_loss

# 主函数
def main():
    print("========= 客户分级导向的量化选股模型训练 ========")
    
    # 加载数据
    clients, stocks, transactions = load_data()
    
    # 特征工程
    client_feats_scaled, stock_feats_scaled, client_scaler, stock_scaler = prepare_features(clients, stocks)
    
    # 保存scalers
    save_scalers(client_scaler, stock_scaler)
    
    # 训练模型
    best_val_loss = train_model(clients, stocks, transactions, client_feats_scaled, stock_feats_scaled)
    
    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存路径: {MODEL_PATH}")
    print(f"Scaler保存路径: {SCALERS_PATH}")
    print("\n请确保在app.py中正确加载这些模型和scalers")

if __name__ == "__main__":
    main()
