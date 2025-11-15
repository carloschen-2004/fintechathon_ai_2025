# models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StockSelectionModel(nn.Module):
    """
    基于客户分级的选股模型
    支持多因子选股和客户风险偏好个性化
    """
    def __init__(self, client_feat_dim, stock_feat_dim, hidden_layers=[64, 32], dropout_rate=0.2):
        super().__init__()
        
        # 客户特征编码器
        self.client_encoder = nn.Sequential(
            nn.Linear(client_feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 股票特征编码器
        self.stock_encoder = nn.Sequential(
            nn.Linear(stock_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 客户等级注意力机制
        self.level_attention = nn.Sequential(
            nn.Linear(16 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),  # 5个客户等级的注意力权重
            nn.Softmax(dim=1)
        )
        
        # 客户等级特定的因子权重网络
        self.level_factors = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(5)
        ])
        
        # 特征融合网络
        fusion_input_dim = 16 + 32 + 5  # 客户编码 + 股票编码 + 等级注意力
        layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_net = nn.Sequential(*layers)
        
    def forward(self, client_features, stock_features, client_levels=None):
        """
        前向传播
        
        参数:
        - client_features: 客户特征 (batch_size, client_feat_dim)
        - stock_features: 股票特征 (batch_size, stock_feat_dim)
        - client_levels: 客户等级 (batch_size,), 可选
        
        返回:
        - scores: 选股评分 (batch_size,)
        - attention_weights: 注意力权重 (batch_size, 5)
        - factor_contributions: 各因子贡献 (batch_size, 5)
        """
        # 编码客户和股票特征
        client_embedding = self.client_encoder(client_features)
        stock_embedding = self.stock_encoder(stock_features)
        
        # 计算客户等级注意力权重
        attention_input = torch.cat([client_embedding, stock_embedding], dim=1)
        attention_weights = self.level_attention(attention_input)
        
        # 计算每个客户等级因子的贡献
        factor_contributions = torch.zeros_like(attention_weights)
        for i in range(5):
            factor_contributions[:, i] = self.level_factors[i](stock_embedding).squeeze(1)
        
        # 加权组合因子贡献
        weighted_factors = (attention_weights * factor_contributions).sum(dim=1, keepdim=True)
        
        # 特征融合
        fusion_input = torch.cat([
            client_embedding,
            stock_embedding,
            attention_weights
        ], dim=1)
        
        # 最终评分
        scores = self.fusion_net(fusion_input).squeeze(1)
        
        # 如果提供了客户等级，进行个性化调整
        if client_levels is not None:
            # 创建等级掩码
            level_masks = torch.zeros_like(attention_weights)
            for i in range(attention_weights.size(0)):
                level_masks[i, client_levels[i]] = 1
            
            # 增强对应等级的权重
            personalized_attention = attention_weights * (1 + 0.5 * level_masks)
            personalized_attention = personalized_attention / personalized_attention.sum(dim=1, keepdim=True)
            
            # 重新计算加权因子
            weighted_factors = (personalized_attention * factor_contributions).sum(dim=1, keepdim=True)
            
            # 重新计算评分（简化版本）
            scores = scores * 0.7 + weighted_factors.squeeze(1) * 0.3
        
        # 应用sigmoid确保评分在0-1之间
        scores = torch.sigmoid(scores)
        
        return scores, attention_weights, factor_contributions

class MultiFactorStockModel(nn.Module):
    """
    多因子选股模型的简化版本，更适合快速训练和部署
    """
    def __init__(self, client_feat_dim, stock_feat_dim):
        super().__init__()
        
        # 客户特征处理
        self.client_branch = nn.Sequential(
            nn.Linear(client_feat_dim, 16),
            nn.ReLU()
        )
        
        # 股票特征处理
        self.stock_branch = nn.Sequential(
            nn.Linear(stock_feat_dim, 32),
            nn.ReLU()
        )
        
        # 组合特征
        self.combined = nn.Sequential(
            nn.Linear(16 + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, client_features, stock_features):
        """
        简化的前向传播
        """
        client_embed = self.client_branch(client_features)
        stock_embed = self.stock_branch(stock_features)
        
        combined = torch.cat([client_embed, stock_embed], dim=1)
        score = self.combined(combined).squeeze(1)
        
        return torch.sigmoid(score)

# 用于兼容原有代码的模型类
class RecommenderMLP(nn.Module):
    """
    兼容原有推荐系统的模型类
    """
    def __init__(self, user_feat_dim, item_feat_dim, hidden=[64,32]):
        super().__init__()
        input_dim = user_feat_dim + item_feat_dim
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, user_x, item_x):
        # user_x: (B, user_feat_dim), item_x: (B, item_feat_dim)
        x = torch.cat([user_x, item_x], dim=1)
        out = self.net(x)
        return torch.sigmoid(out).squeeze(1)  # 输出 0-1 的评分

# 工具函数
def get_model_by_name(model_name, client_feat_dim, stock_feat_dim):
    """
    根据名称获取模型
    """
    if model_name == 'complex':
        return StockSelectionModel(client_feat_dim, stock_feat_dim)
    elif model_name == 'simple':
        return MultiFactorStockModel(client_feat_dim, stock_feat_dim)
    elif model_name == 'compatible':
        return RecommenderMLP(client_feat_dim, stock_feat_dim)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

# 自定义损失函数，考虑客户等级
class ClientLevelAwareLoss(nn.Module):
    """
    客户等级感知的损失函数
    对不同客户等级使用不同的损失权重
    """
    def __init__(self):
        super().__init__()
        # 不同客户等级的损失权重
        # 保守型客户的错误更重要，给予更高权重
        self.level_weights = torch.tensor([1.5, 1.3, 1.0, 0.9, 0.8], dtype=torch.float)
    
    def forward(self, predictions, targets, client_levels):
        """
        计算损失
        """
        # 基本MSE损失
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # 获取每个样本的权重
        weights = self.level_weights[client_levels].to(predictions.device)
        
        # 加权损失
        weighted_loss = (mse_loss * weights).mean()
        
        return weighted_loss
