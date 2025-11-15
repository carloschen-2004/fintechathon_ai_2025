# models/model_utils.py
import torch
import numpy as np
from .model import StockSelectionModel, MultiFactorStockModel, get_model_by_name, ClientLevelAwareLoss
from data.client_segmentation import ClientSegmentation
from data.stock_features import StockFeatureExtractor
import pandas as pd

class StockRecommendationEngine:
    """
    股票推荐引擎
    整合客户分级和股票选择模型
    """
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = None
        self.client_segmenter = ClientSegmentation()
        self.stock_extractor = StockFeatureExtractor()
        self.model_path = model_path
        
        # 如果提供了模型路径，加载模型
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        加载训练好的模型
        """
        try:
            # 加载模型配置和权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 创建模型
            client_feat_dim = checkpoint.get('client_feat_dim', 4)
            stock_feat_dim = checkpoint.get('stock_feat_dim', 10)
            model_type = checkpoint.get('model_type', 'complex')
            
            self.model = get_model_by_name(model_type, client_feat_dim, stock_feat_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"成功加载模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def predict_client_level(self, client_data):
        """
        预测客户等级
        """
        if isinstance(client_data, dict):
            client_data = pd.DataFrame([client_data])
        
        return self.client_segmenter.predict_level(client_data)
    
    def extract_stock_features(self, stock_data):
        """
        提取股票特征
        """
        return self.stock_extractor.extract_features(stock_data)
    
    def recommend_stocks(self, client_data, stock_data, top_k=5, adjust_for_level=True):
        """
        为客户推荐股票
        
        参数:
        - client_data: 客户数据 (DataFrame 或 dict)
        - stock_data: 股票数据 (DataFrame)
        - top_k: 返回前k个推荐
        - adjust_for_level: 是否根据客户等级调整推荐
        
        返回:
        - 推荐结果列表
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 预测客户等级
        client_levels = self.predict_client_level(client_data)
        if isinstance(client_levels, list):
            client_level = client_levels[0]  # 假设只有一个客户
        else:
            client_level = client_levels
        
        # 提取客户特征
        client_features = self.client_segmenter.extract_features(client_data)
        
        # 提取股票特征
        stock_features = self.extract_stock_features(stock_data)
        
        # 准备模型输入
        client_tensor = torch.tensor(client_features, dtype=torch.float32).to(self.device)
        stock_tensor = torch.tensor(stock_features, dtype=torch.float32).to(self.device)
        
        # 复制客户特征以匹配股票数量
        batch_size = len(stock_data)
        client_tensor = client_tensor.repeat(batch_size, 1)
        
        # 进行预测
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                if adjust_for_level:
                    # 使用复杂模型的完整功能
                    if hasattr(self.model, 'level_attention'):
                        # 对于StockSelectionModel
                        client_level_tensor = torch.tensor([client_level] * batch_size, dtype=torch.long).to(self.device)
                        scores, attention_weights, factor_contribs = self.model(
                            client_tensor, stock_tensor, client_level_tensor
                        )
                        
                        # 准备推荐结果
                        results = []
                        for i, (stock_id, score) in enumerate(zip(stock_data['stock_id'], scores.cpu().numpy())):
                            stock_info = stock_data.iloc[i].to_dict()
                            results.append({
                                'stock_id': stock_id,
                                'score': float(score),
                                'attention_weights': attention_weights[i].cpu().numpy().tolist(),
                                'factor_contributions': factor_contribs[i].cpu().numpy().tolist(),
                                'stock_info': stock_info
                            })
                    else:
                        # 对于简化模型
                        scores = self.model(client_tensor, stock_tensor)
                        results = [{
                            'stock_id': stock_id,
                            'score': float(score)
                        } for stock_id, score in zip(stock_data['stock_id'], scores.cpu().numpy())]
                else:
                    # 不考虑客户等级的简化预测
                    if hasattr(self.model, 'level_attention'):
                        scores, _, _ = self.model(client_tensor, stock_tensor)
                    else:
                        scores = self.model(client_tensor, stock_tensor)
                    
                    results = [{
                        'stock_id': stock_id,
                        'score': float(score)
                    } for stock_id, score in zip(stock_data['stock_id'], scores.cpu().numpy())]
            else:
                raise ValueError("模型不包含forward方法")
        
        # 按评分排序并返回前k个
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def explain_recommendation(self, client_data, stock_id, stock_data):
        """
        解释推荐结果
        """
        # 找到特定股票
        stock_row = stock_data[stock_data['stock_id'] == stock_id]
        if len(stock_row) == 0:
            return {"error": "股票未找到"}
        
        # 预测客户等级
        client_level = self.predict_client_level(client_data)[0]
        level_names = ['保守型', '稳健型', '平衡型', '成长型', '进取型']
        client_level_name = level_names[client_level]
        
        # 获取客户特征
        client_features = self.client_segmenter.extract_features(client_data)
        stock_features = self.extract_stock_features(stock_row)
        
        # 准备输入
        client_tensor = torch.tensor(client_features, dtype=torch.float32).to(self.device)
        stock_tensor = torch.tensor(stock_features, dtype=torch.float32).to(self.device)
        
        # 进行预测以获取注意力权重
        with torch.no_grad():
            if hasattr(self.model, 'level_attention'):
                client_level_tensor = torch.tensor([client_level], dtype=torch.long).to(self.device)
                _, attention_weights, factor_contribs = self.model(
                    client_tensor, stock_tensor, client_level_tensor
                )
                
                # 获取股票信息
                stock_info = stock_row.iloc[0].to_dict()
                
                # 准备解释
                explanation = {
                    "client_level": client_level_name,
                    "stock_info": stock_info,
                    "level_attention": {level_names[i]: float(attention_weights[0][i]) 
                                       for i in range(5)},
                    "factor_contributions": {level_names[i]: float(factor_contribs[0][i])
                                            for i in range(5)},
                    "key_factors": self._identify_key_factors(attention_weights, factor_contribs, stock_info),
                    "recommendation_reason": self._generate_recommendation_reason(
                        client_level, attention_weights, stock_info
                    )
                }
                
                return explanation
            else:
                return {"error": "当前模型不支持推荐解释"}
    
    def _identify_key_factors(self, attention_weights, factor_contribs, stock_info):
        """
        识别关键影响因子
        """
        # 计算每个等级的综合得分
        combined_scores = attention_weights[0] * factor_contribs[0]
        
        # 找出得分最高的两个等级
        top_indices = torch.topk(combined_scores, 2).indices.tolist()
        
        level_names = ['保守型', '稳健型', '平衡型', '成长型', '进取型']
        key_factors = []
        
        # 为每个关键等级生成因子描述
        for idx in top_indices:
            level = level_names[idx]
            if idx == 0:  # 保守型
                factors = ["低波动率", "稳定分红"]
            elif idx == 1:  # 稳健型
                factors = ["中等风险", "合理估值"]
            elif idx == 2:  # 平衡型
                factors = ["增长潜力", "风险适中"]
            elif idx == 3:  # 成长型
                factors = ["高成长性", "行业前景"]
            elif idx == 4:  # 进取型
                factors = ["高收益潜力", "创新能力"]
            
            key_factors.append({
                "level": level,
                "factors": factors,
                "score": float(combined_scores[idx])
            })
        
        return key_factors
    
    def _generate_recommendation_reason(self, client_level, attention_weights, stock_info):
        """
        生成推荐理由
        """
        level_names = ['保守型', '稳健型', '平衡型', '成长型', '进取型']
        client_level_name = level_names[client_level]
        
        # 基于客户等级和股票信息生成推荐理由
        reasons = []
        
        # 基础理由
        reasons.append(f"该股票与您的{client_level_name}投资风格匹配度较高")
        
        # 基于股票特征的理由
        if 'volatility' in stock_info and stock_info['volatility'] < 0.15:
            reasons.append("该股票波动率较低，适合稳健投资")
        elif 'volatility' in stock_info and stock_info['volatility'] > 0.25:
            reasons.append("该股票波动性较高，具有较高的风险和潜在收益")
        
        if 'expected_return' in stock_info:
            if stock_info['expected_return'] > 0.15:
                reasons.append(f"该股票预期收益率达{stock_info['expected_return']:.2%}，具有较好的投资价值")
            elif stock_info['expected_return'] < 0.05:
                reasons.append("该股票预期收益率较低，但风险相对可控")
        
        # 基于行业的理由
        if 'industry' in stock_info:
            industry = stock_info['industry']
            if industry in ['金融', '公用事业', '必需消费品']:
                reasons.append(f"该股票属于{industry}行业，具有较好的稳定性")
            elif industry in ['科技', '新能源', '生物医药']:
                reasons.append(f"该股票属于{industry}行业，具有较高的成长潜力")
        
        return '; '.join(reasons)

# 保存模型的工具函数
def save_model(model, model_path, client_feat_dim, stock_feat_dim, model_type='complex'):
    """
    保存模型及其配置
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'client_feat_dim': client_feat_dim,
        'stock_feat_dim': stock_feat_dim,
        'model_type': model_type
    }
    torch.save(checkpoint, model_path)
    print(f"模型已保存至: {model_path}")

# 模型评估函数
def evaluate_model(model, dataloader, device='cpu'):
    """
    评估模型性能
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # 使用客户等级感知损失函数
    criterion = ClientLevelAwareLoss()
    
    with torch.no_grad():
        for client_features, stock_features, targets, client_levels in dataloader:
            client_features = client_features.to(device)
            stock_features = stock_features.to(device)
            targets = targets.to(device)
            client_levels = client_levels.to(device)
            
            # 获取模型输出
            if hasattr(model, 'level_attention'):
                # StockSelectionModel
                outputs, _, _ = model(client_features, stock_features, client_levels)
            else:
                # 简化模型
                outputs = model(client_features, stock_features)
            
            # 计算损失
            loss = criterion(outputs, targets, client_levels)
            
            total_loss += loss.item() * client_features.size(0)
            total_samples += client_features.size(0)
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    return avg_loss

# 用于数据批处理的数据集类
class StockClientDataset(torch.utils.data.Dataset):
    """
    股票和客户数据集
    """
    def __init__(self, client_features, stock_features, targets=None, client_levels=None):
        self.client_features = client_features
        self.stock_features = stock_features
        self.targets = targets
        self.client_levels = client_levels
    
    def __len__(self):
        return len(self.client_features)
    
    def __getitem__(self, idx):
        # 处理DataFrame或numpy数组的索引访问
        if hasattr(self.client_features, 'iloc'):
            client_feat = torch.tensor(self.client_features.iloc[idx].values, dtype=torch.float32)
        else:
            client_feat = torch.tensor(self.client_features[idx], dtype=torch.float32)
            
        if hasattr(self.stock_features, 'iloc'):
            stock_feat = torch.tensor(self.stock_features.iloc[idx].values, dtype=torch.float32)
        else:
            stock_feat = torch.tensor(self.stock_features[idx], dtype=torch.float32)
        
        if self.targets is not None and self.client_levels is not None:
            # 处理targets的索引访问
            if hasattr(self.targets, 'iloc'):
                target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)
            else:
                target = torch.tensor(self.targets[idx], dtype=torch.float32)
            
            # 处理client_levels的索引访问
            if hasattr(self.client_levels, 'iloc'):
                level = torch.tensor(self.client_levels.iloc[idx], dtype=torch.long)
            else:
                level = torch.tensor(self.client_levels[idx], dtype=torch.long)
            
            return client_feat, stock_feat, target, level
        else:
            return client_feat, stock_feat