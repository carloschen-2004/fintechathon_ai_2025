# data/stock_features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class StockFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def fit(self, stocks_df):
        """
        训练特征提取器
        """
        # 提取特征
        features = self._extract_features(stocks_df)
        
        # 训练标准化器
        self.scaler.fit(features)
        
        return self
    
    def transform(self, stocks_df):
        """
        转换股票数据为特征向量
        """
        # 提取特征
        features = self._extract_features(stocks_df)
        
        # 标准化特征
        scaled_features = self.scaler.transform(features)
        
        return scaled_features
    
    def _extract_features(self, stocks_df):
        """
        从股票数据中提取特征
        """
        # 定义需要的特征列
        fundamental_cols = ['pe', 'pb', 'roe', 'net_profit_growth']
        technical_cols = ['macd', 'rsi']
        risk_cols = ['volatility', 'beta']
        return_cols = ['expected_return']
        
        # 确保所有需要的列都存在
        all_required_cols = fundamental_cols + technical_cols + risk_cols + return_cols
        for col in all_required_cols:
            if col not in stocks_df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 提取基本面特征
        pe = stocks_df['pe'].values
        pb = stocks_df['pb'].values
        roe = stocks_df['roe'].values
        net_profit_growth = stocks_df['net_profit_growth'].values
        
        # 提取技术面特征
        macd = stocks_df['macd'].values
        rsi = stocks_df['rsi'].values
        
        # 计算额外的技术指标
        # MA5/MA20 比率（如果存在）
        if 'ma5' in stocks_df.columns and 'ma20' in stocks_df.columns:
            ma_ratio = stocks_df['ma5'].values / stocks_df['ma20'].values
        else:
            ma_ratio = np.ones(len(stocks_df))
        
        # 提取风险特征
        volatility = stocks_df['volatility'].values
        beta = stocks_df['beta'].values
        
        # 提取收益特征
        expected_return = stocks_df['expected_return'].values
        
        # 计算价值因子
        # 低PE、低PB、高ROE通常被视为价值股票的特征
        value_score = (1/pe) + (1/pb) + roe
        
        # 计算成长因子
        # 净利润增长率、预期收益
        growth_score = net_profit_growth + expected_return
        
        # 计算动量因子
        # 使用RSI和MACD作为动量指标的代理
        momentum_score = macd + (rsi - 50)/50  # 将RSI归一化到-1到1
        
        # 计算质量因子
        # 高ROE、稳定的增长
        quality_score = roe - np.abs(net_profit_growth - 0.1)  # 偏离正常增长率的惩罚
        
        # 计算风险调整后收益
        sharpe_ratio = np.zeros_like(expected_return)
        mask = volatility > 0
        sharpe_ratio[mask] = expected_return[mask] / volatility[mask]
        
        # 组合所有特征
        features = np.column_stack([
            pe, pb, roe, net_profit_growth,
            macd, rsi, ma_ratio,
            volatility, beta,
            expected_return,
            value_score, growth_score, momentum_score, quality_score,
            sharpe_ratio
        ])
        
        # 保存特征列名
        self.feature_columns = [
            'pe', 'pb', 'roe', 'net_profit_growth',
            'macd', 'rsi', 'ma_ratio',
            'volatility', 'beta',
            'expected_return',
            'value_score', 'growth_score', 'momentum_score', 'quality_score',
            'sharpe_ratio'
        ]
        
        return features
    
    def get_feature_importance_weights(self, client_level):
        """
        根据客户等级返回不同的特征权重
        """
        # 不同客户等级的特征权重配置
        # 权重格式: [基本面, 技术面, 风险, 收益, 价值因子, 成长因子, 动量因子, 质量因子, 风险调整后收益]
        weights = {
            0: [0.1, 0.05, 0.3, 0.1, 0.2, 0.05, 0.05, 0.1, 0.05],  # 保守型: 重视风险
            1: [0.15, 0.1, 0.25, 0.15, 0.15, 0.05, 0.05, 0.08, 0.02],  # 稳健型: 平衡风险和收益
            2: [0.15, 0.15, 0.2, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02],  # 平衡型: 平衡各方面
            3: [0.1, 0.2, 0.15, 0.25, 0.05, 0.15, 0.05, 0.03, 0.02],  # 成长型: 重视成长性
            4: [0.05, 0.25, 0.1, 0.3, 0.05, 0.2, 0.03, 0.02, 0.0]  # 进取型: 重视高收益
        }
        
        return np.array(weights.get(client_level, weights[2]))
    
    def save(self, path):
        """
        保存特征提取器
        """
        data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path):
        """
        加载特征提取器
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls()
        extractor.scaler = data['scaler']
        extractor.feature_columns = data['feature_columns']
        
        return extractor

# 股票行业分类
def get_industry_groups():
    """
    返回行业分类组
    """
    return {
        '金融': ['银行', '证券', '保险', '多元金融'],
        '科技': ['软件服务', '硬件设备', '通信设备', '半导体'],
        '医药': ['医药制造', '医疗器械', '医疗服务', '生物科技'],
        '消费': ['食品饮料', '家用电器', '日用品', '零售'],
        '能源': ['石油石化', '煤炭', '电力', '新能源'],
        '材料': ['化工', '建材', '钢铁', '有色金属'],
        '工业': ['机械制造', '工业服务', '航空航天', '国防军工'],
        '可选消费': ['汽车', '传媒娱乐', '旅游酒店', '教育'],
        '公用事业': ['水务', '燃气', '公共交通'],
        '电信服务': ['电信运营商', '通信服务']
    }

# 风险评级函数
def get_stock_risk_level(stocks_df):
    """
    根据股票特征评定风险等级
    """
    # 计算综合风险分数
    risk_score = 0
    
    # 波动率权重 0.4
    risk_score += 0.4 * (stocks_df['volatility'] - stocks_df['volatility'].min()) / \
                 (stocks_df['volatility'].max() - stocks_df['volatility'].min())
    
    # 贝塔系数权重 0.3
    risk_score += 0.3 * (stocks_df['beta'] - stocks_df['beta'].min()) / \
                 (stocks_df['beta'].max() - stocks_df['beta'].min())
    
    # 行业风险因子权重 0.3
    # 定义不同行业的风险系数
    industry_risk = {
        '金融': 0.3,
        '科技': 0.8,
        '医药': 0.6,
        '消费': 0.4,
        '能源': 0.7,
        '材料': 0.6,
        '工业': 0.5,
        '可选消费': 0.5,
        '公用事业': 0.2,
        '电信服务': 0.3
    }
    
    # 映射行业风险
    industry_risk_scores = stocks_df['industry'].map(industry_risk).fillna(0.5)
    risk_score += 0.3 * industry_risk_scores
    
    # 将风险分数映射到风险等级
    def map_risk_level(score):
        if score < 0.3:
            return 0  # 低风险
        elif score < 0.5:
            return 1  # 较低风险
        elif score < 0.7:
            return 2  # 中等风险
        elif score < 0.85:
            return 3  # 较高风险
        else:
            return 4  # 高风险
    
    # 应用映射函数
    risk_levels = risk_score.apply(map_risk_level)
    risk_level_names = ['低风险', '较低风险', '中等风险', '较高风险', '高风险']
    risk_level_names_series = risk_levels.map(lambda x: risk_level_names[x])
    
    return risk_levels, risk_level_names_series

if __name__ == "__main__":
    # 示例用法
    try:
        # 加载股票数据
        stocks = pd.read_csv('data/stocks.csv')
        
        # 创建特征提取器
        extractor = StockFeatureExtractor()
        extractor.fit(stocks)
        
        # 提取特征
        features = extractor.transform(stocks)
        
        # 保存特征提取器
        extractor.save('stock_feature_extractor.pkl')
        
        # 获取风险等级
        risk_levels, risk_level_names = get_stock_risk_level(stocks)
        
        # 添加风险等级到股票数据
        stocks_with_risk = stocks.copy()
        stocks_with_risk['risk_level'] = risk_levels
        stocks_with_risk['risk_level_name'] = risk_level_names
        
        # 保存结果
        stocks_with_risk.to_csv('data/stocks_with_risk.csv', index=False)
        
        print(f"特征提取完成，共提取 {features.shape[1]} 个特征")
        print(f"特征列名: {', '.join(extractor.feature_columns)}")
        print("\n股票风险等级分布:")
        print(risk_level_names.value_counts().sort_index())
        print("\n增强后的股票数据已保存到 data/stocks_with_risk.csv")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已生成股票数据文件 data/stocks.csv")