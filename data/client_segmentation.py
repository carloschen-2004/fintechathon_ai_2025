# data/client_segmentation.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

class ClientSegmentation:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.level_labels = {0: '保守型', 1: '稳健型', 2: '平衡型', 3: '成长型', 4: '进取型'}
        
    def fit(self, clients_df):
        """
        训练客户分级模型
        """
        # 提取用于聚类的特征
        features = self._extract_features(clients_df)
        
        # 标准化特征
        scaled_features = self.scaler.fit_transform(features)
        
        # 训练聚类模型
        self.cluster_model.fit(scaled_features)
        
        return self
    
    def predict(self, clients_df):
        """
        预测客户等级
        """
        # 提取特征
        features = self._extract_features(clients_df)
        
        # 标准化
        scaled_features = self.scaler.transform(features)
        
        # 预测等级
        levels = self.cluster_model.predict(scaled_features)
        
        # 添加等级名称
        level_names = [self.level_labels[level] for level in levels]
        
        # 返回结果
        result = clients_df.copy()
        result['client_level'] = levels
        result['client_level_name'] = level_names
        
        return result
    
    def _extract_features(self, clients_df):
        """
        从客户数据中提取用于聚类的特征
        """
        # 确保必要的列存在
        required_columns = ['risk_pref', 'capital_tier', 'experience_years']
        for col in required_columns:
            if col not in clients_df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 提取特征并标准化经验年数
        risk_pref = clients_df['risk_pref'].values
        capital_tier = clients_df['capital_tier'].values
        experience_years = clients_df['experience_years'].values / 5  # 归一化到0-1范围
        
        # 组合特征
        features = np.column_stack([risk_pref, capital_tier, experience_years])
        
        return features
    
    def save_model(self, model_path):
        """
        保存模型
        """
        model_data = {
            'cluster_model': self.cluster_model,
            'scaler': self.scaler,
            'level_labels': self.level_labels,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, model_path):
        """
        加载模型
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建实例
        instance = cls(
            n_clusters=model_data['n_clusters'],
            random_state=model_data['random_state']
        )
        
        # 加载模型参数
        instance.cluster_model = model_data['cluster_model']
        instance.scaler = model_data['scaler']
        instance.level_labels = model_data['level_labels']
        
        return instance
    
    def get_client_profile(self, level):
        """
        获取客户等级对应的投资建议和风险偏好
        """
        profiles = {
            0: {  # 保守型
                'risk_tolerance': '低',
                'return_expectation': '低',
                'investment_horizon': '中长期',
                'suitable_assets': ['债券', '货币基金', '蓝筹股'],
                'allocation_suggestion': '债券类资产占比70%以上，权益类资产占比不超过30%'
            },
            1: {  # 稳健型
                'risk_tolerance': '较低',
                'return_expectation': '较低',
                'investment_horizon': '中长期',
                'suitable_assets': ['债券', '价值股', '指数基金'],
                'allocation_suggestion': '债券类资产占比50-70%，权益类资产占比30-50%'
            },
            2: {  # 平衡型
                'risk_tolerance': '中等',
                'return_expectation': '中等',
                'investment_horizon': '中长期',
                'suitable_assets': ['价值股', '成长股', '债券', '基金'],
                'allocation_suggestion': '权益类资产和固定收益类资产各占约50%'
            },
            3: {  # 成长型
                'risk_tolerance': '较高',
                'return_expectation': '较高',
                'investment_horizon': '中长期',
                'suitable_assets': ['成长股', '科技股', '医药股', '基金'],
                'allocation_suggestion': '权益类资产占比70-80%，固定收益类资产占比20-30%'
            },
            4: {  # 进取型
                'risk_tolerance': '高',
                'return_expectation': '高',
                'investment_horizon': '中长期',
                'suitable_assets': ['成长股', '科技股', '高波动性行业股票'],
                'allocation_suggestion': '权益类资产占比80%以上，可考虑少量衍生品'
            }
        }
        
        return profiles.get(level, profiles[0])

# 使用示例
def segment_clients(clients_df, model_path='client_segmentation_model.pkl'):
    """
    对客户进行分级
    """
    # 创建并训练模型
    segmentation = ClientSegmentation()
    segmentation.fit(clients_df)
    
    # 预测并返回结果
    result = segmentation.predict(clients_df)
    
    # 保存模型
    segmentation.save_model(model_path)
    
    return result

if __name__ == "__main__":
    # 示例用法
    import pandas as pd
    
    # 加载客户数据
    try:
        clients = pd.read_csv('data/clients.csv')
        
        # 进行客户分级
        segmented_clients = segment_clients(clients)
        
        # 打印分级结果统计
        print("客户分级结果:")
        print(segmented_clients['client_level_name'].value_counts().sort_index())
        
        # 保存分级后的客户数据
        segmented_clients.to_csv('data/clients_segmented.csv', index=False)
        print("\n分级后的客户数据已保存到 data/clients_segmented.csv")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已生成客户数据文件 data/clients.csv")