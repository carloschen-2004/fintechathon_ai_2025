# 客户分级为导向的量化机器学习选股系统

## 系统架构

### 核心模块

1. **客户分级模块**

   - 基于客户特征（年龄、风险偏好、资金规模）进行客户分级
   - 使用聚类算法将客户分为不同风险承受能力的群体
2. **股票数据处理模块**

   - 股票基础数据处理（价格、成交量、市值等）
   - 技术指标计算（均线、MACD、RSI、波动率等）
   - 基本面数据处理（PE、PB、ROE、净利润增长率等）
3. **特征工程模块**

   - 股票特征提取和标准化
   - 客户特征与股票特征的匹配机制
4. **选股模型模块**

   - 基于机器学习的多因子选股模型
   - 针对不同客户等级的个性化权重调整
5. **推荐引擎模块**

   - 根据客户等级和风险偏好推荐适合的股票组合
   - 投资组合优化（风险分散、收益最大化）

## 技术栈

- Python 3.8+
- PyTorch（深度学习模型）
- Flask（Web服务）
- Pandas/NumPy（数据处理）
- Scikit-learn（机器学习算法）

## 数据流

1. 客户数据 -> 客户分级模块 -> 客户等级标签
2. 股票数据 -> 特征工程模块 -> 股票特征向量
3. 客户等级 + 股票特征 -> 选股模型 -> 选股结果
4. 选股结果 -> 推荐引擎 -> 个性化股票组合

## 运行指南

# 1. 创建并激活虚拟环境（可选）

python3 -m venv venv
source venv/bin/activate  # mac/linux

# venv\Scripts\activate    # windows

# 2. 安装依赖

pip install -r requirements.txt

# 3. 生成模拟数据

python3 data/generate_data.py

# 4. 训练模型（会保存 model.pth, u_scaler.pkl, i_scaler.pkl）

python3 train_model.py

# 5. 启动后端

python app.py

# 6. 在浏览器打开

# 访问 http://127.0.0.1:5000/

# 前端页面在 static/index.html，会通过 /recommend 接口获取 Top-K 推荐
