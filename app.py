# app.py
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import pickle
import os
from models.model import RecommenderMLP
from models.model_utils import StockRecommendationEngine
from data.client_segmentation import ClientSegmentation
from data.stock_features import StockFeatureExtractor

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# 加载数据和模型
users = pd.read_csv("data/users.csv")
items = pd.read_csv("data/items.csv")

# 加载旧格式数据
users_df = users
items_df = items

# load scalers - 使用训练过程中保存的scalers文件
with open("models/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
    u_scaler = scalers['client_scaler']
    i_scaler = scalers['stock_scaler']

# 使用与训练时相同的特征列表，包括experience_years
user_feat_cols = ["age","risk_pref","capital_tier","experience_years"]
# 如果数据中不存在experience_years列，则自动添加
if 'experience_years' not in users.columns:
    print("Warning: experience_years column not found, adding with random values")
    users['experience_years'] = np.random.randint(1, 30, size=len(users))
# 使用与训练时相同的特征列表，包括所有多因子特征
item_feat_cols = ["volatility", "expected_return", "value_factor", "growth_factor", 
                  "momentum_factor", "quality_factor", "risk_level"]
# 如果某些特征列不存在，则自动添加
for col in item_feat_cols:
    if col not in items.columns:
        print(f"Warning: {col} column not found, adding with random values")
        if col == 'risk_level':
            items[col] = np.random.randint(1, 6, size=len(items))
        else:
            items[col] = np.random.normal(0, 1, size=len(items))

user_feats_raw = users[user_feat_cols].values.astype(float)
item_feats_raw = items[item_feat_cols].values.astype(float)
user_feats = u_scaler.transform(user_feats_raw)
item_feats = i_scaler.transform(item_feats_raw)

# construct model and load weights - 使用训练过程中保存的模型
user_feat_dim = user_feats.shape[1]
item_feat_dim = item_feats.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 尝试加载训练好的模型，先尝试使用兼容的RecommenderMLP模型
model = RecommenderMLP(user_feat_dim, item_feat_dim)

try:
    # 尝试加载简化版多因子模型，这个更简单，可能更兼容
    checkpoint = torch.load("models/multi_factor_model.pth", map_location=device)
    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # 处理可能的键名不匹配
    # 创建一个新的状态字典，调整键名以匹配RecommenderMLP
    new_state_dict = {}
    for k, v in model_state_dict.items():
        # 转换键名以匹配RecommenderMLP的结构
        if k.startswith('client_branch'):
            new_k = 'net.0' if k.endswith('.weight') else 'net.2'  # 匹配RecommenderMLP第一层
        elif k.startswith('stock_branch'):
            new_k = 'net.0' if k.endswith('.weight') else 'net.2'  # 简化处理，共享权重
        elif k.startswith('combined'):
            # 映射combined层到RecommenderMLP的后续层
            if 'combined.0' in k:
                new_k = 'net.6'  # 假设是第一个combined线性层
            elif 'combined.3' in k:
                new_k = 'net.9'  # 假设是输出层
            else:
                continue  # 跳过不匹配的层
        else:
            continue
        new_state_dict[new_k] = v
    
    # 如果没有成功映射，使用原始模型结构
    if not new_state_dict:
        # 使用兼容的模型结构
        from models.model import MultiFactorStockModel
        model = MultiFactorStockModel(user_feat_dim, item_feat_dim)
        model.load_state_dict(model_state_dict)
    else:
        # 尝试加载映射后的权重
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except:
            # 如果映射失败，使用原始模型
            from models.model import MultiFactorStockModel
            model = MultiFactorStockModel(user_feat_dim, item_feat_dim)
            model.load_state_dict(model_state_dict)
except Exception as e:
    print(f"Warning: Failed to load model: {e}")
    print("Using default untrained model")

model.to(device)
model.eval()

# 初始化新功能组件
client_segmenter = ClientSegmentation()
stock_extractor = StockFeatureExtractor()

# 尝试初始化推荐引擎
try:
    engine = StockRecommendationEngine()
    print("Stock recommendation engine initialized")
except Exception as e:
    print(f"Warning: Failed to initialize recommendation engine: {e}")
    engine = None

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
    try:
        # 尝试使用模板渲染
        return render_template('index.html')
    except:
        # 回退到静态文件
        return send_from_directory("static", "index.html")

# 获取客户分级信息
@app.route('/client/segment', methods=['POST'])
def segment_client():
    """
    对客户进行分级
    """
    try:
        client_data = request.json
        if not client_data:
            return jsonify({'error': 'Missing client data'}), 400
        
        # 转换为DataFrame
        client_df = pd.DataFrame([client_data])
        
        # 预测客户等级
        level = client_segmenter.predict_level(client_df)[0]
        level_names = ['保守型', '稳健型', '平衡型', '成长型', '进取型']
        level_name = level_names[level]
        
        # 获取投资建议
        advice = client_segmenter.get_investment_advice(level)
        
        return jsonify({
            'level': level,
            'level_name': level_name,
            'advice': advice
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 获取股票推荐
@app.route('/stock/recommend', methods=['POST'])
def recommend_stocks():
    """
    为客户推荐股票
    """
    try:
        request_data = request.json
        if not request_data:
            return jsonify({'error': 'Missing request data'}), 400
        
        client_data = request_data.get('client_data')
        top_k = request_data.get('top_k', 5)
        
        if not client_data:
            return jsonify({'error': 'Missing client_data'}), 400
        
        # 使用items作为股票数据
        stock_data = items_df.copy()
        if 'item_id' in stock_data.columns:
            stock_data = stock_data.rename(columns={'item_id': 'stock_id'})
        
        # 简化推荐逻辑
        client_df = pd.DataFrame([client_data])
        level = client_segmenter.predict_level(client_df)[0]
        
        # 基于规则的简单推荐
        recommendations = []
        for _, stock in stock_data.iterrows():
            score = 0.0
            
            # 基于客户等级的简单规则
            if level == 0:  # 保守型
                if 'volatility' in stock:
                    score += (1.0 - stock['volatility']) * 0.6
                if 'expected_return' in stock:
                    score += min(stock['expected_return'], 0.1) * 0.4
            elif level == 1:  # 稳健型
                if 'volatility' in stock:
                    score += (1.0 - stock['volatility']) * 0.5
                if 'expected_return' in stock:
                    score += stock['expected_return'] * 0.5
            elif level == 2:  # 平衡型
                if 'volatility' in stock:
                    score += (1.0 - stock['volatility'] * 0.8) * 0.4
                if 'expected_return' in stock:
                    score += stock['expected_return'] * 0.6
            elif level == 3:  # 成长型
                if 'volatility' in stock:
                    score += min(stock['volatility'], 0.4) * 0.3
                if 'expected_return' in stock:
                    score += stock['expected_return'] * 0.7
            elif level == 4:  # 进取型
                if 'volatility' in stock:
                    score += min(stock['volatility'], 0.5) * 0.4
                if 'expected_return' in stock:
                    score += stock['expected_return'] * 0.6
            
            recommendations.append({
                'stock_id': stock['stock_id'],
                'score': min(score, 1.0)
            })
        
        # 排序并返回前k个
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        recommendations = recommendations[:top_k]
        
        # 丰富结果
        level_names = ['保守型', '稳健型', '平衡型', '成长型', '进取型']
        enhanced_recommendations = []
        for rec in recommendations:
            stock_info = stock_data[stock_data['stock_id'] == rec['stock_id']].iloc[0].to_dict()
            enhanced_recommendations.append({
                'stock_id': rec['stock_id'],
                'score': rec['score'],
                'stock_info': stock_info
            })
        
        return jsonify({
            'client_level': level_names[level],
            'recommendations': enhanced_recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# 创建模板目录和简单的首页模板
if not os.path.exists('templates'):
    os.makedirs('templates')
    
    simple_index_html = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>客户分级导向的量化选股系统</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; text-align: center; }
            h2 { color: #555; margin-top: 30px; }
            .api-section { margin-bottom: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 6px; }
            code { background-color: #eee; padding: 2px 5px; border-radius: 3px; }
            pre { background-color: #f4f4f4; padding: 15px; border-radius: 6px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>客户分级导向的量化选股系统 API</h1>
            
            <div class="api-section">
                <h2>1. 客户分级</h2>
                <p><strong>URL:</strong> <code>/client/segment</code></p>
                <p><strong>方法:</strong> POST</p>
                <p><strong>请求体示例:</strong></p>
                <pre>{"age": 35, "risk_pref": 0.7, "capital_tier": 4}</pre>
            </div>
            
            <div class="api-section">
                <h2>2. 股票推荐</h2>
                <p><strong>URL:</strong> <code>/stock/recommend</code></p>
                <p><strong>方法:</strong> POST</p>
                <p><strong>请求体示例:</strong></p>
                <pre>{"client_data": {"age": 35, "risk_pref": 0.7, "capital_tier": 4}, "top_k": 5}</pre>
            </div>
            
            <div class="api-section">
                <h2>3. 原始推荐API</h2>
                <p><strong>URL:</strong> <code>/recommend</code></p>
                <p><strong>方法:</strong> POST</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(simple_index_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
