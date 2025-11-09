# 1. 创建并激活虚拟环境（可选）

python3 -m venv venv
source venv/bin/activate  # mac/linux

# venv\Scripts\activate    # windows

# 2. 安装依赖

pip install -r requirements.txt

# 3. 生成模拟数据

python data/generate_data.py

# 4. 训练模型（会保存 model.pth, u_scaler.pkl, i_scaler.pkl）

python train_model.py

# 5. 启动后端

python app.py

# 6. 在浏览器打开

# 访问 http://127.0.0.1:5000/

# 前端页面在 static/index.html，会通过 /recommend 接口获取 Top-K 推荐
