# models/model.py
import torch
import torch.nn as nn

class RecommenderMLP(nn.Module):
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
