import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super(AttentionModule, self).__init__()

        if hidden_dim is None:
            hidden_dim = in_dim // 2

        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)  # 获取查询向量
        key = self.key(x)  # 获取键向量
        value = self.value(x)  # 获取值向量

        energy = torch.matmul(query, key.transpose(-2, -1))  # 计算注意力能量
        attention_weights = self.softmax(energy)  # 应用softmax获得注意力权重

        attended_values = torch.matmul(attention_weights, value)  # 利用注意力权重加权求和值向量

        return attended_values
