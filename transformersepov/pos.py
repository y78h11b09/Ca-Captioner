import torch.nn as nn
import torch
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        self.height = height
        self.width = width
        self.d_model = d_model
        pe = torch.zeros(height * width, d_model)
        position = torch.arange(0, height * width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :self.height*self.width, :].to(x.device)

height = 64
width = 64
d_model = 512

pe = PositionalEncoding(d_model, height, width)

# 对于输入 x 的 shape 为 [batch_size, height*width, d_model]
x = torch.randn(2, 64*64, 512)

# 使用 pe 对 x 进行位置编码
output = pe(x)
print(output)