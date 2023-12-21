import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import math
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def replace_negative_values(x,dim):
    mask = x<0.0
    mask = mask.type(torch.float32)
    x = x*(1-mask)
    x = x.masked_fill(mask, -float('inf'))
    return  x

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)      ##xavier初始化
        if bias:
            init.zeros_(self.linear.bias)           ##全0分布

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):  ##点乘注意力
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = (torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor).to(device)  ##matmul函数详解https://blog.csdn.net/qsmx666/article/details/105783610/
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)   ##masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn
def Xnorm(x,gamma):
    norm_tensor = torch.norm(x,2,-1,True)
    return  x*gamma/norm_tensor
def softmax_with_bias(x,bias):
    # x = x.cuda(1).data.cpu().numpy()
    x = torch.tensor(x).cuda(1).data.cpu().numpy()
    x = x / bias
    exp =np.exp(x)
    return exp / np.sum(exp,axis=-1,keepdims=True)
def softmax_with_bias1(x,bias):
    # x = x.cuda(1).data.cpu().numpy()
    #x = torch.tensor(x).cuda(1).data.cpu().numpy()
    x = x / bias
    exp =torch.exp(x)
    return exp /torch.sum(exp,dim=-1,keepdim=True)
class DepthConv2d(nn.Module):
    def __init__(self,in_channel, out_channel,kernel_size=3,padding=1,stride=1):
        super(DepthConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channel,in_channel,kernel_size=kernel_size,padding=padding,stride=stride)
        self.pointwise = nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,stride=1)
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class ScaledDotProductAttention1(nn.Module):  ##点乘注意力
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention1, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.ratio = nn.Parameter(torch.tensor((0.9)))
        self.bias = nn.Parameter(torch.tensor(0.8))
        self.pe = PositionalEncoding(d_model=64,len_q=144)
        self.depth = DepthConv2d(in_channel=64,out_channel=64)
        #self.pe = PosEncoding(d_mode=64, len_q=144)
    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = (torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor).to(device)  ##matmul函数详解https://blog.csdn.net/qsmx666/article/details/105783610/
        # print(scores.shape,0)
        #a = self.softmax(scores)
        # print(q.shape,1)
        # print(k.shape,2)
        pos = self.pe(q)
        # print(pos.shape,3)
        pos1 = self.pe(k)
        pos2 = torch.matmul(pos, pos1.transpose(-1, -2)).to(device)
        # print(pos2.shape,4)
        scores = scores+pos2
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)   ##masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充
        #ratio = 0.90
        ratio = torch.sigmoid(self.ratio)
        top_k = int(ratio*scores.shape[-1])
        val,indices = torch.topk(scores,top_k,dim=-1)
        #print(val.shape)
        filter_value = -float('inf')
        index = scores<val[:,:,:,-1].unsqueeze(-1).repeat(1,1,1,scores.shape[-1])
        scores_ = scores.detach()
        scores_[index] = filter_value
        #b = self.softmax(scores_)
        # print(a[:,:,:,1])
        #pos =
        b = softmax_with_bias1(scores_,self.bias)
        # #print(a[:,:,:,1])
        #
        b = torch.tensor(b).to(device)
        attn = self.dropout(b)
        context = torch.matmul(attn, v)
        #print(v.shape)
        v = v.permute(0,3,1,2)
        #print(v.shape)
        v1 = self.depth(v)
        #print(v1.shape)
        v1 = v1.permute(0,2,3,1)
        context = context+v1
        return context, attn

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))  ##将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.beta = nn.Parameter(torch.zeros(d_hid))  ##ones返回一个全为1 的张量
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)  ##标准差
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, len_q):
        super(PositionalEncoding, self).__init__()
        self.len_q = len_q
        self.d_model = d_model
        pe = torch.zeros(len_q, d_model)
        position = torch.arange(0, len_q, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return  self.pe[:, :self.len_q, :].to(x.device)


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]  ##power 乘方
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)   ##concat多数组拼接 astype类型转换，

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)##功能：torch.from_numpy(ndarray) → Tensor，即 从numpy.ndarray创建一个张量。

    def forward(self, input_len):
        max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])
        input_pos = torch.zeros((input_len.size(0),max_len)).long().to(device)
        for i,len in enumerate(input_len):
            input_pos[i,:len] = torch.arange(1,len+1)   ##torch.arange(start=1.0,end=6.0)的结果不包括end
                                                        #torch.range(start=1.0, end=6.0)的结果包括end
        return self.pos_enc(input_pos)