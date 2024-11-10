import torch
import torch.nn as nn

# -------------hidden_states[:, :, :96]---------------- torch.Size([1, 307, 96])
# ------------hidden_states[:, :, 96:]--------------- torch.Size([1, 307, 96])



class spaceTimeCrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(spaceTimeCrossAttention, self).__init__()

        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)

    def forward(self, input_a, input_b):
        # 线性映射
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        print("------------mapped_a-------------",mapped_a.shape)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)
        print("------------mapped_b-------------",mapped_b.shape)
        # zzz = torch.einsum("knm,bmc->bknc", supports, x)
        y = mapped_b.transpose(1, 2)
        print("-----------y------------", y.shape)
        print()
        # 计算注意力权重
        scores = torch.einsum("lnc,lkc->lnk", mapped_a, mapped_b)
        # scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  # (batch_size, seq_len_a, seq_len_b)
        print("-----------scores-----------", scores.shape)
        attentions_a = torch.softmax(scores, dim=-1)  # 在维度2上进行softmax，归一化为注意力权重 (batch_size, seq_len_a, seq_len_b)
        # print("----------attentions_a----------", attentions_a.shape)
        attentions_b = torch.softmax(scores.transpose(1, 2),
                                     dim=-1)  # 在维度1上进行softmax，归一化为注意力权重 (batch_size, seq_len_b, seq_len_a)
        print("----------attentions_a----------", attentions_a.shape)
        print("----------attentions_b----------", attentions_b.shape)
        print("--------------input_a----------",input_a.shape)
        print("--------------input_b----------",input_b.shape)

        # 使用注意力权重来调整输入表示                   attentions_b [16, 192, 36]   [16, 192, 36]
        # output_a = torch.matmul(attentions_b, input_b)  # (batch_size, seq_len_a, input_dim_b)  原始
        # output_a = torch.matmul(attentions_a.transpose(2, 1), input_b)  # (batch_size, seq_len_a, input_dim_b)  原始
        # output_b = torch.matmul(attentions_b.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)
        output_a = torch.einsum("lnc,lkc->lnk", attentions_a, input_b)
        # output_a = torch.matmul(attentions_a.transpose(2, 1), input_b)  # (batch_size, seq_len_a, input_dim_b)  原始
        # output_b = torch.matmul(attentions_b.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)
        # print("-----------------output_b--------------",output_b.shape)
        print("-----------------output_a--------------", output_a.shape)
        return output_a, output_b
        # return output_b

# -------------hidden_states[:, :, :96]---------------- torch.Size([1, 307, 96])
# ------------hidden_states[:, :, 96:]--------------- torch.Size([1, 307, 96])
# 准备数据
# input_a = torch.randn(16, 36, 192)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
# input_b = torch.randn(16, 192, 36)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)

input_a = torch.randn(1, 307, 96)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
input_b = torch.randn(1, 307, 96)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)
# 定义模型
input_dim_a = input_a.shape[-1]
input_dim_b = input_b.shape[-1]
hidden_dim = 64
cross_attention = spaceTimeCrossAttention(input_dim_a, input_dim_b, hidden_dim)

# 前向传播
output_a, output_b = cross_attention(input_a, input_b)
print("Adjusted output A:\n", output_a)
print("Adjusted output B:\n", output_b)

# x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
