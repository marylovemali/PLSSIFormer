import torch
import torch.nn as nn

# -------------hidden_states[:, :, :96]---------------- torch.Size([1, 307, 96])
# ------------hidden_states[:, :, 96:]--------------- torch.Size([1, 307, 96])



class SpaceCrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(SpaceCrossAttention, self).__init__()

        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)

    def forward(self, input_a, input_b):
        # 线性映射
        input_a = input_a.transpose(1, 3)
        input_b = input_b.transpose(1, 3)


        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        print("------------mapped_a-------------",mapped_a.shape)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)
        print("------------mapped_b-------------",mapped_b.shape)
        # zzz = torch.einsum("knm,bmc->bknc", supports, x)
        # y = mapped_b.transpose(1, 2)
        # print("-----------y------------", y.shape)
        # print()
        # 计算注意力权重
        scores = torch.einsum("bcnl,bcxl->bcnx", mapped_a, mapped_b)
        # scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  # (batch_size, seq_len_a, seq_len_b)
        print("-----------scores-----------", scores.shape)
        # attentions_a = torch.softmax(scores, dim=-1)  # 在维度2上进行softmax，归一化为注意力权重 (batch_size, seq_len_a, seq_len_b)
        # attentions_b = torch.softmax(scores.transpose(1, 2),dim=-1)  # 在维度1上进行softmax，归一化为注意力权重 (batch_size, seq_len_b, seq_len_a)

        attentions_a = torch.softmax(scores, dim=-1)  # 在维度2上进行softmax，归一化为注意力权重 (batch_size, seq_len_a, seq_len_b)
        attentions_b = torch.softmax(scores, dim=-1)  # 在维度1上进行softmax，归一化为注意力权重 (batch_size, seq_len_b, seq_len_a)


        print("----------attentions_a----------", attentions_a.shape)
        print("----------attentions_b----------", attentions_b.shape)
        print("--------------input_a----------",input_a.shape)
        print("--------------input_b----------",input_b.shape)

        # 使用注意力权重来调整输入表示                   attentions_b [16, 192, 36]   [16, 192, 36]
        # output_a = torch.matmul(attentions_b, input_b)  # (batch_size, seq_len_a, input_dim_b)  原始
        # output_a = torch.matmul(attentions_a.transpose(2, 1), input_b)  # (batch_size, seq_len_a, input_dim_b)  原始
        # output_b = torch.matmul(attentions_b.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)
        output_a = torch.einsum("blnn,blnc->blnc", attentions_a, input_b)
        output_b = torch.einsum("blnn,blnc->blnc", attentions_b, input_a)

        output_a = output_a.transpose(1, 3)
        output_b = output_b.transpose(1, 3)

        print("-----------------output_b--------------",output_b.shape)
        print("-----------------output_a--------------", output_a.shape)
        return output_a, output_b


        # return output_b
# -------------hidden_states[:, :, :96]---------------- torch.Size([1, 307, 96])
# ------------hidden_states[:, :, 96:]--------------- torch.Size([1, 307, 96])
# 准备数据
# input_a = torch.randn(16, 36, 192)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
# input_b = torch.randn(16, 192, 36)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)

# input_a = torch.randn(1, 307, 96)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
# input_b = torch.randn(1, 307, 96)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)
output = torch.randn(1, 256, 307, 1)
hidden_states_s = torch.randn([1, 256, 307, 1])

# 定义模型
input_dim_a = output.shape[1]
input_dim_b = hidden_states_s.shape[1]
hidden_dim = 64
cross_attention = SpaceCrossAttention(input_dim_a, input_dim_b, hidden_dim)

# 前向传播
output_a, output_b = cross_attention(output, hidden_states_s)
print("Adjusted output A:\n", output_a.shape)
print("Adjusted output B:\n", output_b.shape)

# x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
