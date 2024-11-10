import torch
import torch.nn as nn
import math


class attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        '''
        :param embed_dim: 嵌入特征个数
        :param num_heads: scale dot-product attention层数
        '''
        super(attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.w_q = [nn.Linear(embed_dim, embed_dim) for i in range(num_heads)]
        print(self.w_q)
        self.w_k = [nn.Linear(embed_dim, embed_dim) for i in range(num_heads)]
        self.w_v = [nn.Linear(embed_dim, embed_dim) for i in range(num_heads)]
        self.w_o = nn.Linear(embed_dim * num_heads, embed_dim)
        self.softmax = nn.Softmax()


    def single_head(self, q, k, v, head_idx):
        '''scale dot-scale attention '''
        print("---------q----------",q.shape)
        print("---------k----------",q.shape)
        print("---------v----------",q.shape)
        print("---------head_idx-----------", head_idx)
        q = self.w_q[head_idx](q)
        k = self.w_k[head_idx](k)
        v = self.w_v[head_idx](v)
        print("---------k.permute(0, 2, 1)----------", k.permute(0, 2, 1).shape)
        out = torch.matmul(torch.matmul(q, k.permute(0, 2, 1)), v) / self.embed_dim
        return out

    def forward(self, q, k, v):
        output = []
        for i in range(self.num_heads):
            out = self.single_head(q, k, v, i)
            print("-------循环组-------",out.shape)
            output.append(out)
        print(type(output))

        output = torch.cat(output, dim=2)
        print("-------------output-------",output.shape)
        output = self.w_o(output)
        print("-----------self.w_o(output)-----",output.shape)
        return output


if __name__ == '__main__':
    # x = torch.randn(size=(3, 2, 8), dtype=torch.float32)
    x = torch.randn(size=(72, 307, 96), dtype=torch.float32)
    print(x.shape)
    q, k, v = x, x, x
    # print("---------q------------",q.shape)
    # print("---------k------------",k.shape)
    # print("----------v-----------",v.shape)
    # att = attention(embed_dim=8, num_heads=4)
    att = attention(embed_dim=96, num_heads=4)

    # output, attention_weight = att(q, k, v)
    output = att(q, k, v)

    print("----------output-----------",output.shape)
    # print(type(output))
    # print(len(output))
    # print(type(attention_weight))
    # print(len(attention_weight))

