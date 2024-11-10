import torch
import torch.nn as nn
from .AGCN import AVWGCN

class AGCRNCell_v6(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell_v6, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)
        self.Linear = nn.Linear(128, 64)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        # print("--------------------显示z_r--------------", z_r.shape)
        # z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        # print("--------x-xxxx-----",x.shape)
        # print("--------z_r-xxxx-----", z_r.shape)
        # print("--------z-xxxx-----", z.shape)
        # print("--------state-xxxx-----", state.shape)
        z_r = self.Linear(z_r)
        # candidate = torch.cat((x, z*state), dim=-1)
        candidate = torch.cat((x, z_r), dim=-1)
        # print("--------------------显示candidate--------------", candidate.shape)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        # print("--------------------显示hc--------------",hc.shape)
        # h = r*state + (1-r)*hc


        return hc

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)