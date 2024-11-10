import torch
import torch.nn as nn
from .AGCN_test import AVWGCN

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # print("---------走了吗--1----------")
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        print("-------------AGCRNCell----xxx-----------")
        print("--------------------state----xxxxx------------------",state.shape)
        print("----------------X-XXX--XXX--------------",x.shape)
        print("----------------X-XXX--node_embeddings--------------",node_embeddings.shape)
        input_and_state = torch.cat((x, state), dim=-1)
        print("--------------------input_and_state----xxxxx------------------", input_and_state.shape)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        print("-------------------self.gate(input_and_state, node_embeddings)----xxxxx------------------", (self.gate(input_and_state, node_embeddings)).shape)
        print("--------------------z_r----xxxxx------------------", z_r.shape)
        print("-------------------self.hidden_dim-----------------", self.hidden_dim)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        print("--------------------z----xxxxx------------------", z.shape)
        print("--------------------r----xxxxx------------------", r.shape)

        candidate = torch.cat((x, z*state), dim=-1)
        print("--------------------candidate----xxxxx------------------", candidate.shape)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        print("--------------------hc----xxxxx------------------", hc.shape)
        h = r*state + (1-r)*hc
        print("--------------------h----xxxxx------------------", h.shape)
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)