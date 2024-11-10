import numpy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from stdmae.stdmae_arch.graphwavenet.net.Transformer_EncDec import Encoder, EncoderLayer
from stdmae.stdmae_arch.graphwavenet.net.SelfAttention_Family import FullAttention, AttentionLayer

class LoopTransformer_itrvirlong01(nn.Module):
    """
        Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
        Link: https://arxiv.org/abs/1906.00121
        Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    # def __init__(self, num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2,d_model=307, **kwargs):
    def __init__(self, num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):

        super(LoopTransformer_itrvirlong01, self).__init__()

        print("------------supports----xxxxx------",np.array(supports).size)
        print("------------supports--type--xxxxx------", type(supports))
        print("------------supports0----xxxxx------",np.array(supports[0]).size)
        print("------------supports0--type--xxxxx------", type(supports[0]))
        print("------------supports1----xxxxx------",np.array(supports[1]).size)
        print("------------supports1--type--xxxxx------", type(supports[1]))
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.fc_his_t = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=512, kernel_size=(1,1), bias=True), nn.ReLU(),
                                      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), bias=True), nn.ReLU())
        self.fc_his_s = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=512, kernel_size=(1,1), bias=True), nn.ReLU(),
                                      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), bias=True), nn.ReLU())
        # 20240705 原始模型
        # self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.start_conv = nn.Conv2d(in_channels=2, out_channels=residual_channels, kernel_size=(1, 1))
        # print("------------检查问题-------------------")
        print( self.start_conv )

        self.skip_convs = nn.ModuleList()
        self.supports = supports
        self.num_nodes = num_nodes

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=1, attention_dropout=dropout,
                                      output_attention=False), d_model=self.num_nodes, n_heads=8),
                    d_model=self.num_nodes,
                    d_ff=2048,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(self.num_nodes)
        )

        receptive_field = 1


        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1,kernel_size),dilation=new_dilation))
                self.skip_convs.append(
                    nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.receptive_field = receptive_field

    def forward(self, input,hidden_states):
        """feed forward of Graph WaveNet.
        Args:
            input (torch.Tensor): input history MTS with shape [B, L, N, C].
            His (torch.Tensor): the output of TSFormer of the last patch (segment) with shape [B, N, d].
        Returns:
            torch.Tensor: prediction with shape [B, N, L]
        """

        # reshape input: [B, L, N, C] -> [B, C, N, L]

        print("-----------input--xxxxxxx--------",input.shape)
        print("-----------hidden_states--xxxxxxx--------", hidden_states.shape)
        # print("----------hidden_states---------",hidden_states.shape)
        # print("-------------xxxxx-------------检测跑了么")
        # print("--------------xxxxxx---input--------", input.shape)
        input = input.transpose(1, 3)
        # print("--------------xxxxxx---input1--------", input.shape)
        # feed forward
        input = nn.functional.pad(input,(1,0,0,0))
        # print("--------------xxxxxx---input2--------", input.shape)
        input = input[:, :2, :, :]
        # print("--------------xxxxxx---input3--------", input.shape)
        in_len = input.size(3)
        print("----------self.receptive_field-------------",self.receptive_field)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        # print("--------------xxxxxx---xxxx----------",x.shape)
        x = self.start_conv(x)
        # print("--------------xxxxxx---self.start_conv(x)---------",x.shape)
        skip = 0

        # print("---------------出问题00-------------")



        # calculate the current adaptive adj matrix
        # new_supports = None
        # if self.gcn_bool and self.addaptadj and self.supports is not None:
        #     adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        #     new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1 skip_convs
            #                                          |
            # ---------------------------------------> + ------------->	*skip*  输出

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            # print("---------filter-----------", filter.shape)

            s = filter
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0

            # if isinstance(skip, torch.Tensor):
            #     print("----------数据处理前---skip-----------", skip.shape)
            # else:
            #     print("----------数据处理前---skip-----------", skip)
            # print("----------数据处理前---s-----------", s.shape)
            skip = s + skip
            # print("----------数据处理后---skip-----------", skip.shape)


            # gate = self.gate_convs[i](residual)
            # gate = torch.sigmoid(gate)
            # x = filter * gate
            # print("-----skip------",skip)

            # print("---------------出问题01-------------")
            # print("--------filter-------------",filter.shape)

            sm = filter.transpose(2, 3)
            # print("---------x-001------------", sm.shape)
            sm = sm.reshape(sm.shape[0], sm.shape[1] * sm.shape[2], sm.shape[3])
            # print("---------x-002------------", sm.shape)
            enc_out, attns = self.encoder(sm, attn_mask=None)
            # print("---------enc_out-循环了几次-----------", enc_out.shape)

            xsm = enc_out.reshape(filter.shape[0], filter.shape[1], filter.shape[3], filter.shape[2])
            # print("---------xsm-----------", xsm.shape)
            x = xsm.transpose(2, 3)
            # print("---------xsmxxxx-----------", x.shape)
            # print("--------------------------------------", xsm.shape)
            x = x + filter
            x = self.bn[i](x)
            print("--------xxxx----xxxxx----------",x.shape)


        # hidden_states_t = self.fc_his_t(hidden_states[:,:,:96])        # B, N, D
        # hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)
        # skip = skip + hidden_states_t
        # hidden_states_s = self.fc_his_s(hidden_states[:,:,96:])        # B, N, D
        # hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)
        # skip = skip + hidden_states_s
        # print("-----------------skip---------------",skip)
        # print("---------------出问题02-------------")
        # print("-----------------skip-------------------",skip.shape)

        hidden_states = hidden_states[:, :,:, :96]
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        # print("----hidden_states------",hidden_states.shape)
        hidden_states_t = self.fc_his_t(hidden_states)        # B, N, D
        # hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)

        # print("----------hidden_states_t -------",hidden_states_t .shape)
        hidden_states_s = self.fc_his_s(hidden_states)        # B, N, D
        # hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)
        # print("----------hidden_states_s -------", hidden_states_s.shape)

        x = hidden_states_t + hidden_states_s
        # print("--------hidden_states_t + hidden_states_s------", x.shape)
        x = skip + x


        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        # print("---------------出问题03-------------")
        x = self.end_conv_2(x)

        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        # print("------------x----------------",x.shape)
        return x


if __name__=='__main__':
    input = torch.rand(1, 12, 307, 2)
    # input = torch.rand(1, 12, 883, 2)
    # 2, 12, 883, 2
    # [2, 883, 1, 192]
    # hidden_states = torch.rand(1, 307, 192)
    hidden_states = torch.rand(1, 307, 1, 192)
    # hidden_states = torch.rand(1, 883, 1, 192)
    matrix = [[0 for _ in range(307)] for _ in range(307)]
    supports = [matrix,matrix]

    net = LoopTransformer_itrvirlong01(num_nodes=307,supports=supports)
    x = net(input,hidden_states)
    print(x.shape)


    pass