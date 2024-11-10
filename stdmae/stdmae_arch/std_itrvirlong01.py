import torch
from torch import nn

# from .mask import Mask
#
# from .graphwavenet import LoopTransformer_itrvirlong01



class STDITRVIRLONG01(nn.Module):
    """Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting"""

    def __init__(self, dataset_name, pre_trained_tmae_path,pre_trained_smae_path, mask_args, backend_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.pre_trained_smae_path = pre_trained_smae_path
        # iniitalize 
        self.tmae = Mask(**mask_args)
        self.smae = Mask(**mask_args)

        self.backend = LoopTransformer_itrvirlong01(**backend_args)

        print("self.backend",self.backend)

        # load pre-trained model
        self.load_pre_trained_model()


    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters

        # 原始东西  20240604
        # checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        # checkpoint_dict = torch.load(self.pre_trained_tmae_path, map_location=torch.device('cpu'))
        #
        # self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])

        # 原始东西  20240604
        # checkpoint_dict = torch.load(self.pre_trained_smae_path)
        # checkpoint_dict = torch.load(self.pre_trained_smae_path,map_location=torch.device('cpu'))
        #
        # self.smae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False
    # def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of STDMAE.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """
        # print("-------history_dataxxxxx---",history_data.shape)
        # print("-------long_history_data---",long_history_data.shape)


        test2 = torch.randn(1, 864, 307, 2)
        long_history_data = test2
        # reshape
        short_term_history = history_data     # [B, L, N, 1]


        print("------------long_history_data----------",type(long_history_data))
        print("------------short_term_history----------", type(short_term_history))

        print("------------long_history_data-xxxx---------",long_history_data.shape)
        print("------------short_term_history---xxxx-------", short_term_history.shape)



        # batch_size, _, num_nodes, _ = history_data.shape



        hidden_states_t = self.tmae(long_history_data[..., [0]])
        hidden_states_s = self.smae(long_history_data[..., [0]])
        hidden_states=torch.cat((hidden_states_t,hidden_states_s),-1)
        # print("------hidden_states-0------",hidden_states.shape)
        # enhance
        # out_len=1
        # hidden_states = hidden_states[:, :, -out_len, :]
        hidden_states = hidden_states[:, :, 71:72, :]
        # print("------hidden_states-1------",hidden_states.shape)
        y_hat = self.backend(short_term_history,hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1)

        return y_hat

if __name__ == '__main__':
    from thop import profile
    # from stdmae.stdmae_arch.mask import Mask
    from mask import Mask
    # from stdmae.stdmae_arch.graphwavenet import LoopTransformer_itrvirlong01
    from graphwavenet import LoopTransformer_itrvirlong01

    dataset_name= 'PEMS04',
    pre_trained_tmae_path ="mask_save/TMAE_PEMS04_864.pt",
    pre_trained_smae_path= "mask_save/SMAE_PEMS04_864.pt",
    # pre_trained_tmae_path ="",
    # pre_trained_smae_path= "",
    mask_args= {
                    "patch_size":12,
                    "in_channel":1,
                    "embed_dim":96,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "mask_ratio":0.25,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "mode":"forecasting"
    }
    backend_args={
                    "num_nodes" : 307,
                    # "supports"  :[torch.tensor(i) for i in adj_mx],
                    "supports": [torch.rand(94249),torch.rand(94249)],
                    "dropout"   : 0.3,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "aptinit"   : None,
                    "in_dim"    : 2,
                    "out_dim"   : 12,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : 256,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 4,
                    "layers"            : 2
    }

    model = STDITRVIRLONG01(dataset_name,pre_trained_tmae_path,pre_trained_smae_path,mask_args,backend_args)
    test1 = torch.randn(1,1, 12, 307, 2)

    flops, params = profile(model, (test1))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

