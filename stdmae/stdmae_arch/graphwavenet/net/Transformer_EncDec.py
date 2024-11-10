import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        # print("------EncoderLayer-----")
        # print("---------attentionxxxx---------------",attention)
        # print("---------d_model-------------", d_model)
        # print("---------d_ff--------------",d_ff)
        # print("-----------activation-----------------",activation)
        # print("------------ EncoderLayer  核心区结束-----------------------")


        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        print("----------self.conv1--xxx------------",self.conv1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        print("----------self.conv2--xxx------------",self.conv2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # print("-------------xxxx---attn_mask----------",attn_mask)
        # print("-------------xxxx---tau----------", tau)
        # print("-------------xxxx---delta----------", delta)
        # print("-------------00-------------",x.shape)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # print("-------------01-------------",new_x.shape)
        # print("-------------02-------------",attn)
        x = x + self.dropout(new_x)
        # print("-------------03-------------", x.shape)

        y = x = self.norm1(x)
        # print("-------------04-------------", y.shape)
        # print("-------------05-------------", (y.transpose(-1, 1)).shape)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # print("-------------06-------------", y.shape)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # print("-------------07-------------", y.shape)
        # print("-------------08-------------", self.norm2(x + y).shape)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        # print("---------打印进入结构---attn_layers-----",attn_layers)
        # print("---------打印进入结构---conv_layers-----",conv_layers)
        # print("---------打印进入结构---norm_layer-----",norm_layer)
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)

        else:
            for attn_layer in self.attn_layers:
                # print("------循环了几次-----------------",attn_layer)
                # print("------循环了几次-attn_mask----------------", attn_mask)
                # print("------循环了几次-tau----------------", tau)
                # print("------循环了几次-delta----------------", delta)
                # print("-----------输入注意力的尺寸-----------",x.shape)
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                # print("-----------输出注意力的尺寸-x----------", x.shape)
                # print("-----------输出注意力的尺寸-attn----------", attn)
                attns.append(attn)
            # print("--------------走的这条线--------------")

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
