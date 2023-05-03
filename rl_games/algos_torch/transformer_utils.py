import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList

def w_init(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


def make_mlp(dim_list):
    init_ = lambda m: w_init(m)

    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(init_(nn.Linear(dim_in, dim_out)))
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity=nn.ReLU()):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nonlinearity)

    if not final_nonlinearity:
        layers.pop()
    return nn.Sequential(*layers)

def make_mlp_default(dim_list, dense_func = nn.Linear, norm_func_name = None, norm_only_first_layer=False, final_nonlinearity=True, nonlinearity=nn.ReLU):
    in_size = dim_list[0]
    layers = []
    need_norm = True
    for unit in dim_list[1:]:
        layers.append(dense_func(in_size, unit))
        layers.append(nonlinearity)

        if not need_norm:
            continue
        if norm_only_first_layer and norm_func_name is not None:
           need_norm = False 
        if norm_func_name == 'layer_norm':
            layers.append(torch.nn.LayerNorm(unit))
        elif norm_func_name == 'batch_norm':
            layers.append(torch.nn.BatchNorm1d(unit))
        in_size = unit
    return nn.Sequential(*layers)


def num_params(model, only_trainable=True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = model.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for l in self.layers:
            output = l(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def get_attention_maps(self, src, mask=None, src_key_padding_mask=None):
        attention_maps = []
        output = src

        for l in self.layers:
            # NOTE: Shape of attention map: Batch Size x MAX_JOINTS x MAX_JOINTS
            # pytorch avgs the attention map over different heads; in case of
            # nheads > 1 code needs to change.
            output, attention_map = l(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True
            )
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_maps
    
class TransformerEncoderLayerResidual(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayerResidual, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayerResidual, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False):
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if return_attention:
            return src, attn_weights
        else:
            return src

class TransformerModel(nn.Module):
    def __init__(self, cfg, obs_space, act_fn):
        super(TransformerModel, self).__init__()
        self.cfg = cfg
        self.seq_len = self.cfg["input_sequence_length"]
        # Embedding layer for per limb obs
        self.d_model = self.cfg["transform_embed_dim"]
        self.embed = nn.Linear(16, self.d_model)

        if self.cfg["pos_embedding"] == "learnt":
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.cfg["pos_embedding"] == "abs":
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            self.cfg["transform_embed_dim"],
            self.cfg["num_head"],
            self.cfg["dim_feedforward"],
            self.cfg["dropout"],
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.cfg["num_layer"], norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model + self.cfg["state_embed_dim"][-1]
        self.state_encoder = MLPObsEncoder(obs_space["state"][0], act_fn, self.cfg)
        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = make_mlp_default(
            [decoder_input_dim] + list(self.cfg["decoder_mlp_dim"]),
            final_nonlinearity=True,
            nonlinearity=act_fn,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.state_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)    
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)    
        initrange = self.cfg["embed_init"]
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, return_attention=False):
        #print("B, N, 16", obs["transforms"].shape)
        # (batch_size, num_transforms, 16) -> (num_transforms, batch_size, 16)
        obs_t = obs["transforms"].permute(1,0,2)
        #print("N, B, 16", obs_t.shape)
        # (num_transforms, batch_size, 16) -> (num_transforms, batch_size, d_model)
        transforms_embed = self.embed(obs_t) * math.sqrt(self.d_model)
        #print("N, B, d_model", transforms_embed.shape)
        _, batch_size, _ = transforms_embed.shape

        attention_maps = None
        state_obs = self.state_encoder(obs["state"])
        #print("B, d_state", state_obs.shape)
        state_obs = state_obs.repeat(self.seq_len, 1)
        state_obs = state_obs.reshape(self.seq_len, batch_size, -1)
        #print("N, B, d_state", state_obs.shape)

        if self.cfg["pos_embedding"] in ["learnt", "abs"]:
            transforms_embed = self.pos_embedding(transforms_embed)
        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                transforms_embed, src_key_padding_mask=obs["masks"].bool()
            )
        else:
            # (num_transforms, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                transforms_embed, src_key_padding_mask=None#obs["masks"].bool()
            )
        #print("N, B, d_model", obs_embed_t.shape)
        decoder_input = torch.cat([obs_embed_t, state_obs], axis=2)
        #print("=======")
        #print(obs_embed_t.shape)
        #print(decoder_input.shape)
        #print("N, B, d_model + d_state", decoder_input.shape)

        # (num_transforms, batch_size, N)
        output = self.decoder(decoder_input)
        # (batch_size, num_transforms, N)
        output = output.permute(1, 0, 2)
        #print("B, N, decoder_out", output.shape)

        return output, attention_maps, batch_size

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class MLPObsEncoder(nn.Module):
    def __init__(self, obs_dim, act, cfg):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + list(cfg["state_embed_dim"])
        self.encoder = make_mlp_default(mlp_dims, nonlinearity=act)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)