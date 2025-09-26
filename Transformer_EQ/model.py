import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class TransformerEqualizer(nn.Module):
    def __init__(
            self, d_model=64, d_inner=256, n_layers=3, n_head=4, d_k=16, d_v=16,
            dropout=0.1, n_position=200):
        super().__init__()

        self.d_model = d_model
        # The input is a sequence of continuous values, so we use a linear layer as input embedding
        self.input_linear = nn.Linear(1, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # The output layer maps the Transformer output to 4 PAM4 levels
        self.output_linear = nn.Linear(d_model, 4)

    def forward(self, src_seq, src_mask=None):
        # -- Reshape input for linear layer
        src_seq_reshaped = src_seq.unsqueeze(-1)

        # -- Forward
        enc_output = self.input_linear(src_seq_reshaped)
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=src_mask)

        # -- Output
        logits = self.output_linear(enc_output)
        return logits