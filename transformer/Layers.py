import torch.nn as nn

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, simple_attention, dynamic_v_attention


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        
        self.slf_attn = dynamic_v_attention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_v, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input_q, enc_input_k, enc_input_v, non_pad_mask=None, slf_attn_mask=None):

        enc_output, enc_slf_attn, loss = self.slf_attn(
            enc_input_q, enc_input_k, enc_input_v, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn, loss

