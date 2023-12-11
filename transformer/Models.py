import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8))#, diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_non_event_mask(seq):
    """ For masking out the non-event time point"""
    len_q = seq.size(1)
    padding_mask = seq.eq(Constants.DUMMY)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)  # event_type embedding

        n_layers = 1
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        # self.transformer_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, query_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask_non_event = get_non_event_mask(event_type)
        slf_attn_mask_non_event = slf_attn_mask_non_event.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq + slf_attn_mask_non_event).gt(0)
        

        # 3 kinds of entry cannot be queried to: (1) future entry (2) padding entry (3) non-event entry

        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        # Important! When doing type embedding, the correspondded number must be integer and within [0, num_types]
        # However, non-event time point has a dummy type setting as -1
        # For now we need to convert the type of non-event time point to 0
        kv_type = event_type.clone()
        kv_type[kv_type == -1] = 0

        query_input = torch.cat([tem_enc, self.event_emb(query_type)], dim=2)
        enc_input_k = torch.cat([tem_enc, self.event_emb(kv_type)], dim=2)

        Loss = 0
        for enc_layer in self.layer_stack:
            
            enc_output, _, loss = enc_layer(
                enc_input_q=query_input,
                enc_input_k=enc_input_k,
                enc_input_v=enc_input_k,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            
            Loss += loss
        return enc_output, Loss

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        self.num_types = num_types
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # \omega * h(t) = kernel to be aligned
        if self.num_types > 1:
            self.intensity_decoder = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_v, 1),
                    nn.Softplus()
                ) for _ in range(self.num_types)])
        else:
            self.intensity_decoder = nn.Sequential(
                    nn.Linear(d_v, 1),
                    nn.Softplus()
                )
        
        self.time_predictor = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softplus()
        )

        self.type_predictor = nn.Linear(
            2*d_model, num_types
        )
        


    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        # print(event_type[1,:], 'event_type')

        non_pad_mask = get_non_pad_mask(event_type)
        log_event_ll = 0
        non_event_ll = 0
        bs, seq_len = event_type.shape
        pred_type = torch.zeros(bs, seq_len, self.num_types).to(event_time.device)

        next_gap = torch.cat((event_time[:, 1:] - event_time[:, :-1],
                              torch.zeros((bs,1)).cuda()), dim=-1)
        next_gap[next_gap<0]=0
        
        Loss = 0

        enc_dict = {}
        for i in range(1, self.num_types + 1):  # event type start from 1
            _type_i = event_type.clone()
            _type_i[_type_i != 0] = i  # manually set all entries except padding to be type i.

            enc_output, loss = self.encoder(event_type, _type_i, event_time, non_pad_mask)

            enc_dict[i]=enc_output

            Loss += loss
            
            _type_i_timestamps = (event_type == i).unsqueeze(-1)  # Select the occuring time of type-i

            indices = torch.where((event_type == i) | (event_type == 0))  # padding, or type-i

            pred_m = self.type_predictor(enc_output)
            pred_type[indices[0], indices[1], :] = pred_m[indices[0], indices[1], :]


            non_type_i_timestamps = ((event_type == -1).unsqueeze(-1))* non_pad_mask

            if self.num_types > 1:
                intensity_type_i = self.intensity_decoder[i-1](enc_output)  # Generate the intensity
            else:
                intensity_type_i = self.intensity_decoder(enc_output)  # Generate the intensity

            log_intensity_type_i = torch.log(intensity_type_i)

            log_intensity_event = log_intensity_type_i * _type_i_timestamps  # Select event intensity
            
            non_event_intensity = intensity_type_i *  non_type_i_timestamps  # Select non-event intensity

            log_event_ll += torch.sum(log_intensity_event)
            non_event_ll += torch.sum(non_event_intensity * 1e-1)

        indices = torch.where(event_type >= 0)  # skip non-event
        padding_n_valid_types = pred_type[indices[0], indices[1], :].reshape(event_time.shape[0], -1, self.num_types)

        return enc_dict, ( log_event_ll, non_event_ll, padding_n_valid_types), Loss
