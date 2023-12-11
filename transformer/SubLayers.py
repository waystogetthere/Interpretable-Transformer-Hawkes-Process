import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Modules import ScaledDotProductAttention

import pickle as pkl


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head  # Multi-head
        self.d_k = d_k # Dimension of Q, K
        self.d_v = d_v # Dimension of V

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)  

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)   

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # assert False
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)  

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)

        return output, attn  



class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class simple_attention(nn.Module):
    """ This Attention module discard the W^Q and W^K. Dot product is the Query action. """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.d_k = d_k
        self.d_v = d_v # Dimension of V

        self.V = nn.Parameter(torch.tensor(np.random.normal(size=(1,d_v))))

        self.fc = nn.Linear(d_v, 2*d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(2*d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)  # For head axis broadcasting.


        attn = torch.matmul(q / (self.d_k) ** 0.5, k.transpose(-1, 1))

        if mask is not None:
            # print(mask.shape, 'here, final goal')
            # print(mask[0,:, :])
            # assert False
            attn = attn.masked_fill(mask, 0)

        
        # attn = self.dropout(F.softmax(attn, dim=-1))
        # attn = self.dropout(F.softplus(attn))
        attn = self.dropout(attn)
        batch_, seq_len, _ = q.shape
        v_ = self.V.repeat(batch_,seq_len).reshape(batch_, seq_len, -1).float()
        output = torch.matmul(attn, v_)
        visualization_ = True
        if visualization_:
            # print(attn.shape, 'shape!!!!!')
            # print(attn[~mask].std())
            # print(attn[~mask].mean())

            # print(attn[~mask].unique(), len(attn[~mask].unique()), 'unique!')


            a = attn[0, :, :].cpu().detach().numpy()
            print(a.shape,'shape!!')

            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure()
            plt.imshow(a, cmap='hot')#, interpolation='nearest')
            plt.colorbar()
            fig_name = 'data/attn_score.png'
            plt.savefig(fig_name)
            print(fig_name)
            plt.close()

            

            # # assert (v_[0,0,:] == v_[-,-3,:]).all()

            
            b = output[0, :, :].cpu().detach().numpy()

            std = np.std(b, axis=-1)
            mean = np.mean(b, axis=-1)
            print(std, 'std~')
            print(mean, 'mean~')
            print(output.shape, 'output.shape~~')
            
            plt.figure()
            plt.imshow(b, cmap='hot')#, interpolation='nearest')
            plt.colorbar()
            plt.savefig('data/attn_.png')
            plt.close()
            assert False
        # output, attn = self.attention(q, k, self.V, mask=mask)
        output = self.dropout(self.fc(output))
        output += residual

        # if not self.normalize_before:
        #     output = self.layer_norm(output)

        return output, attn  

class dynamic_v_attention(nn.Module):
    """ This Attention module discard the W^Q and W^K. Dot product is the Query action. """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.d_k = d_k
        self.d_v = d_v # Dimension of V

        self.w_vs = nn.Linear(2*d_model, d_v, bias=False)

        # self.V = nn.Parameter(torch.tensor(np.random.normal(size=(1,d_v))))

        self.fc = nn.Linear(d_v, 2*d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(2*d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)
        
        # if mask is not None:
        #     mask = mask.unsqueeze(1)  # For head axis broadcasting.
        
        attn = torch.matmul(q / (self.d_k) ** 0.5, k.transpose(-1, 1))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        # if mask is not None:
        #     attn = attn.masked_fill(mask, 0)
        sz_b, len_v, _ = v.shape
        v = self.w_vs(v).view(sz_b, len_v, self.d_v)

        loss = 0

        output = torch.matmul(attn, v)
        visualization_ = False
        if visualization_:

            # print(attn.shape, 'shape!!!!!')
            # print(attn[~mask].std())
            # print(attn[~mask].mean())

            # print(attn[~mask].unique(), len(attn[~mask].unique()), 'unique!')
            load = True
            if load:
                with open('selected_seq_map.pkl', 'rb') as f:
                    attn = pkl.load(f)
            else:
                if mask is not None:
                    attn = attn.masked_fill(mask, 0)
            a = attn[1, :, :].cpu().detach().numpy()


            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure()
            plt.imshow(a[:200,:200], cmap='Greys')#, interpolation='nearest')
            # plt.imshow(a, cmap='hot')#, interpolation='nearest')
            plt.colorbar()
            # plt.xlabel('Timestamp Index', fontsize=22)
            # plt.xlabel('Timestamp Index',fontsize=22)
            plt.tight_layout()
            # data_ = 'TAXI'
            # data_ = 'AMAZON'
            data_ = 'CONTTIME'
            
            fig_name = 'data/attn_score_'+data_

            plt.savefig(fig_name+'.pdf',dpi=900)
            plt.savefig(fig_name+'.png',dpi=900)
            print(fig_name+'.png')
            plt.close()
            assert False

            import matplotlib.pyplot as plt
            import numpy as np
            # plt.figure()
            # plt.imshow(a, cmap='BuGn')#, interpolation='nearest')
            # # plt.imshow(a, cmap='hot')#, interpolation='nearest')

            # plt.colorbar()
            # # plt.xlabel('Timestamp Index', fontsize=22)
            # # plt.xlabel('Timestamp Index',fontsize=22)
            # plt.tight_layout()
            # plt.savefig('data/attn_score_light.pdf',dpi=900)
            # plt.savefig('data/attn_score_light.png', dpi=900)
            # plt.close()
            
            if not load:
                
                with open('selected_seq_map_{}.pkl'.format(data_), 'wb') as f:
                    pkl.dump(attn, f)

        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)

        return output, attn, loss

        

        
        



        
        

