from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


from transformersepov.modules3 import Linear
from transformersepov.modules3 import PosEncoding
from transformersepov.layers3 import EncoderLayer, DecoderLayer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i+1].sum() - 1) / (i+1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs-t, min=0.0)

#for self-attention in multi-head(sequence 2 sequence or masked self-attention)
def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k



def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.to(device)
    return subsequent_mask


class Encoder(nn.Module):

    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.layer_type = EncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs,  return_attn=False):
        enc_inputs=enc_inputs.clone().detach().to(device)
        enc_outputs=enc_inputs
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, self_attn_mask=None)
            if return_attn:
                enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0 )
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, enc_outputs, return_attn=False):

        # enc_inputs=enc_inputs.to(device)
        enc_outputs = enc_outputs.to(device)
        dec_outputs = self.tgt_emb(dec_inputs)

        dec_outputs += self.pos_emb(dec_inputs_len) # Adding positional encoding # TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).int()
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).int()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        #dec_inputs:100*46,enc_inputs: 100*49 ---100 46 49### test [5,1],[5,9]--->[5,1,9]
        # dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             self_attn_mask=dec_self_attn_mask,
                                                             enc_attn_mask=None)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns



class Transformer(nn.Module):
    def __init__(self,n_layers_dec, n_layers_enc, d_k, d_v, d_model, d_ff, n_heads,max_seq_len, tgt_vocab_size,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_layers_enc, d_k, d_v, d_model, d_ff, n_heads,
                 dropout)
        self.decoder = Decoder(n_layers_dec, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1)
        self.tgt_proj = Linear(d_model, tgt_vocab_size, bias=False)
    def encode(self, enc_inputs,return_attn=False):

        return self.encoder(enc_inputs ,return_attn)

    def decode(self, dec_inputs, dec_inputs_len, enc_outputs, return_attn=False):
        return self.decoder(dec_inputs, dec_inputs_len, enc_outputs, return_attn)

    def forward(self, enc_inputs, dec_inputs, dec_inputs_len, return_attn=False):

        enc_inputs = enc_inputs.clone().detach().to(device)

        enc_outputs, enc_self_attns = self.encoder(enc_inputs, return_attn)

        dec_inputs = dec_inputs.clone().detach().to(device)
        dec_inputs_len = dec_inputs_len.clone().detach().long().to(device)

        dec_outputs, dec_self_attns, dec_enc_attns =self.decoder(dec_inputs, dec_inputs_len, enc_outputs, return_attn)
        dec_logits = self.tgt_proj(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)),dec_logits
