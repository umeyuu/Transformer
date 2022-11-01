import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import gelu
from torch.nn.functional import relu
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout_rate=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        scalar = np.sqrt(self.d_k)
        # Q*K / D^0.5
        # attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar
        attention_weight = torch.matmul(q, torch.transpose(k, 2, 3)) / scalar
        
        # maskに対する処理
        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    f'mask.dim != attention_weight.dim, mask.dim={mask.dim()}, attention_weight.dim={attention_weight.dim()}'
                )
            attention_weight = attention_weight.data.masked_fill_(mask, -torch.finfo(torch.float).max)
            
        attention_weight = nn.functional.softmax(attention_weight, dim=-1)
        # attention_weight = self.dropout(attention_weight)
        
        return torch.matmul(attention_weight, v)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.d_k = d_model // head
        self.d_v = d_model // head
        
        # # [ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)]
        # self.W_k = nn.Parameter(torch.Tensor(head, d_model, self.d_k))
        # self.W_q = nn.Parameter(torch.Tensor(head, d_model, self.d_k))
        # self.W_v = nn.Parameter(torch.Tensor(head, d_model, self.d_v))
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.scaled_dot_product_attention = ScaleDotProductAttention(self.d_k)
        self.liner = nn.Linear(head * self.d_v, d_model)
        
    def forward(self, x, memory, mask=None):
        
        batch_size, seq_len = x.size(0), x.size(1)
        # # [head, batch_size, seq_len, d_model]
        # q = x.repeat(self.head, 1, 1, 1)
        # k = memory.repeat(self.head, 1, 1, 1)
        # v = memory.repeat(self.head, 1, 1, 1)
        
        # # Scaled dot product 前の線形変換
        # # [head, batch_size, seq_len, d_k]
        # # q = torch.einsum('hijk,hkl->hijl', (q, self.W_q))
        # # k = torch.einsum('hijk,hkl->hijl', (k, self.W_k))
        # # v = torch.einsum('hijk,hkl->hijl', (v, self.W_v))
        # q = self.q_linear(q)
        # k = self.k_linear(k)
        # v = self.v_linear(v)
        
        # # heda数に分割
        # q = q.view(self.head * batch_size, seq_len, self.d_k)
        # k = k.view(self.head * batch_size, seq_len, self.d_k)
        # v = v.view(self.head * batch_size, seq_len, self.d_v)
        
        q = self.split(self.q_linear(x))
        k = self.split(self.k_linear(memory))
        v = self.split(self.v_linear(memory))
        
        if mask is not None:
            mask_size = mask.size()
            mask = mask.view(mask_size[0], 1, mask_size[1], mask_size[2])
            mask = mask.repeat(1, self.head, 1, 1)
        
        # Scaled dot procuct attention
        attention_output = self.scaled_dot_product_attention(q, k , v, mask)
        
        # attention_output = torch.chunk(attention_output, self.head, dim=0)
        # attention_output = torch.cat(attention_output, dim=2)
        attention_output = self.combine(attention_output)

        # 線形変化
        output = self.liner(attention_output)
        return output
    
    def split(self, x):
        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1, self.head, self.d_k)
        return x.transpose(1, 2)
    
    def combine(self, x):
        batch_size, _, _, _ = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, -1, self.d_model)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, dim, n_heads, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()

#         self.n_heads = n_heads
#         self.dim = dim
#         self.dropout = dropout

#         self.q_lin = nn.Linear(dim, dim)
#         self.k_lin = nn.Linear(dim, dim)
#         self.v_lin = nn.Linear(dim, dim)
#         self.out_lin = nn.Linear(dim, dim)

#     def forward(self, x, memory, mask):
#         batch_size, _, dim = x.shape
#         assert dim == self.dim, 'dimension mismatched'

#         dim_per_head = dim // self.n_heads

#         def split(x):
#             x = x.view(batch_size, -1, self.n_heads, dim_per_head)
#             return x.transpose(1, 2)

#         def combine(x):
#             x = x.transpose(1, 2)
#             return x.contiguous().view(batch_size, -1, dim)


#         q = split(self.q_lin(x))
#         k = split(self.k_lin(memory))
#         v = split(self.v_lin(memory))

#         q = q / math.sqrt(dim_per_head)

#         shape = mask.shape
#         mask_shape = (shape[0], 1, shape[1], shape[2]) if mask.dim() == 3 else (shape[0], 1, 1, shape[1])
#         logit = torch.matmul(q, k.transpose(2, 3))
#         logit.masked_fill_(mask.view(mask_shape), -float('inf'))

#         weights = F.softmax(logit, dim=-1)
#         weights = F.dropout(weights, p=self.dropout, training=self.training)

#         output = torch.matmul(weights, v)
#         output = combine(output)

#         return self.out_lin(output)
       

class PositionalEmbeding(nn.Module):
    def __init__(self, max_len, d_model, dropout_rate=0.1):
        super().__init__()
        self.embeddings = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        
    def forward(self, x):
        positions = torch.arange(self.max_len).to(x.device).unsqueeze(0)
        x = x + self.embeddings(positions).expand_as(x)
        return x
 

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_weight = self._initialize_weight()
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_weight[:seq_len, :].unsqueeze(0).to(x.device)
        
    def _get_positional_encoding(self, pos, i):
        w = pos / (10000 ** (((2*i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)
        
    def _initialize_weight(self):
        positional_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_weight).float()
    
    
    
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(relu(self.linear1(x)))


class MLP(nn.Module):
    def __init__(self, d_input, d_hiden, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_input, d_hiden)
        self.linear2 = nn.Linear(d_hiden, d_out)
        
    def forward(self, x):
        return self.linear2(relu(self.linear1(x)))
    
    
class TransfomerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, head_num, dropout_rate, layer_norm_eps):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model, head_num)
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
    def forward(self, x, mask=None):
        # Pre-LN
        x= self.__self_attention_block(self.layer_norm_self_attention(x), mask) + x
        x = self.__feed_forward_block(self.layer_norm_ffn(x)) + x
        
        # # Post-LN
        # x= self.layer_norm_self_attention(self.__self_attention_block(x, mask) + x)
        # x = self.layer_norm_ffn(self.__feed_forward_block(x) + x)
        
        return x
        
    def __self_attention_block(self, x, mask):
        x = self.multi_head_attention(x, x, mask)
        return self.dropout_self_attention(x)
    
    def __feed_forward_block(self, x):
        x = self.ffn(x)
        return self.dropout_ffn(x)
   
 
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, hinshi_size, katuyou_size):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        self.token_embedding =nn.Embedding(vocab_size, d_model, pad_idx)
        self.positional_embeding = PositionalEmbeding(max_len, d_model)
        # self.positional_embeding = AddPositionalEncoding(d_model, max_len)
        self.hinshi_embedding = nn.Embedding(hinshi_size, d_model, padding_idx=2)
        self.katuyou_embeding = nn.Embedding(katuyou_size, d_model, padding_idx=1)
        
        self.layer_norm_emb = nn.LayerNorm(d_model, layer_norm_eps)
        self.layer_norm_out = nn.LayerNorm(d_model, layer_norm_eps)
        
        self.encoder_layers = nn.ModuleList(
            [TransfomerEncoderLayer(d_model, d_ff, heads_num, dropout_rate, layer_norm_eps) for _ in range(N)]
        )
    
    def forward(self, x, hinshi, katuyou, mask=None):
        pad_mask = x==self.pad_idx
        
        x = self.token_embedding(x)
        x = self.positional_embeding(x)
        if hinshi != None:
            x = self.hinshi_embedding(hinshi) + x
        if katuyou != None:
            x = self.katuyou_embeding(katuyou) + x
            
        x = self.layer_norm_emb(x)
        
        x[pad_mask] = 0
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
            x[pad_mask] = 0
        # Pre-LN
        x = self.layer_norm_out(x)
        return x
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads_num, dropout_rate, layer_norm_eps):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.src_tgt_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_src_tgt_attention = nn.Dropout(dropout_rate)
        self.layer_norm_src_tgt_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        
    def forward(self, tgt, src, mask_src_tgt, mask_self):
        # Pre-LN
        tgt = self.__self_attention_block(self.layer_norm_self_attention(tgt), mask_self) + tgt
        x = self.__src_tgt_attention_block(src, self.layer_norm_src_tgt_attention(tgt), mask_src_tgt) + tgt
        x = self.__feed_forward_block(self.layer_norm_ffn(x)) + x
        
        # # Post-LN
        # tgt = self.layer_norm_self_attention(self.__self_attention_block(tgt, mask_self) + tgt)
        # x = self.layer_norm_src_tgt_attention(self.__src_tgt_attention_block(src, tgt, mask_src_tgt) + tgt)
        # x = self.layer_norm_ffn(self.__feed_forward_block(x) + x)
        
        return x
        
    def __src_tgt_attention_block(self, src, tgt, mask):
        output = self.src_tgt_attention(tgt, src, mask)
        return self.dropout_src_tgt_attention(output)
        
    def __self_attention_block(self, x, mask):
        output = self.self_attention(x, x, mask)
        return self.dropout_self_attention(output)
    
    def __feed_forward_block(self, x):
        return self.dropout_ffn(self.ffn(x))
    

class TransformerDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, hinshi_size, katuyou_size):
        super().__init__()
        self.pad_idx = pad_idx
        
        self.token_embeding = nn.Embedding(tgt_vocab_size, d_model, pad_idx)
        self.positional_embeding = PositionalEmbeding(max_len, d_model)
        # self.positional_embeding = AddPositionalEncoding(d_model, max_len)
        self.hinshi_embedding = nn.Embedding(hinshi_size, d_model, padding_idx=2)
        self.katuyou_embeding = nn.Embedding(katuyou_size, d_model, padding_idx=1)
        
        self.layer_norm_emb = nn.LayerNorm(d_model, layer_norm_eps)
        self.layer_norm_out = nn.LayerNorm(d_model, layer_norm_eps)
        
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, d_ff, heads_num, dropout_rate, layer_norm_eps) for _ in range(N)]
        )
        
    def forward(self, tgt, src, hinshi, katuyou, mask_src_tgt, mask_self):
        pad_mask = tgt==self.pad_idx
        
        tgt = self.token_embeding(tgt)
        tgt = self.positional_embeding(tgt)
        if hinshi != None:
            tgt = self.hinshi_embedding(hinshi) + tgt
        if katuyou != None:
            tgt = self.katuyou_embeding(katuyou) + tgt
            
        tgt = self.layer_norm_emb(tgt)
        
        tgt[pad_mask] = 0
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, src, mask_src_tgt, mask_self)
            tgt[pad_mask] = 0
        # Pre-LN
        tgt = self.layer_norm_out(tgt)
        return tgt
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, out_size, hinshi_size=17, max_len=150, d_model=512, heads_num=8, d_ff=2048, enc_layer_num=6, dec_layer_num=6, dropout_rate=0.1, layer_norm_eps=1e-5, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        
        self.encoder = TransformerEncoder(src_vocab_size, max_len, pad_idx, d_model, enc_layer_num, d_ff, heads_num, dropout_rate, layer_norm_eps, hinshi_size)
        self.decoder = TransformerDecoder(tgt_vocab_size, max_len, pad_idx, d_model, dec_layer_num, d_ff, heads_num, dropout_rate, layer_norm_eps, hinshi_size)
        self.linear = nn.Linear(d_model, out_size)
        
        
    def forward(self, src, tgt, hinshi_src=None, hinshi_tgt=None):
        
        pad_mask_src = self._pad_mask(src)
        src = self.encoder(src, hinshi_src, pad_mask_src)
        
        mask_self_attn = torch.logical_or(
            self._subsequent_mask(tgt), self._pad_mask(tgt)
        ) 
        dec_output = self.decoder(tgt, src, hinshi_tgt, pad_mask_src, mask_self_attn)
        output = self.linear(dec_output)
        return F.log_softmax(output, dim=-1)
        # return output
    
    def _pad_mask(self, x):
        seq_len = x.size(1)
        mask = x.eq(self.pad_idx)
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)
        return mask.to(x.device)
    
    def _subsequent_mask(self, x):
        batch_size  =x.size(0)
        max_len = x.size(1)
        return (torch.tril(torch.ones(batch_size, max_len, max_len)).eq(0).to(x.device))
        
        
class KutotenTransformer(nn.Module):
    def __init__(self, src_vocab_size, output_size, max_len=150, d_model=512, heads_num=8, d_ff=2048, N=6, dropout_rate=0.1, layer_norm_eps=1e-5, pad_idx=0, pre_train=True):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps)
        self.pred_head = nn.Linear(d_model, output_size-1)
        self.layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        # self.ward_pred = MaskedWordPredictions(output_size, d_model)
        self.pred_head_FT = nn.Linear(d_model, output_size)
        self.pad_idx = pad_idx
        self.pre_train = pre_train
        
    def forward(self, src):
        pad_mask_src = self._pad_mask(src)
        src = self.encoder(src, pad_mask_src)
        src = self.layer_norm(src)
        
        if self.pre_train:
            output = self.pred_head(src)
        else:
            output = self.pred_head_FT(src)

        
        return F.log_softmax(output, dim=-1)
    
    def _pad_mask(self, x):
        seq_len = x.size(1)
        mask = x.eq(self.pad_idx)
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)
        return mask.to(x.device)
    
        
class MaskedWordPredictions(nn.Module):
    def __init__(self, tgt_vocab_size, d_model):
        super().__init__()
        
        self.layer = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, x):
        x = self.layer(x)
        
        return F.softmax(x, dim=-1)
    
    # def get_mask(self, x):
        
    
class LSTM_Encoder(nn.Module):
    def __init__(self, d_model, vocab_size, hinshi_size, katuyou_size, pad_idx, dropout_rate=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.token_embedding =nn.Embedding(vocab_size, d_model, pad_idx)
        self.hinshi_embedding = nn.Embedding(hinshi_size, d_model, padding_idx=2)
        self.katuyou_embedding = nn.Embedding(katuyou_size, d_model, padding_idx=1)
        
        self.bilstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        
        self.layer_norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(d_model*2, eps=layer_norm_eps)
        self.layer_norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.linear = nn.Linear(d_model*2, d_model)
        
    def forward(self, x, hinshi, katuyou):
        x = self.token_embedding(x)
        x = self.katuyou_embedding(katuyou) + x
        x = self.hinshi_embedding(hinshi) + x
        
        x = self.layer_norm1(x)
        x, _ = self.bilstm(x)
        x = self.linear(x)
        
        return self.layer_norm_out(x)
        
    
class KutotenModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, out_size, katuyou_size, hinshi_size=17, max_len=150, d_model=512, heads_num=8, d_ff=2048, enc_layer_num=6, dec_layer_num=6, dropout_rate=0.1, layer_norm_eps=1e-5, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        
        self.lstm = LSTM_Encoder(d_model, src_vocab_size, hinshi_size, katuyou_size, pad_idx)
        self.encoder = TransformerEncoder(src_vocab_size, max_len, pad_idx, d_model, enc_layer_num, d_ff, heads_num, dropout_rate, layer_norm_eps, hinshi_size, katuyou_size)
        self.decoder = TransformerDecoder(tgt_vocab_size, max_len, pad_idx, d_model, dec_layer_num, d_ff, heads_num, dropout_rate, layer_norm_eps, hinshi_size, katuyou_size)
        self.linear = nn.Linear(d_model, out_size)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.mlp = MLP(d_input=d_model*2, d_hiden=d_ff, d_out=d_model)
        
    def forward(self, src, tgt, hinshi_src=None, hinshi_tgt=None, katuyou_src=None, katuyou_tgt=None):
        
        pad_mask_src = self._pad_mask(src)
        hiden = self.lstm(src, hinshi_src, katuyou_src)
        src = self.encoder(src, hinshi_src, katuyou_src, mask=pad_mask_src)
        src = torch.cat([src, hiden], axis=-1)
        src = self.mlp(src)
        src = self.layer_norm1(src)
        
        mask_self_attn = torch.logical_or(
            self._subsequent_mask(tgt), self._pad_mask(tgt)
        ) 
        dec_output = self.decoder(tgt, src, hinshi_tgt, katuyou_tgt, mask_src_tgt=pad_mask_src, mask_self=mask_self_attn)
        dec_output = self.layer_norm2(dec_output)
        output = self.linear(dec_output)
        return F.log_softmax(output, dim=-1)
        # return output
    
    def _pad_mask(self, x):
        seq_len = x.size(1)
        mask = x.eq(self.pad_idx)
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)
        return mask.to(x.device)
    
    def _subsequent_mask(self, x):
        batch_size  =x.size(0)
        max_len = x.size(1)
        return (torch.tril(torch.ones(batch_size, max_len, max_len)).eq(0).to(x.device))
