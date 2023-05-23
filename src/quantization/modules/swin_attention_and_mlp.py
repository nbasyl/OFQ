import copy
import imp
import os
from re import S
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import Attention
from .qlinear import LSQ_input
from src.deit_vision_transformer import Attention as deit_attention
from ..modules.qbias import LearnableBias
from .qlinear import QLinear
from ..quantizer.lsq import LsqQuantizer, LsqQuantizer4v
from ..quantizer.statsq import StatsQuantizer, StatsQuantizer_specific_4_qkreparam_cga

import math

from src.swin import ShiftedWindowAttention
from torchvision.ops.misc import MLP as swin_MLP
from timm.models.layers import to_2tuple

class QMLP_swin(torch.nn.Module):
    def __init__(self, *kargs, m: swin_MLP, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                    weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq", act_layer=nn.GELU,
                    pretrained_initialized = False,
                    **kwargs):
            super(QMLP_swin, self).__init__()

            self.fc1 = QLinear(m = m[0],weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=True,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)

            self.act_layer = act_layer
            
            if act_layer != 'rprelu':
                if act_layer != 'None':
                    self.act = act_layer()
                else:
                    self.act = nn.Identity()
                

            self.drop1 = m[2]
            self.fc2 = QLinear(m = m[3],weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=False,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)
            self.drop2 = m[4]

    def forward(self, x):
        
        x = self.fc1(x)
        if self.act_layer != 'rprelu':
            x = self.act(x)
        else:
            x = self.move1(x)
            x = self.act(x)
            x = self.move2(x)

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class QAttention_swin(ShiftedWindowAttention):
    def __init__(self, m: ShiftedWindowAttention, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):

        assert type(m) == ShiftedWindowAttention
        super().__init__(
        dim = m.dim,
        window_size = m.window_size,
        shift_size =  m.shift_size,
        num_heads = m.num_heads,
        qkv_bias = True,
        proj_bias = True,
        attention_dropout = 0.0,
        dropout = 0.0,
        qqkkvv = m.qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.input_channelwise = input_channelwise
        self.attention_dropout = 0.0
        self.dropout = 0.0

        self.qkv = QLinear(
            m = self.qkv,
            weight_bits = weight_bits,
            input_bits = input_bits,
            weight_channelwise = weight_channelwise,
            input_channelwise = input_channelwise,
            weight_quant_method = weight_quant_method,
            input_quant_method = input_quant_method,
            aq_learnable = aq_learnable, ## act 
            wq_learnable = wq_learnable,## weight
            symmetric = True, ## act
            pretrained_initialized = pretrained_initialized
        )
        self.proj = QLinear(
            m = self.proj,
            weight_bits = weight_bits,
            input_bits = input_bits,
            weight_channelwise = weight_channelwise,
            input_channelwise = input_channelwise,
            weight_quant_method = weight_quant_method,
            input_quant_method = input_quant_method,
            aq_learnable = aq_learnable, ## act 
            wq_learnable = wq_learnable,## weight
            symmetric = True, ## act
            pretrained_initialized = pretrained_initialized
        )

        self.quan_a_q_fn = LsqQuantizer(bit=input_bits,all_positive=False,per_channel=True, learnable =  aq_learnable)
        self.quan_a_k_fn = LsqQuantizer(bit=input_bits,all_positive=False,per_channel=True, learnable =  aq_learnable)
        self.quan_a_v_fn = LsqQuantizer4v(bit=input_bits,all_positive=False,per_channel=True, learnable =  aq_learnable)

        self.move_qkv_b4 = LearnableBias(m.qkv.in_features*3)
        self.move_q_aft = LearnableBias(m.qkv.in_features)
        self.move_k_aft = LearnableBias(m.qkv.in_features)
        self.move_v_aft = LearnableBias(m.qkv.in_features)

        self.quan_a_softmax_fn = LsqQuantizer(bit=input_bits,all_positive=True,per_channel=True, learnable =  aq_learnable)

        ## swin components
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        

    def forward(self, x):

        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        
        B, H, W, C = x.shape
        # pad feature maps to multiples of window size
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        # cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x = x.view(B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)  # B*nW, Ws*Ws, C

        # multi-head attention
        qkv = self.qkv(x)
        
        if self.input_bits < 32:
            qkv = self.move_qkv_b4(qkv)
        
        qkv = qkv.reshape(x.size(0), x.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.quan_a_q_fn(q)  # quantize along N
        k = self.quan_a_k_fn(k)  # quantize along N
        
        v = v.permute(0,2,1,3).reshape(B*num_windows,self.window_size[0] * self.window_size[1],C)
        v = self.quan_a_v_fn(v)  # quantize along C
        v = v.reshape(B*num_windows,self.window_size[0] * self.window_size[1],self.num_heads,C//self.num_heads).permute(0,2,1,3)

        if self.input_bits < 32:

            q = q.permute(0,2,1,3).reshape(B*num_windows,self.window_size[0] * self.window_size[1],C)
            k = k.permute(0,2,1,3).reshape(B*num_windows,self.window_size[0] * self.window_size[1],C)
            v = v.permute(0,2,1,3).reshape(B*num_windows,self.window_size[0] * self.window_size[1],C)
            q = self.move_q_aft(q)
            k = self.move_k_aft(k)
            v = self.move_v_aft(v)
        
        q = q.reshape(B*num_windows,self.window_size[0] * self.window_size[1],self.num_heads,C//self.num_heads).permute(0,2,1,3) # B, num_heads, N, C // self.num_heads
        k = k.reshape(B*num_windows,self.window_size[0] * self.window_size[1],self.num_heads,C//self.num_heads).permute(0,2,1,3)
        v = v.reshape(B*num_windows,self.window_size[0] * self.window_size[1],self.num_heads,C//self.num_heads).permute(0,2,1,3)
        
        
        attn = q.matmul(k.transpose(-2, -1)) * ((C // self.num_heads) ** -0.5)
        # add relative position bias
        attn = attn + relative_position_bias

        if sum(self.shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -self.shift_size[0]), (-self.shift_size[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -self.shift_size[1]), (-self.shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = self.quan_a_softmax_fn(attn)
        attn = F.dropout(attn, p=self.attention_dropout)

        x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout)

        # reverse windows
        x = x.view(B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        if self.qqkkvv:
            q_score = torch.matmul(q, q.transpose(-1, -2))
            q_score = q_score / math.sqrt(C // self.num_heads)
            k_score = torch.matmul(k, k.transpose(-1, -2))
            k_score = k_score / math.sqrt(C // self.num_heads)
            v_score = torch.matmul(v, v.transpose(-1, -2))
            v_score = v_score / math.sqrt(C // self.num_heads)
            
            return x, (attn, q_score, k_score, v_score)
        else:
            return x, None
        
class QAttention_swin_qkreparam(ShiftedWindowAttention):
    def __init__(self, m: ShiftedWindowAttention, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):

        assert type(m) == ShiftedWindowAttention
        super().__init__(
        dim = m.dim,
        window_size = m.window_size,
        shift_size =  m.shift_size,
        num_heads = m.num_heads,
        qkv_bias = True,
        proj_bias = True,
        attention_dropout = 0.0,
        dropout = 0.0,
        qqkkvv = m.qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.input_channelwise = input_channelwise
        self.attention_dropout = 0.0
        self.dropout = 0.0
        
        self.quant_x_4_qkv =  LSQ_input(bit = input_bits, all_positive= (symmetric==False), learnable= aq_learnable, learanbaleBiasdim=m.qkv.in_features)

        
        self.q = nn.Linear(in_features=m.qkv.in_features,out_features=m.qkv.in_features, bias=False)
        self.k = nn.Linear(in_features=m.qkv.in_features,out_features=m.qkv.in_features, bias= False)
        self.v = nn.Linear(in_features=m.qkv.in_features,out_features=m.qkv.in_features)

        if pretrained_initialized:
            with torch.no_grad():
                q_k_v_dim = int(m.qkv.weight.shape[0]/3)
                copy_weight = m.qkv.weight.detach()
                copy_bias = m.qkv.bias.detach()
                self.q.weight.copy_(copy_weight[:q_k_v_dim*1,:])
                self.k.weight.copy_(copy_weight[q_k_v_dim*1:q_k_v_dim*2,:])
                self.v.weight.copy_(copy_weight[q_k_v_dim*2:q_k_v_dim*3,:])
                self.v.bias.copy_(copy_bias[q_k_v_dim*2:q_k_v_dim*3])
        
        self.qk_quant = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable) # num_heads*in_features, in_features
        self.v_quant = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable) #.to(m.weight.device)

        self.proj = QLinear(
            m = self.proj,
            weight_bits = weight_bits,
            input_bits = input_bits,
            weight_channelwise = weight_channelwise,
            input_channelwise = input_channelwise,
            weight_quant_method = weight_quant_method,
            input_quant_method = input_quant_method,
            aq_learnable = aq_learnable, ## act 
            wq_learnable = wq_learnable,## weight
            symmetric = True, ## act
            pretrained_initialized = pretrained_initialized
        )

        ## THIS IS FOR QK PART
        self.quan_a_qkx_fn = LsqQuantizer(bit=input_bits,all_positive=False,per_channel=True, learnable =  aq_learnable) # for W_q*W_k*X -> B, num_heads, N, C//numheads
        self.quan_a_v_fn = LsqQuantizer4v(bit=input_bits,all_positive=False,per_channel=True, learnable =  aq_learnable)

        # for W_q*W_k*X
        self.move_qkx_b4 = LearnableBias(self.num_heads * m.qkv.in_features)
        self.move_qkx_aft = LearnableBias(self.num_heads * m.qkv.in_features)
        
        # for v
        self.move_v_b4 = LearnableBias(m.qkv.in_features)
        self.move_v_aft = LearnableBias(m.qkv.in_features)

        self.quan_a_softmax_fn = LsqQuantizer(bit=input_bits,all_positive=True,per_channel=True, learnable =  aq_learnable)
        
        ## no longer need qkv
        del self.qkv

        ## swin components
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        

    def forward(self, x):

        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        
        B, H, W, C = x.shape
        # pad feature maps to multiples of window size
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        # cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x = x.view(B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)  # B*nW, Ws*Ws, C

        # multi-head attention
        ## first quantize input x
        quant_x = self.quant_x_4_qkv(x)
        ## V 
        quant_v_weight = self.v_quant(self.v.weight)
        v_out = nn.functional.linear(quant_x, quant_v_weight)
        v_out += self.v.bias.view(1, -1).expand_as(v_out) # B*nW, Ws*Ws, C
        ## TO MULTI_HEAD V
        v_out = self.move_v_b4(v_out)
        v_out = self.quan_a_v_fn(v_out)    
        v_out = self.move_v_aft(v_out) # B*nW, Ws*Ws, C
        v = v_out.reshape(B*num_windows,self.window_size[0] * self.window_size[1],self.num_heads,C//self.num_heads).permute(0,2,1,3)
        
        ## TO MULTI_HEAD QK
        multi_head_q_weight = self.q.weight.reshape(self.num_heads,self.q.out_features // self.num_heads, self.q.in_features) # out_feat , in_feat -> num_heads, out_feat // num_heads, in_feat
        multi_head_k_weight = self.k.weight.reshape(self.num_heads,self.k.out_features // self.num_heads, self.k.in_features)
        
        multi_head_qk = multi_head_q_weight.transpose(-2, -1).contiguous() @ multi_head_k_weight # num_heads, in_features, in_features
        multi_head_qk = multi_head_qk.reshape(self.num_heads*self.q.out_features,self.q.in_features)# num_heads*in_features, in_features
        multi_head_qk_qunat = self.qk_quant(multi_head_qk)
        multi_head_qk_qunat = multi_head_qk_qunat.reshape(self.num_heads, self.q.in_features,self.q.in_features) #num_heads, in_features, in_features


        ## W_qk@X^T torch.einsum('BNC,BACD -> BAND',quant_x,quant_qkx)
        # quant_x: B*nW, Ws*Ws, C
        qkx =  torch.einsum('HDC, BCN -> BHDN', multi_head_qk_qunat, quant_x.transpose(-2, -1).contiguous() ) # B*nW, num_heads, in_features, Ws*Ws
        qkx = qkx.permute(0,3,1,2).reshape(B * num_windows,self.window_size[0] * self.window_size[1], self.num_heads * C)     #B*nW, Ws*Ws, num_heads*in_features
        qkx = self.move_qkx_b4(qkx) 
        qkx = qkx.reshape(B * num_windows,self.window_size[0] * self.window_size[1]*self.num_heads, C)
        quant_qkx = self.quan_a_qkx_fn(qkx) # B, num_heads*N, in_features
        quant_qkx = quant_qkx.reshape(B * num_windows,self.window_size[0] * self.window_size[1], self.num_heads * C)      #B, N, num_heads*in_features
        quant_qkx = self.move_qkx_aft(quant_qkx) #B, N, num_heads*in_features
        quant_qkx = quant_qkx.reshape(B * num_windows, self.window_size[0] * self.window_size[1], self.num_heads, -1).permute(0, 2, 3, 1) # B, num_heads, in_features, N
        
        ## x@W_qk@X^T, quant_x: B,N,C and quant_qkx: B,num_heads, C, N
        xqkx = torch.einsum('BNC,BHCD -> BHND',quant_x,quant_qkx) # B*nW, num_heads, Ws*Ws, Ws*Ws
        # B, num_heads, N, C // self.num_heads
        value_4_softmax = xqkx
        attn = value_4_softmax * ((C // self.num_heads) ** -0.5)
        # add relative position bias
        attn = attn + relative_position_bias

        if sum(self.shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -self.shift_size[0]), (-self.shift_size[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -self.shift_size[1]), (-self.shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = self.quan_a_softmax_fn(attn)
        attn = F.dropout(attn, p=self.attention_dropout)

        x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout)

        # reverse windows
        x = x.view(B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        if self.qqkkvv:
            q_score = torch.matmul(q, q.transpose(-1, -2))
            q_score = q_score / math.sqrt(C // self.num_heads)
            k_score = torch.matmul(k, k.transpose(-1, -2))
            k_score = k_score / math.sqrt(C // self.num_heads)
            v_score = torch.matmul(v, v.transpose(-1, -2))
            v_score = v_score / math.sqrt(C // self.num_heads)
            
            return x, (attn, q_score, k_score, v_score)
        else:
            return x, None
        
class QAttention_swin_qkreparam_4_cga(ShiftedWindowAttention):
    def __init__(self, m: ShiftedWindowAttention, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,  boundaryRange = 0.005,
                 **kwargs):

        assert type(m) == ShiftedWindowAttention
        super().__init__(
        dim = m.dim,
        window_size = m.window_size,
        shift_size =  m.shift_size,
        num_heads = m.num_heads,
        qkv_bias = True,
        proj_bias = True,
        attention_dropout = 0.0,
        dropout = 0.0,
        qqkkvv = m.qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.input_channelwise = input_channelwise
        self.attention_dropout = 0.0
        self.dropout = 0.0
        
        self.quant_x_4_qkv =  LSQ_input(bit = input_bits, all_positive= (symmetric==False), learnable= aq_learnable, learanbaleBiasdim=m.qkv.in_features)

        
        self.q = nn.Linear(in_features=m.qkv.in_features,out_features=m.qkv.in_features, bias=False)
        self.k = nn.Linear(in_features=m.qkv.in_features,out_features=m.qkv.in_features, bias= False)
        self.v = nn.Linear(in_features=m.qkv.in_features,out_features=m.qkv.in_features)

        if pretrained_initialized:
            with torch.no_grad():
                q_k_v_dim = int(m.qkv.weight.shape[0]/3)
                copy_weight = m.qkv.weight.detach()
                copy_bias = m.qkv.bias.detach()
                self.q.weight.copy_(copy_weight[:q_k_v_dim*1,:])
                self.k.weight.copy_(copy_weight[q_k_v_dim*1:q_k_v_dim*2,:])
                self.v.weight.copy_(copy_weight[q_k_v_dim*2:q_k_v_dim*3,:])
                self.v.bias.copy_(copy_bias[q_k_v_dim*2:q_k_v_dim*3])
        
        self.qk_quant = StatsQuantizer_specific_4_qkreparam_cga(num_bits=self.weight_bits, clip_learnable=wq_learnable, boundaryRange=boundaryRange) # num_heads*in_features, in_features
        self.v_quant = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable) #.to(m.weight.device)

        self.proj = QLinear(
            m = self.proj,
            weight_bits = weight_bits,
            input_bits = input_bits,
            weight_channelwise = weight_channelwise,
            input_channelwise = input_channelwise,
            weight_quant_method = weight_quant_method,
            input_quant_method = input_quant_method,
            aq_learnable = aq_learnable, ## act 
            wq_learnable = wq_learnable,## weight
            symmetric = True, ## act
            pretrained_initialized = pretrained_initialized
        )

        ## THIS IS FOR QK PART
        self.quan_a_qkx_fn = LsqQuantizer(bit=input_bits,all_positive=False,per_channel=True, learnable =  aq_learnable) # for W_q*W_k*X -> B, num_heads, N, C//numheads
        self.quan_a_v_fn = LsqQuantizer4v(bit=input_bits,all_positive=False,per_channel=True, learnable =  aq_learnable)

        # for W_q*W_k*X
        self.move_qkx_b4 = LearnableBias(self.num_heads * m.qkv.in_features)
        self.move_qkx_aft = LearnableBias(self.num_heads * m.qkv.in_features)
        
        # for v
        self.move_v_b4 = LearnableBias(m.qkv.in_features)
        self.move_v_aft = LearnableBias(m.qkv.in_features)

        self.quan_a_softmax_fn = LsqQuantizer(bit=input_bits,all_positive=True,per_channel=True, learnable =  aq_learnable)
        
        ## no longer need qkv
        del self.qkv

        ## swin components
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        

    def forward(self, x):

        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        
        B, H, W, C = x.shape
        # pad feature maps to multiples of window size
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        # cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x = x.view(B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)  # B*nW, Ws*Ws, C

        # multi-head attention
        ## first quantize input x
        quant_x = self.quant_x_4_qkv(x)
        ## V 
        quant_v_weight = self.v_quant(self.v.weight)
        v_out = nn.functional.linear(quant_x, quant_v_weight)
        v_out += self.v.bias.view(1, -1).expand_as(v_out) # B*nW, Ws*Ws, C
        ## TO MULTI_HEAD V
        v_out = self.move_v_b4(v_out)
        v_out = self.quan_a_v_fn(v_out)    
        v_out = self.move_v_aft(v_out) # B*nW, Ws*Ws, C
        v = v_out.reshape(B*num_windows,self.window_size[0] * self.window_size[1],self.num_heads,C//self.num_heads).permute(0,2,1,3)
        
        ## TO MULTI_HEAD QK
        multi_head_q_weight = self.q.weight.reshape(self.num_heads,self.q.out_features // self.num_heads, self.q.in_features) # out_feat , in_feat -> num_heads, out_feat // num_heads, in_feat
        multi_head_k_weight = self.k.weight.reshape(self.num_heads,self.k.out_features // self.num_heads, self.k.in_features)
        
        multi_head_qk = multi_head_q_weight.transpose(-2, -1).contiguous() @ multi_head_k_weight # num_heads, in_features, in_features
        multi_head_qk = multi_head_qk.reshape(self.num_heads*self.q.out_features,self.q.in_features)# num_heads*in_features, in_features
        multi_head_qk_qunat = self.qk_quant(multi_head_qk)
        multi_head_qk_qunat = multi_head_qk_qunat.reshape(self.num_heads, self.q.in_features,self.q.in_features) #num_heads, in_features, in_features


        ## W_qk@X^T torch.einsum('BNC,BACD -> BAND',quant_x,quant_qkx)
        # quant_x: B*nW, Ws*Ws, C
        qkx =  torch.einsum('HDC, BCN -> BHDN', multi_head_qk_qunat, quant_x.transpose(-2, -1).contiguous() ) # B*nW, num_heads, in_features, Ws*Ws
        qkx = qkx.permute(0,3,1,2).reshape(B * num_windows,self.window_size[0] * self.window_size[1], self.num_heads * C)     #B*nW, Ws*Ws, num_heads*in_features
        qkx = self.move_qkx_b4(qkx) 
        qkx = qkx.reshape(B * num_windows,self.window_size[0] * self.window_size[1]*self.num_heads, C)
        quant_qkx = self.quan_a_qkx_fn(qkx) # B, num_heads*N, in_features
        quant_qkx = quant_qkx.reshape(B * num_windows,self.window_size[0] * self.window_size[1], self.num_heads * C)      #B, N, num_heads*in_features
        quant_qkx = self.move_qkx_aft(quant_qkx) #B, N, num_heads*in_features
        quant_qkx = quant_qkx.reshape(B * num_windows, self.window_size[0] * self.window_size[1], self.num_heads, -1).permute(0, 2, 3, 1) # B, num_heads, in_features, N
        
        ## x@W_qk@X^T, quant_x: B,N,C and quant_qkx: B,num_heads, C, N
        xqkx = torch.einsum('BNC,BHCD -> BHND',quant_x,quant_qkx) # B*nW, num_heads, Ws*Ws, Ws*Ws
        # B, num_heads, N, C // self.num_heads
        value_4_softmax = xqkx
        attn = value_4_softmax * ((C // self.num_heads) ** -0.5)
        # add relative position bias
        attn = attn + relative_position_bias

        if sum(self.shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -self.shift_size[0]), (-self.shift_size[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -self.shift_size[1]), (-self.shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = self.quan_a_softmax_fn(attn)
        attn = F.dropout(attn, p=self.attention_dropout)

        x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout)

        # reverse windows
        x = x.view(B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        if self.qqkkvv:
            q_score = torch.matmul(q, q.transpose(-1, -2))
            q_score = q_score / math.sqrt(C // self.num_heads)
            k_score = torch.matmul(k, k.transpose(-1, -2))
            k_score = k_score / math.sqrt(C // self.num_heads)
            v_score = torch.matmul(v, v.transpose(-1, -2))
            v_score = v_score / math.sqrt(C // self.num_heads)
            
            return x, (attn, q_score, k_score, v_score)
        else:
            return x, None
        
 
        

