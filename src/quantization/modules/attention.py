import torch
import torch.nn as nn
import torch.nn.functional as F

from .qlinear import LSQ_input
from src.deit_vision_transformer import Attention as deit_attention
from ..modules.qbias import LearnableBias
from .qlinear import QLinear, LSQ_w_and_act_QLinear
from ..quantizer.lsq import LsqQuantizer, LsqQuantizer4v
from ..quantizer.statsq import StatsQuantizer, StatsQuantizer_specific_4_qkreparam_cga

class QAttention(deit_attention):
    def __init__(self, m: deit_attention, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):
        assert type(m) == deit_attention
        super().__init__(
            dim = m.qkv.in_features,
            num_heads=m.num_heads,
            attn_drop=m.attn_drop.p,
            proj_drop=m.proj_drop.p,
            qqkkvv= m.qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.input_channelwise = input_channelwise

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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x) # B, N, 3*C
        if self.input_bits < 32:
            qkv = self.move_qkv_b4(qkv)
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4) # 3, B, num_heads, N, C // self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2] # B, num_heads, N, C//self.num_heads

        q = self.quan_a_q_fn(q)  # quantize along N
        k = self.quan_a_k_fn(k)  # quantize along N
        
        v = v.permute(0,2,1,3).reshape(B,N,C)
        v = self.quan_a_v_fn(v)  # quantize along C
        v = v.reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        if self.input_bits < 32:

            q = q.permute(0, 2, 1, 3).reshape(B, N, C)
            k = k.permute(0, 2, 1, 3).reshape(B, N, C)
            v = v.permute(0, 2, 1, 3).reshape(B, N, C)
            q = self.move_q_aft(q)
            k = self.move_k_aft(k)
            v = self.move_v_aft(v)
        
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, num_heads, N, C // self.num_heads
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn_weights = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn_prob = F.softmax(attn_weights, dim=-1)

        attn_prob = self.quan_a_softmax_fn(attn_prob)
        attn_prob = self.attn_drop(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

class QAttention_qkreparam(deit_attention):
    def __init__(self, m: deit_attention, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):
        assert type(m) == deit_attention
        super().__init__(
            dim = m.qkv.in_features,
            num_heads=m.num_heads,
            attn_drop=m.attn_drop.p,
            proj_drop=m.proj_drop.p,
            qqkkvv= m.qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.input_channelwise = input_channelwise
        
        self.quant_x_4_qkv =  LSQ_input(bit = input_bits, all_positive= False, learnable= aq_learnable, learanbaleBiasdim=m.qkv.in_features)

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
        self.v_quant = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable)#.to(m.weight.device)
        
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

    def forward(self, x):
        B, N, C = x.shape
        ## first quantize input x
        quant_x = self.quant_x_4_qkv(x)
        ## V 
        quant_v_weight = self.v_quant(self.v.weight)
        v_out = nn.functional.linear(quant_x, quant_v_weight)
        v_out += self.v.bias.view(1, -1).expand_as(v_out) # B,N,C
    
        ## TO MULTI_HEAD V
        v_out = self.move_v_b4(v_out)
        v_out = self.quan_a_v_fn(v_out)    
        v_out = self.move_v_aft(v_out) # B, N, C
        v = v_out.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, num_heads, N, C // num_heads
        
        ## TO MULTI_HEAD QK
        multi_head_q_weight = self.q.weight.reshape(self.num_heads,self.q.out_features // self.num_heads, self.q.in_features) # out_feat , in_feat -> num_heads, out_feat // num_heads, in_feat
        multi_head_k_weight = self.k.weight.reshape(self.num_heads,self.k.out_features // self.num_heads, self.k.in_features)
        
        multi_head_qk = multi_head_q_weight.transpose(-2, -1).contiguous() @ multi_head_k_weight # num_heads, in_features, in_features
        multi_head_qk = multi_head_qk.reshape(self.num_heads*self.q.out_features,self.q.in_features)# num_heads*in_features, in_features
        multi_head_qk_qunat = self.qk_quant(multi_head_qk)
        multi_head_qk_qunat = multi_head_qk_qunat.reshape(self.num_heads, self.q.in_features,self.q.in_features) #num_heads, in_features, in_features
        
        ## W_qk@X^T torch.einsum('BNC,BACD -> BAND',quant_x,quant_qkx)
        # quant_x: B,C,N
        qkx =  torch.einsum('HDC, BCN -> BHDN', multi_head_qk_qunat, quant_x.transpose(-2, -1).contiguous() ) # B, num_heads, in_features, N
        qkx = qkx.permute(0,3,1,2).reshape(B,N, self.num_heads * C)     #B, N, num_heads*in_features
        qkx = self.move_qkx_b4(qkx) 
        qkx = qkx.reshape(B,N*self.num_heads, C)
        quant_qkx = self.quan_a_qkx_fn(qkx) # B, num_heads*N, in_features
        quant_qkx = quant_qkx.reshape(B,N,self.num_heads * C)      #B, N, num_heads*in_features
        quant_qkx = self.move_qkx_aft(quant_qkx) #B, N, num_heads*in_features
        quant_qkx = quant_qkx.reshape(B, N, self.num_heads, -1).permute(0, 2, 3, 1) # B, num_heads, in_features, N
        
        ## x@W_qk@X^T, quant_x: B,N,C and quant_qkx: B,num_heads, C, N
        xqkx = torch.einsum('BNC,BHCD -> BHND',quant_x,quant_qkx) # B, num_heads, N, N
        # B, num_heads, N, C // self.num_heads
        value_4_softmax = xqkx
        attn_weights = (value_4_softmax) * self.scale
        attn_prob = F.softmax(attn_weights, dim=-1)

        attn_prob = self.quan_a_softmax_fn(attn_prob)
        attn_prob = self.attn_drop(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

class QAttention_qkreparam_4_cga(deit_attention):
    def __init__(self, m: deit_attention, clip_val=2.5, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False, boundaryRange = 0.005,
                 **kwargs):
        assert type(m) == deit_attention
        super().__init__(
            dim = m.qkv.in_features,
            num_heads=m.num_heads,
            attn_drop=m.attn_drop.p,
            proj_drop=m.proj_drop.p,
            qqkkvv= m.qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.input_channelwise = input_channelwise
        
        self.quant_x_4_qkv =  LSQ_input(bit = input_bits, all_positive= False, learnable= aq_learnable, learanbaleBiasdim=m.qkv.in_features)

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
        
        self.qk_quant = StatsQuantizer_specific_4_qkreparam_cga(num_bits=self.weight_bits, clip_learnable=wq_learnable,boundaryRange=boundaryRange) # num_heads*in_features, in_features
        self.v_quant = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable)#.to(m.weight.device)
        
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

    def forward(self, x):
        B, N, C = x.shape
        ## first quantize input x
        quant_x = self.quant_x_4_qkv(x)
        ## V 
        quant_v_weight = self.v_quant(self.v.weight)
        v_out = nn.functional.linear(quant_x, quant_v_weight)
        v_out += self.v.bias.view(1, -1).expand_as(v_out) # B,N,C
    
        ## TO MULTI_HEAD V
        v_out = self.move_v_b4(v_out)
        v_out = self.quan_a_v_fn(v_out)    
        v_out = self.move_v_aft(v_out) # B, N, C
        v = v_out.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, num_heads, N, C // num_heads
        
        ## TO MULTI_HEAD QK
        multi_head_q_weight = self.q.weight.reshape(self.num_heads,self.q.out_features // self.num_heads, self.q.in_features) # out_feat , in_feat -> num_heads, out_feat // num_heads, in_feat
        multi_head_k_weight = self.k.weight.reshape(self.num_heads,self.k.out_features // self.num_heads, self.k.in_features)
        
        multi_head_qk = multi_head_q_weight.transpose(-2, -1).contiguous() @ multi_head_k_weight # num_heads, in_features, in_features
        multi_head_qk = multi_head_qk.reshape(self.num_heads*self.q.out_features,self.q.in_features)# num_heads*in_features, in_features
        multi_head_qk_qunat = self.qk_quant(multi_head_qk)
        multi_head_qk_qunat = multi_head_qk_qunat.reshape(self.num_heads, self.q.in_features,self.q.in_features) #num_heads, in_features, in_features
        
        ## W_qk@X^T torch.einsum('BNC,BACD -> BAND',quant_x,quant_qkx)
        # quant_x: B,C,N
        qkx =  torch.einsum('HDC, BCN -> BHDN', multi_head_qk_qunat, quant_x.transpose(-2, -1).contiguous() ) # B, num_heads, in_features, N
        qkx = qkx.permute(0,3,1,2).reshape(B,N, self.num_heads * C)     #B, N, num_heads*in_features
        qkx = self.move_qkx_b4(qkx) 
        qkx = qkx.reshape(B,N*self.num_heads, C)
        quant_qkx = self.quan_a_qkx_fn(qkx) # B, num_heads*N, in_features
        quant_qkx = quant_qkx.reshape(B,N,self.num_heads * C)      #B, N, num_heads*in_features
        quant_qkx = self.move_qkx_aft(quant_qkx) #B, N, num_heads*in_features
        quant_qkx = quant_qkx.reshape(B, N, self.num_heads, -1).permute(0, 2, 3, 1) # B, num_heads, in_features, N
        
        ## x@W_qk@X^T, quant_x: B,N,C and quant_qkx: B,num_heads, C, N
        xqkx = torch.einsum('BNC,BHCD -> BHND',quant_x,quant_qkx) # B, num_heads, N, N
        # B, num_heads, N, C // self.num_heads
        value_4_softmax = xqkx
        attn_weights = (value_4_softmax) * self.scale
        attn_prob = F.softmax(attn_weights, dim=-1)

        attn_prob = self.quan_a_softmax_fn(attn_prob)
        attn_prob = self.attn_drop(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

class QAttention_lsq(deit_attention):
    def __init__(self, m: deit_attention, clip_val=2.5, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="lsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):
        assert type(m) == deit_attention
        super().__init__(
            dim = m.qkv.in_features,
            num_heads=m.num_heads,
            attn_drop=m.attn_drop.p,
            proj_drop=m.proj_drop.p,
            qqkkvv= m.qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.input_channelwise = input_channelwise

        self.qkv = LSQ_w_and_act_QLinear(
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
        self.proj = LSQ_w_and_act_QLinear(
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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x) # B, N, 3*C
        if self.input_bits < 32:
            qkv = self.move_qkv_b4(qkv)
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4) # 3, B, num_heads, N, C // self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.quan_a_q_fn(q)  
        k = self.quan_a_k_fn(k)
        v = v.permute(0,2,1,3).reshape(B,N,C)
        v = self.quan_a_v_fn(v)  # quantize along C
        v = v.reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)



        if self.input_bits < 32:

            q = q.permute(0, 2, 1, 3).reshape(B, N, C)
            k = k.permute(0, 2, 1, 3).reshape(B, N, C)
            v = v.permute(0, 2, 1, 3).reshape(B, N, C)
            q = self.move_q_aft(q)
            k = self.move_k_aft(k)
            v = self.move_v_aft(v)
        
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, num_heads, N, C // self.num_heads
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        attn_weights = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn_prob = F.softmax(attn_weights, dim=-1)

        attn_prob = self.quan_a_softmax_fn(attn_prob)
        attn_prob = self.attn_drop(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, None

