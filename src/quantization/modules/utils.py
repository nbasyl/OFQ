from email.policy import default
import os
import torch
# from .conv import QConv2d, QConvBn2d
# from .linear import QLinear
from .qlinear import LSQ_QConv2d
from .qlinear import QLinear, QMLP, LSQ_w_and_act_QMLP, LSQ_w_and_act_QLinear, LSQ_QLinear4head
from .attention import QAttention, QAttention_lsq
from .attention import QAttention_qkreparam_4_cga
from .attention import QAttention_qkreparam

# from src.utils import Attention
from src.deit_vision_transformer import Attention as deit_attention 
from src.deit_vision_transformer import Mlp
from src.swin import ShiftedWindowAttention
from torchvision.ops.misc import MLP as swin_MLP

from .swin_attention_and_mlp import QAttention_swin, QMLP_swin,  QAttention_swin_qkreparam, QAttention_swin_qkreparam_4_cga


QMODULE_MAPPINGS = {
    torch.nn.Linear: QLinear,
    deit_attention: QAttention,
    Mlp: QMLP
}

##  0: QAttention_qkreparam, 1: QAttention_qkreparam_4_cga
QMODULE_MAPPINGS_QK_REPARAM = [
    {
        torch.nn.Linear: QLinear,
        deit_attention: QAttention_qkreparam,
        Mlp: QMLP
    },
    {
        torch.nn.Linear: QLinear,
        deit_attention: QAttention_qkreparam_4_cga,
        Mlp: QMLP
    }
]
QMODULE_MAPPINGS_W_AND_ACT = {
    torch.nn.Linear: LSQ_w_and_act_QLinear,
    deit_attention: QAttention_lsq,
    Mlp: LSQ_w_and_act_QMLP
}
def get_module_by_name(model, module_name):
    names = module_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def set_module_by_name(model, module_name, module):
    if module_name == 'head' or module_name == 'head_dist':
        setattr(model, module_name, module)
    else:
        names = module_name.split(".")
        parent = get_module_by_name(model, ".".join(names[:-1]))
        setattr(parent, names[-1], module)


def replace_module_by_qmodule_deit(model, qconfigs, pretrained_initialized = False,
                        qk_reparam = False, qk_reparam_type = 0, boundaryRange = 0.005): 

        if qconfigs[list(qconfigs.keys())[0]]["weight"]["mode"] == 'lsq' and qconfigs[list(qconfigs.keys())[0]]["act"]["mode"] == 'lsq':

            for name, cfg in qconfigs.items():
                if name == "patch_embed.proj":
                    module = get_module_by_name(model, name)
                    
                    qmodule = LSQ_QConv2d(
                        m = module,
                        weight_bits = 8,
                        input_bits = 8,
                        weight_channelwise = True,
                        input_channelwise = True,
                        weight_quant_method = 'lsq',
                        input_quant_method = 'lsq',
                        aq_learnable = True, ## act 
                        wq_learnable = True,## weight
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)
                elif name == "head" or name == "head_dist":
                    module = get_module_by_name(model, name)
                    qmodule = LSQ_QLinear4head(
                        m = module,
                        weight_bits = 8,
                        input_bits = 8,
                        weight_channelwise = True,
                        input_channelwise = True,
                        weight_quant_method = 'lsq',
                        input_quant_method = 'lsq',
                        aq_learnable = True, ## act 
                        wq_learnable = True,## weight
                        symmetric = True, ## ac
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)
                else:
                    module = get_module_by_name(model, name)
                    
                    qmodule = QMODULE_MAPPINGS_W_AND_ACT[type(module)](
                        m = module,
                        weight_bits = cfg["weight"]['bit'],
                        input_bits = cfg["act"]['bit'],
                        weight_channelwise = cfg["weight"]["per_channel"],
                        input_channelwise = cfg["act"]["per_channel"],
                        weight_quant_method = cfg["weight"]["mode"],
                        input_quant_method = cfg["act"]["mode"],
                        aq_learnable = cfg["act"]["learnable"], ## act 
                        wq_learnable = cfg["weight"]["learnable"],## weight
                        symmetric = not cfg["act"]['all_positive'], ## ac
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)

        elif qk_reparam:
            if qk_reparam_type == 0:
                for name, cfg in qconfigs.items():
                    if name == "patch_embed.proj":
                        module = get_module_by_name(model, name)
                        qmodule = LSQ_QConv2d(
                            m = module,
                            weight_bits = 8,
                            input_bits = 8,
                            weight_channelwise = True,
                            input_channelwise = True,
                            weight_quant_method = 'lsq',
                            input_quant_method = 'lsq',
                            aq_learnable = True, ## act 
                            wq_learnable = True,## weight
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)
                    elif name == "head" or name == "head_dist":
                        module = get_module_by_name(model, name)
                        qmodule = LSQ_QLinear4head(
                            m = module,
                            weight_bits = 8,
                            input_bits = 8,
                            weight_channelwise = True,
                            input_channelwise = True,
                            weight_quant_method = 'lsq',
                            input_quant_method = 'lsq',
                            aq_learnable = True, ## act 
                            wq_learnable = True,## weight
                            symmetric = True, ## ac
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)
                    else:
                        module = get_module_by_name(model, name)
                        qmodule = QMODULE_MAPPINGS_QK_REPARAM[qk_reparam_type][type(module)](
                            m = module,
                            weight_bits = cfg["weight"]['bit'],
                            input_bits = cfg["act"]['bit'],
                            weight_channelwise = cfg["weight"]["per_channel"],
                            input_channelwise = cfg["act"]["per_channel"],
                            weight_quant_method = cfg["weight"]["mode"],
                            input_quant_method = cfg["act"]["mode"],
                            aq_learnable = cfg["act"]["learnable"], ## act 
                            wq_learnable = cfg["weight"]["learnable"],## weight
                            # symmetric = not cfg["act"]['all_positive'], ## ac
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)
            elif qk_reparam_type == 1:
                for name, cfg in qconfigs.items():
                    if name == "patch_embed.proj":
                        module = get_module_by_name(model, name)
                        qmodule = LSQ_QConv2d(
                            m = module,
                            weight_bits = 8,
                            input_bits = 8,
                            weight_channelwise = True,
                            input_channelwise = True,
                            weight_quant_method = 'lsq',
                            input_quant_method = 'lsq',
                            aq_learnable = True, ## act 
                            wq_learnable = True,## weight
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)
                    elif name == "head" or name == "head_dist":
                        module = get_module_by_name(model, name)
                        qmodule = LSQ_QLinear4head(
                            m = module,
                            weight_bits = 8,
                            input_bits = 8,
                            weight_channelwise = True,
                            input_channelwise = True,
                            weight_quant_method = 'lsq',
                            input_quant_method = 'lsq',
                            aq_learnable = True, ## act 
                            wq_learnable = True,## weight
                            symmetric = True, ## ac
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)
                    else:
                        module = get_module_by_name(model, name)
                        qmodule = QMODULE_MAPPINGS_QK_REPARAM[qk_reparam_type][type(module)](
                            m = module,
                            weight_bits = cfg["weight"]['bit'],
                            input_bits = cfg["act"]['bit'],
                            weight_channelwise = cfg["weight"]["per_channel"],
                            input_channelwise = cfg["act"]["per_channel"],
                            weight_quant_method = cfg["weight"]["mode"],
                            input_quant_method = cfg["act"]["mode"],
                            aq_learnable = cfg["act"]["learnable"], ## act 
                            wq_learnable = cfg["weight"]["learnable"],## weight
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized,
                            boundaryRange = boundaryRange
                        )
                        set_module_by_name(model, name, qmodule)

        else: ## statsq w quant
            for name, cfg in qconfigs.items():
                if name == "patch_embed.proj":
                    module = get_module_by_name(model, name)
                    
                    qmodule = LSQ_QConv2d(
                        m = module,
                        weight_bits = 8,
                        input_bits = 8,
                        weight_channelwise = True,
                        input_channelwise = True,
                        weight_quant_method = 'lsq',
                        input_quant_method = 'lsq',
                        aq_learnable = True, ## act 
                        wq_learnable = True,## weight
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)
                elif name == "head" or name == "head_dist":
                    module = get_module_by_name(model, name)
                    qmodule = LSQ_QLinear4head(
                        m = module,
                        weight_bits = 8,
                        input_bits = 8,
                        weight_channelwise = True,
                        input_channelwise = True,
                        weight_quant_method = 'lsq',
                        input_quant_method = 'lsq',
                        aq_learnable = True, ## act 
                        wq_learnable = True,## weight
                        symmetric = True, ## ac
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)
                else:
                    module = get_module_by_name(model, name)
                    
                    qmodule = QMODULE_MAPPINGS[type(module)](
                        m = module,
                        weight_bits = cfg["weight"]['bit'],
                        input_bits = cfg["act"]['bit'],
                        weight_channelwise = cfg["weight"]["per_channel"],
                        input_channelwise = cfg["act"]["per_channel"],
                        weight_quant_method = cfg["weight"]["mode"],
                        input_quant_method = cfg["act"]["mode"],
                        aq_learnable = cfg["act"]["learnable"], ## act 
                        wq_learnable = cfg["weight"]["learnable"],## weight
                        # symmetric = not cfg["act"]['all_positive'], ## ac
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)

        return model



QMODULE_MAPPINGS_SWIN = {
    torch.nn.Linear: QLinear,
    ShiftedWindowAttention: QAttention_swin,
    swin_MLP: QMLP_swin
}

QMODULE_MAPPINGS_QK_REPARAM_SWIN = [
    {
        torch.nn.Linear: QLinear,
        ShiftedWindowAttention: QAttention_swin_qkreparam,
        swin_MLP: QMLP_swin
    },
    {
        torch.nn.Linear: QLinear,
        ShiftedWindowAttention: QAttention_swin_qkreparam_4_cga,
        swin_MLP: QMLP_swin
    }
]

def replace_module_by_qmodule_swin(model, qconfigs, pretrained_initialized = False, qk_reparam = False, qk_reparam_type = 0, boundaryRange = 0.005):  
        if qk_reparam:
                for name, cfg in qconfigs.items():
                    if name == "features.0.0":
                        module = get_module_by_name(model, name)
                        qmodule = LSQ_QConv2d(
                            m = module,
                            weight_bits = 8,
                            input_bits = 8,
                            weight_channelwise = True,
                            input_channelwise = True,
                            weight_quant_method = 'lsq',
                            input_quant_method = 'lsq',
                            aq_learnable = True, ## act 
                            wq_learnable = True,## weight
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)
                    elif name == "head":
                        module = get_module_by_name(model, name)
                        qmodule = LSQ_QLinear4head(
                            m = module,
                            weight_bits = 8,
                            input_bits = 8,
                            weight_channelwise = True,
                            input_channelwise = True,
                            weight_quant_method = 'lsq',
                            input_quant_method = 'lsq',
                            aq_learnable = True, ## act 
                            wq_learnable = True,## weight
                            symmetric = True, ## ac
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)
                    else:
                        module = get_module_by_name(model, name)
                        qmodule = QMODULE_MAPPINGS_QK_REPARAM_SWIN[qk_reparam_type][type(module)](
                            m = module,
                            weight_bits = cfg["weight"]['bit'],
                            input_bits = cfg["act"]['bit'],
                            weight_channelwise = cfg["weight"]["per_channel"],
                            input_channelwise = cfg["act"]["per_channel"],
                            weight_quant_method = cfg["weight"]["mode"],
                            input_quant_method = cfg["act"]["mode"],
                            aq_learnable = cfg["act"]["learnable"], ## act 
                            wq_learnable = cfg["weight"]["learnable"],## weight
                            # symmetric = not cfg["act"]['all_positive'], ## ac
                            act_layer = cfg["act_layer"],
                            pretrained_initialized = pretrained_initialized
                        )
                        set_module_by_name(model, name, qmodule)

        else: ## statsq w quant
            for name, cfg in qconfigs.items():
                if name == "features.0.0":
                    module = get_module_by_name(model, name)
                    qmodule = LSQ_QConv2d(
                        m = module,
                        weight_bits = 8,
                        input_bits = 8,
                        weight_channelwise = True,
                        input_channelwise = True,
                        weight_quant_method = 'lsq',
                        input_quant_method = 'lsq',
                        aq_learnable = True, ## act 
                        wq_learnable = True,## weight
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)
                elif name == "head":
                    module = get_module_by_name(model, name)
                    qmodule = LSQ_QLinear4head(
                        m = module,
                        weight_bits = 8,
                        input_bits = 8,
                        weight_channelwise = True,
                        input_channelwise = True,
                        weight_quant_method = 'lsq',
                        input_quant_method = 'lsq',
                        aq_learnable = True, ## act 
                        wq_learnable = True,## weight
                        symmetric = True, ## ac
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)
                else:
                    module = get_module_by_name(model, name)
                    
                    qmodule = QMODULE_MAPPINGS_SWIN[type(module)](
                        m = module,
                        weight_bits = cfg["weight"]['bit'],
                        input_bits = cfg["act"]['bit'],
                        weight_channelwise = cfg["weight"]["per_channel"],
                        input_channelwise = cfg["act"]["per_channel"],
                        weight_quant_method = cfg["weight"]["mode"],
                        input_quant_method = cfg["act"]["mode"],
                        aq_learnable = cfg["act"]["learnable"], ## act 
                        wq_learnable = cfg["weight"]["learnable"],## weight
                        # symmetric = not cfg["act"]['all_positive'], ## ac
                        act_layer = cfg["act_layer"],
                        pretrained_initialized = pretrained_initialized
                    )
                    set_module_by_name(model, name, qmodule)

        return model
