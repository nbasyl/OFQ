from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.qlinear import QLinear, LSQ_w_and_act_QLinear
from timm.loss import SoftTargetCrossEntropy

def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grad = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grad)

class Multi_KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target, T=1.0):
        output = output[0] if isinstance(output, tuple) else output
        target = target[0] if isinstance(target, tuple) else target
        output, target = output / T, target / T
        target_prob = F.softmax(target, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        loss = - torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target, T=1.0):
        output = output[0] if isinstance(output, tuple) else output
        target = target[0] if isinstance(target, tuple) else target
        output, target = output / T, target / T
        target_prob = F.softmax(target, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        loss = - torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class KDLossSoftandHard(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.KLSoft =  KLLossSoft()
        self.Hard = nn.CrossEntropyLoss()
        
    def forward(self, output, hard_target, soft_target):

        if isinstance(output, tuple):
            cls_output = output[0]
            dist_output = output[1]
            soft_loss = self.KLSoft(dist_output,soft_target)
            hard_loss = self.Hard(cls_output, hard_target)
        
        else:
            soft_loss = self.KLSoft(output,soft_target)
            hard_loss = self.Hard(output, hard_target)

        return soft_loss + hard_loss

def dampening_loss(w_fp, w_q, x_min, x_max):
    # L &= (s*w_{int} - w)^2
    # We also need to add clipping for both cases, we can do so by using the forward
      # this is also clipped and our target
    # clamp w in FP32 domain to not change range learning (min(max) is needed for per-channel)
    w_fp_clip = torch.min(torch.max(w_fp, x_min), x_max)
    loss = (w_q - w_fp_clip) ** 2
    
    return loss.sum()

class DampeningLoss(torch.nn.Module):
    def __init__(self, weighting=1.0, weight_quant_method = 'nu2u') -> None:
        super().__init__()
        """
        Calculates the dampening loss for all weights in a given quantized model. It is
        expected that all quantized weights are in a Hijacker module.

        """
        self.weighting = weighting
        self.weight_quant_method = weight_quant_method

    def forward(self, model):
        total_bin_loss = 0
        for name, module in model.named_modules():
            if isinstance(module, QLinear) or isinstance(module, LSQ_w_and_act_QLinear) :
                # print(name,"calculate dampening loss")
                # FP32 weight tensor, potential folded but before quantization
                weight = module.weight
                # The matching weight quantizer (not manager, direct quantizer class)
                if self.weight_quant_method == 'lsq':
                    weight_q = module.lsqw_fn(weight).detach()
                    weight_q_min = (module.lsqw_fn.thd_neg * module.lsqw_fn.s).unsqueeze(dim=-1)
                    weight_q_max = (module.lsqw_fn.thd_pos * module.lsqw_fn.s).unsqueeze(dim=-1)
                    
                elif self.weight_quant_method == 'nu2u':
                    weight_q = module.nu2u_fn(weight).detach()
                    weight_q_min, _ = torch.min(weight_q, 1)
                    weight_q_min = weight_q_min.unsqueeze(dim=-1)
                    weight_q_max, _ = torch.max(weight_q, 1)
                    weight_q_max = weight_q_max.unsqueeze(dim=-1)
                
                total_bin_loss += dampening_loss(weight, weight_q, weight_q_min, weight_q_max)
        return total_bin_loss * self.weighting

class KDLossSoftandHard_dampening(torch.nn.Module):
    def __init__(self, weight_quant_method) -> None:
        super().__init__()
        self.KLSoft =  KLLossSoft()
        self.Hard = nn.CrossEntropyLoss()
        self.dampening_loss =  DampeningLoss(weighting=0, weight_quant_method=weight_quant_method)

    def forward(self, output, hard_target, soft_target, model):

        if isinstance(output, tuple):
            cls_output = output[0]
            dist_output = output[1]
            soft_loss = self.KLSoft(dist_output,soft_target)
            hard_loss = self.Hard(cls_output, hard_target)
        
        else:
            soft_loss = self.KLSoft(output,soft_target)
            hard_loss = self.Hard(output, hard_target)
            
        dampening_loss = self.dampening_loss(model) ## as for now only works for LSQ_QLinea AND LSQ_w_and_act_QLinear
 
        return soft_loss + hard_loss + dampening_loss

class KDLossSoftandSoftTargetCE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.KLSoft =  KLLossSoft()
        self.Hard = SoftTargetCrossEntropy()
        
    def forward(self, output, hard_target, soft_target):

        if isinstance(output, tuple):
            cls_output = output[0]
            dist_output = output[1]
            soft_loss = self.KLSoft(dist_output,soft_target)
            hard_loss = self.Hard(cls_output, hard_target)
        
        else:
            soft_loss = self.KLSoft(output,soft_target)
            hard_loss = self.Hard(output, hard_target)

        return soft_loss + hard_loss

def att_loss_r2b(Q_s, Q_t):
    Q_s_norm = Q_s / torch.norm(Q_s, p=2)
    Q_t_norm = Q_t / torch.norm(Q_t, p=2)
    tmp = Q_s_norm - Q_t_norm
    loss = torch.norm(tmp, p=2)
    return loss

def direction_matching_distillation(student_scores, teacher_scores):
    tmp_loss = 0.
    # new_teacher_scores = [teacher_scores[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)] 
    for student_score, teacher_score in zip(student_scores, teacher_scores):
        student_score = torch.where(student_score <= -1e2, 
                                    torch.zeros_like(student_score).to(student_scores[0].device),
                                    student_score)
        teacher_score = torch.where(teacher_score <= -1e2,
                                    torch.zeros_like(teacher_score).to(student_scores[0].device),
                                    teacher_score)
        tmp_loss += att_loss_r2b(student_score, teacher_score)
    return tmp_loss

class KDLossSoftandHard_qk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.KLSoft =  KLLossSoft()
        self.Hard = nn.CrossEntropyLoss()
        
    def forward(self, student_logit, student_attn_info ,target, teacher_logit, teacher_attn_info):
        
        student_q = []
        teacher_q = []
        student_k = []
        teacher_k = []
        for layer in range(len(student_attn_info)):
            student_q.append(student_attn_info[layer][1])
            teacher_q.append(teacher_attn_info[layer][1])
            student_k.append(student_attn_info[layer][2])
            teacher_k.append(teacher_attn_info[layer][2])

        if isinstance(student_logit, tuple):
            cls_output = student_logit[0]
            dist_output = student_logit[1]
            soft_loss = self.KLSoft(dist_output, teacher_logit)
            hard_loss = self.Hard(cls_output, target)
        
        else:
            soft_loss = self.KLSoft(student_logit,teacher_logit)
            hard_loss = self.Hard(student_logit, target)


        q_loss = direction_matching_distillation(student_q, teacher_q)
        k_loss = direction_matching_distillation(student_k, teacher_k)


        return soft_loss + hard_loss + q_loss + k_loss

class KDLossSoftandHard_qkv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.KLSoft =  KLLossSoft()
        self.Hard = nn.CrossEntropyLoss()
        
    def forward(self, student_logit, student_attn_info ,target, teacher_logit, teacher_attn_info):
        
        student_q = []
        teacher_q = []
        student_k = []
        teacher_k = []
        student_v = []
        teacher_v = []
        for layer in range(len(student_attn_info)):
            student_q.append(student_attn_info[layer][1])
            teacher_q.append(teacher_attn_info[layer][1])
            student_k.append(student_attn_info[layer][2])
            teacher_k.append(teacher_attn_info[layer][2])
            student_v.append(student_attn_info[layer][3])
            teacher_v.append(teacher_attn_info[layer][3])

        if isinstance(student_logit, tuple):
            cls_output = student_logit[0]
            dist_output = student_logit[1]
            soft_loss = self.KLSoft(dist_output, teacher_logit)
            hard_loss = self.Hard(cls_output, target)
        
        else:
            soft_loss = self.KLSoft(student_logit,teacher_logit)
            hard_loss = self.Hard(student_logit, target)


        q_loss = direction_matching_distillation(student_q, teacher_q)
        k_loss = direction_matching_distillation(student_k, teacher_k)
        v_loss = direction_matching_distillation(student_v, teacher_v)

        return soft_loss + hard_loss + q_loss + k_loss + v_loss

class KLTokenMSELoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        kd_type: str = "last",
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kl_loss = KLLossSoft(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.kd_type = kd_type

    def _kl_loss(self, output, target):
        return self.kl_loss(output, target)

    def _mse_loss(self, output, target):
        mse_loss = 0
        if self.kd_type == "last":
            if isinstance(output, torch.Tensor):
                _, N, _ = target.size()
                mse_loss = self.mse_loss(output[:, -N:], target)
            else:
                _, N, _ = target[-1].size()
                mse_loss = self.mse_loss(output[-1][:, -N:], target[-1])
        elif self.kd_type == "all":
            if isinstance(output, torch.Tensor):
                _, N, _ = target.size()
                mse_loss = self.mse_loss(output[:, -N:], target)
            else:
                assert len(output) == len(target)
                for i in range(len(output)):
                    _, N, _ = target[i].size()
                    mse_loss += self.mse_loss(output[i][:, -N:], target[i])
                mse_loss = mse_loss / len(output)
        else:
            raise NotImplementedError
        return mse_loss

    def forward(self, output, target):
        assert len(output) == len(target)
        
        kl_loss = self.kl_loss(output[0], target[0])
        mse_loss = self._mse_loss(output[1], target[1])
        loss = kl_loss + self.alpha * mse_loss
        # print(f"KL loss {kl_loss}, MSE loss {mse_loss}, total loss {loss}")

        return loss

