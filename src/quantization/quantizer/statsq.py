import torch
import torch.nn as nn
import math
import numpy as np

## create 1D mask
def create_mask(s2, prob):
    raw = torch.zeros((s2,))
    raw[:int((1-prob) * s2)] = 1.0/(1.0-prob)  # set EXACTLY 30% of the pixels in the mask
    ridx = torch.randperm(s2)   # a random permutation of the entries
    return raw[ridx]

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def clip(x, eps):
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()

def modify_grad(x, freeze_inds):
    x = x * freeze_inds * 0 + x * (1-freeze_inds) 
    return x


class TrackOscillation(nn.Module):
    """
    This is a wrapper of the int_forward function of a quantizer.
    It tracks the oscillations in integer domain.
    """

    def __init__(self, momentum=0.01, freeze_threshold=0, use_ema_x_int=True):
        super(TrackOscillation, self).__init__()
        self.momentum = momentum

        self.prev_x_int = None
        self.prev_switch_dir = None

        # Statistics to log
        self.ema_oscillation = None
        self.oscillated_sum = None
        self.total_oscillation = None
        self.iters_since_reset = 0

        # Extra variables for weight freezing
        self.freeze_threshold = freeze_threshold  # This should be at least 2-3x the momentum value.
        self.use_ema_x_int = use_ema_x_int
        self.frozen = None
        self.frozen_x_int = None
        self.ema_x_int = None

    def __call__(self, x_int, skip_tracking=False, *args, **kwargs):
       
        # Apply weight freezing
        if self.frozen is not None:
            x_int = ~self.frozen * x_int + self.frozen * self.frozen_x_int

        if skip_tracking:
            return x_int

        with torch.no_grad():
            # Check if everything is correctly initialized, otherwise do so
            self.check_init(x_int)

            # detect difference in x_int  NB we round to avoid int inaccuracies
            delta_x_int = torch.round(self.prev_x_int - x_int).detach()  # should be {-1, 0, 1}
            switch_dir = torch.sign(delta_x_int)  # This is {-1, 0, 1} as sign(0) is mapped to 0
            # binary mask for switching
            switched = delta_x_int != 0

            oscillated = (self.prev_switch_dir * switch_dir) == -1
            self.ema_oscillation = (
                self.momentum * oscillated + (1 - self.momentum) * self.ema_oscillation
            )

            # Update prev_switch_dir for the switch variables
            self.prev_switch_dir[switched] = switch_dir[switched]
            self.prev_x_int = x_int
            self.oscillated_sum = oscillated.sum()
            self.total_oscillation += oscillated
            self.iters_since_reset += 1

            # Freeze some weights
            if self.freeze_threshold > 0:
                freeze_weights = self.ema_oscillation > self.freeze_threshold
                self.frozen[freeze_weights] = True  # Set them to frozen
                if self.use_ema_x_int:
                    self.frozen_x_int[freeze_weights] = torch.round(self.ema_x_int[freeze_weights])
                    # Update x_int EMA which can be used for freezing
                    self.ema_x_int = self.momentum * x_int + (1 - self.momentum) * self.ema_x_int
                else:
                    self.frozen_x_int[freeze_weights] = x_int[freeze_weights]

        return x_int

    def check_init(self, x_int):
        if self.prev_x_int is None:
            # Init prev switch dir to 0
            self.prev_switch_dir = torch.zeros_like(x_int)
            self.prev_x_int = x_int.detach()  # Not sure if needed, don't think so
            self.ema_oscillation = torch.zeros_like(x_int)
            self.oscillated_sum = 0
            self.total_oscillation = torch.zeros_like(x_int)
        else:
            assert (
                self.prev_x_int.shape == x_int.shape
            ), "Tracking shape does not match current tensor shape."

        # For weight freezing
        if self.frozen is None and self.freeze_threshold > 0:
            self.frozen = torch.zeros_like(x_int, dtype=torch.bool)
            self.frozen_x_int = torch.zeros_like(x_int)
            if self.use_ema_x_int:
                self.ema_x_int = x_int.detach().clone()

class StatsQuantizer(nn.Module):
    def __init__(self, num_bits, clip_learnable):
        super(StatsQuantizer, self).__init__()
        self.num_bits = num_bits
        init_act_clip_val = 2.0

        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=False)

        self.s = None

    
    def forward(self, weight):

        real_weights = weight

        if len(weight.shape) == 2:
            scaling_factor = 2 * torch.mean(abs(real_weights),dim=1,keepdim=True) # dim, 1
        elif len(weight.shape) == 3:
            scaling_factor = 2 * torch.mean(torch.mean(abs(real_weights),dim=-1,keepdim=True),dim=0,keepdim=True) # 1, dim, 1

        scaling_factor = scaling_factor.detach()
        self.s = scaling_factor.squeeze().cpu()
        scaled_weights = real_weights/scaling_factor
        cliped_weights = torch.clamp(scaled_weights, min=(-self.clip_val/2), max=(self.clip_val/2)-1e-6)
        n = float(2 ** (self.num_bits - 1))
        quan_weights_no_grad = scaling_factor * ((torch.round((cliped_weights) * n - 0.5 ) + 0.5) / n)
        quan_weights = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        return quan_weights
    

      
class StatsQuantizer_specific_4_qkreparam_cga(nn.Module):
    def __init__(self, num_bits, clip_learnable, boundaryRange = 0.005):
        super(StatsQuantizer_specific_4_qkreparam_cga, self).__init__()

        self.num_bits = num_bits
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=False)
        self.s = None
        self.boundaryRange = boundaryRange 
    
    def forward(self, weight):

        real_weights = weight

        if len(weight.shape) == 2:
            scaling_factor = 2 * torch.mean(abs(real_weights),dim=1,keepdim=True) # dim, 1
        elif len(weight.shape) == 3:
            scaling_factor = 2 * torch.mean(torch.mean(abs(real_weights),dim=-1,keepdim=True),dim=0,keepdim=True) # 1, dim, 1

        
        scaling_factor = scaling_factor.detach()
        self.s = scaling_factor.squeeze().cpu()
        scaled_weights = real_weights/scaling_factor
        cliped_weights = torch.clamp(scaled_weights, min=(-self.clip_val/2), max=(self.clip_val/2)-1e-6)
        n = float(2 ** (self.num_bits - 1))
        b4_round = (cliped_weights) * n - 0.5 
        
        if self.training:
            not_freeze_idx = torch.zeros_like(real_weights).cuda()
            for i in np.arange(start=-(2**(self.num_bits - 1)),stop=(2**(self.num_bits - 1) - 1)): # 0.5 - boundaryRange < x < 0.5 + boundaryRange
                within_boundary = ((b4_round - i) <= (0.5 + self.boundaryRange)) *  ((b4_round - i) >= (0.5 - self.boundaryRange)) #idx of # 0.5 - boundaryRange < x < 0.5 + boundaryRange
                not_freeze_idx  = not_freeze_idx + within_boundary.float()
            
            freeze_idx = 1.0-not_freeze_idx
            b4_round = b4_round.detach() * freeze_idx + b4_round* (1-freeze_idx) 
        
        quan_weights_no_grad = scaling_factor * ((torch.round( b4_round ) + 0.5) / n)
        quan_weights = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        return quan_weights
    

class StatsQuantizer_4d(nn.Module): # B, num_heads, N, in_features
    def __init__(self, num_bits, clip_learnable):
        super(StatsQuantizer_4d, self).__init__()

        self.num_bits = num_bits
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=False)
        self.s = None
    
    def forward(self, weight):

        real_weights = weight

        scaling_factor = 2 * torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=1,keepdim=True),dim=0,keepdim=True)
        
        scaling_factor = scaling_factor.detach()
        self.s = scaling_factor.squeeze().cpu()
        scaled_weights = real_weights/scaling_factor
        cliped_weights = torch.clamp(scaled_weights, min=(-self.clip_val/2), max=(self.clip_val/2)-1e-6)
        n = float(2 ** (self.num_bits - 1))
        quan_weights_no_grad = scaling_factor * ((torch.round((cliped_weights) * n - 0.5 ) + 0.5) / n)
        quan_weights = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        return quan_weights





