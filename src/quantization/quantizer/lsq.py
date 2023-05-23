import torch

import numpy as np


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def clip(x, eps):
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()

class LsqQuantizerWeight(torch.nn.Module):
    def __init__(self, bit, all_positive=False,per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizerWeight, self).__init__()

        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        self.register_parameter('s', None)
        self.initialized_alpha = False

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            # print("init alpha")
            assert len(x.shape) == 2
            if len(x.shape) == 2:
                init_val = 2 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5)
            if self.learnable:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-2], device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-2], device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        else:
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            if self.learnable: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        self.initialized_alpha = True
        
    def forward(self, x):
        if self.per_channel:
            if (not self.initialized_alpha):
                self.init_from(x)
            alpha = torch.unsqueeze(self.s,dim=-1)


        else:
            if (not self.initialized_alpha):
                self.init_from(x)     
            alpha = self.s       
            
        if self.per_channel:
            assert len(x.shape) == 2
            if len(x.shape) == 2:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-1]) ** 0.5)

        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}"
        )

class TrackOscillation(torch.nn.Module):
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
            # print("Init tracking", x_int.shape)
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

class LsqQuantizerWeight_iterative_freezing(torch.nn.Module):
    def __init__(self, bit, all_positive=False,per_channel=True, learnable = True, freeze_momentum = 0.01, freeze_threshold = 0.0,**kwargs):
        # super().__init__(bit, normalize_first)
        super(LsqQuantizerWeight_iterative_freezing, self).__init__()

        if all_positive:
            # assert not symmetric, "Positive quantization cannot be symmetric"
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        self.register_parameter('s', None)
        self.initialized_alpha = False
        
        ### freeze tracker
        self.weight_freeze_tracker = TrackOscillation(momentum= freeze_momentum, freeze_threshold= freeze_threshold,use_ema_x_int= True)

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            # print("init alpha")
            assert len(x.shape) == 2
            if len(x.shape) == 2:
                init_val = 2 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5)
            if self.learnable:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-2], device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-2], device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        else:
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            if self.learnable: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        self.initialized_alpha = True
        
    def forward(self, x):
        # eps = torch.tensor(0.00001).float().to(alpha.device)
        if self.per_channel:
            if (not self.initialized_alpha):
                self.init_from(x)
            alpha = torch.unsqueeze(self.s,dim=-1)

        else:
            if (not self.initialized_alpha):
                self.init_from(x)     
            alpha = self.s       


        if self.per_channel:
            assert len(x.shape) == 2
            if len(x.shape) == 2:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-1]) ** 0.5)

        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        
        ### add iterative freeze here
        if self.training:
            x = self.weight_freeze_tracker(x_int=x,skip_tracking=False)
        else:
            x = self.weight_freeze_tracker(x_int=x,skip_tracking=True)
            
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}"
        )

class LsqQuantizer4img(torch.nn.Module):
    def __init__(self, bit = 8, all_positive=False,per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizer4img, self).__init__()

        self.register_buffer('signed', torch.zeros(1))
        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        self.register_parameter('s', None)
        self.initialized_alpha = False

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if len(x.shape) == 4:
                init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=0) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=0) / (self.thd_pos ** 0.5)
            else:
                print("img mush have shape B,C,H,W")
            
            if self.learnable:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[1], device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[1], device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())

        self.initialized_alpha = True
        
    def forward(self, x):
        if self.per_channel:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
                
            if self.signed ==0:
                if self.bit == 1:
                    self.thd_neg = 0
                    self.thd_pos = 1
                else:
                    self.thd_neg = 0
                    self.thd_pos = 2 ** self.bit - 1
            else:
                if self.bit == 1:
                    self.thd_neg = -1
                    self.thd_pos = 1

                else:
                    self.thd_neg = - 2 ** (self.bit - 1)
                    self.thd_pos = 2 ** (self.bit - 1) - 1
            
            if (not self.initialized_alpha):
                self.init_from(x)
            alpha = self.s.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if self.per_channel:
            if len(x.shape) == 4:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[0]*x.shape[2]* x.shape[3]) ** 0.5)
            else:
                print("img can only be B,C,H,W")
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}, "
            f"signed={self.signed}"
        )

class LsqQuantizer4Conv2d(torch.nn.Module):
    def __init__(self, bit = 8, all_positive=False,per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizer4Conv2d, self).__init__()

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        if self.bit == 1:
            self.thd_neg = -1
            self.thd_pos = 1
        else:
            self.thd_neg = - 2 ** (self.bit - 1)
            self.thd_pos = 2 ** (self.bit - 1) - 1
        self.register_parameter('s', None)
        self.initialized_alpha = False

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if len(x.shape) == 4:
                init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=-1) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=-1) / (self.thd_pos ** 0.5)
            else:
                print("img mush have shape B,C,H,W")
            
            if self.learnable:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[0], device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[0], device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())

        self.initialized_alpha = True
        
    def forward(self, x):
        if self.per_channel:            
            if (not self.initialized_alpha):
                self.init_from(x)
            alpha = self.s.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        if self.per_channel:
            if len(x.shape) == 4:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[1]*x.shape[2]* x.shape[3]) ** 0.5)
            else:
                print("img can only be B,C,H,W")
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            # f"{self.eps}), "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}"
        )

class LsqQuantizer4head_input(torch.nn.Module):
    def __init__(self, bit, all_positive=False,per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizer4head_input, self).__init__()

        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable

        # you need to register the parameter names earlier
        self.register_parameter('s', None)
        self.initialized_alpha = False

    def init_from(self, x, *args, **kwargs):

        init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
        if self.learnable: 
            self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"))
            self.s.data.copy_(init_val.cuda())
        else: 
            self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"),requires_grad=False)
            self.s.data.copy_(init_val.cuda())
        self.initialized_alpha = True
        
    def forward(self, x):

        if (not self.initialized_alpha):
            self.init_from(x)     
        alpha = self.s       
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}"
        )

class LsqQuantizer(torch.nn.Module):
    def __init__(self, bit, all_positive=False,per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizer, self).__init__()

        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        self.register_parameter('s', None)
        self.initialized_alpha = False

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if len(x.shape) == 3:
                init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=0) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1).mean(dim=0) / (self.thd_pos ** 0.5)
            elif len(x.shape) == 2:
                init_val = 2 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5)
            elif len(x.shape) == 4:
                init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=0).mean(dim=0) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1).mean(dim=0).mean(dim=0) / (self.thd_pos ** 0.5)
            if self.learnable:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-2], device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-2], device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        else:
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            if self.learnable: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        self.initialized_alpha = True
        
    def forward(self, x):
        if self.per_channel:
            if (not self.initialized_alpha):
                self.init_from(x)
            alpha = torch.unsqueeze(self.s,dim=-1)

        else:
            if (not self.initialized_alpha):
                self.init_from(x)     
            alpha = self.s       

        if self.per_channel:
            if len(x.shape) == 3:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[0]*x.shape[-1]) ** 0.5)
            elif len(x.shape) == 2:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-1]) ** 0.5)
            elif len(x.shape) == 4:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[0]*x.shape[1]* x.shape[-1]) ** 0.5)

        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}"
        )

class LsqQuantizer_only_headwise(torch.nn.Module):
    def __init__(self, bit, all_positive=False,per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizer_only_headwise, self).__init__()

        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        self.register_parameter('s', None)
        self.initialized_alpha = False

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if len(x.shape) == 4:
                init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=0) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=0) / (self.thd_pos ** 0.5)
            else:
                print("check your shape sir")

            if self.learnable:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[1], device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[1], device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        else:
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            if self.learnable: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        self.initialized_alpha = True
        
    def forward(self, x):
        if self.per_channel:
            if (not self.initialized_alpha):
                self.init_from(x)
            alpha = self.s.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        else:
            if (not self.initialized_alpha):
                self.init_from(x)     
            alpha = self.s       

        if self.per_channel:
            if len(x.shape) == 4:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[0]*x.shape[-2]* x.shape[-1]) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}"
        )

class LsqQuantizer4v(torch.nn.Module):
    def __init__(self, bit, all_positive=False,per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizer4v, self).__init__()

        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        self.register_parameter('s', None)
        self.initialized_alpha = False

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if len(x.shape) == 3:
                init_val = 2 * x.detach().abs().mean(dim=0).mean(dim=0) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=0).mean(dim=0) / (self.thd_pos ** 0.5)
            elif len(x.shape) == 4:
                init_val = 2 * x.detach().abs().mean(dim=0).mean(dim=0).mean(dim=0) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=0).mean(dim=0).mean(dim=0) / (self.thd_pos ** 0.5)
            else:
                print("shape is not rights")
                
            if self.learnable:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-1], device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else:
                self.s = torch.nn.Parameter(torch.zeros(x.shape[-1], device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        else:
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            if self.learnable: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"))
                self.s.data.copy_(init_val.cuda())
            else: 
                self.s = torch.nn.Parameter(torch.zeros(1, device="cuda"),requires_grad=False)
                self.s.data.copy_(init_val.cuda())
        self.initialized_alpha = True
        
    def forward(self, x):
        if self.per_channel:
            if (not self.initialized_alpha):
                self.init_from(x)
            if len(x.shape) == 3:
                alpha = self.s
                alpha = alpha.unsqueeze(0).unsqueeze(1)
            elif len(x.shape) == 4:
                alpha = self.s
                alpha = alpha.unsqueeze(0).unsqueeze(1).unsqueeze(2)


        else:
            if (not self.initialized_alpha):
                self.init_from(x)     
            alpha = self.s       
            
        if self.per_channel:
            if len(x.shape) == 3:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[0]*x.shape[1]) ** 0.5)
            elif len(x.shape) == 4:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[0]*x.shape[1]*x.shape[2]) ** 0.5)

        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)

        x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"all_positive={self.all_positive}, "
            f"s_learnable={self.learnable}, "
            f"per_channel={self.per_channel}"
        )

