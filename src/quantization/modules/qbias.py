import torch
import torch.nn as nn


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        
        return out

class LearnableBias4img(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias4img, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.reshape(x.shape[-1],x.shape[-2]).expand_as(x)
        
        return out
    
