import torch
import torch.nn as nn
import tensorly as tl
from torch.nn import Parameter, ParameterList
import numpy as np
import pdb

tl.set_backend('pytorch')


def mode2_dot(tensor, matrix, mode):
    ms = matrix.shape
    matrix = matrix.reshape(ms[0]*ms[1], ms[2]*ms[3])
    
    sp = list(tensor.shape)
    sp[mode:mode+2] = [sp[mode]*sp[mode+1], 1]
    
    sn = list(tensor.shape)
    sn[mode:mode+2] = ms[2:4]
    
    tensor = tensor.reshape(sp)
    tensor = tl.tenalg.mode_dot(tensor, matrix.t(), mode)
    return tensor.reshape(sn)

class TTlinear(nn.Module):
    def __init__(self, in_size, out_size, rank,bias = True, **kwargs):
        # increase beta to decrease rank
        #  in_size[0]    in_size[1]                    in_size[k]
        #     |             |                               |
        #     *---rank[0]---*---rank[1]---...---rank[k-1]---*
        #     |             |                               |
        # out_size[0]   out_size[1]                   out_size[k]
        #
        super(TTlinear, self).__init__()
        assert(len(in_size) == len(out_size))
        assert(len(rank) == len(in_size) - 1)
        self.in_size = list(in_size)
        self.out_size = list(out_size)
        self.rank = list(rank)
        self.factors = ParameterList()
        r1 =[1] + list(rank)
        r2 = list(rank) + [1]
        for ri, ro, si, so in zip(r1, r2, in_size, out_size):
            p = Parameter(torch.Tensor(ri, si, so, ro))
            self.factors.append(p)
        if bias:
            self.bias = Parameter(torch.Tensor(np.prod(out_size)))
        else:
            self.register_parameter('bias', None)
        
        self._initialize_weights()
        
    def forward(self, x):
        
        x = x.reshape((x.shape[0], 1, *self.in_size))
        for (i, f) in enumerate(self.factors):
            x = mode2_dot(x, f, i+1)
        x = x.reshape((x.shape[0], -1))
        x = x + self.bias
        return x
    
    def _initialize_weights(self,weight=None):
        if weight == None:
            for f in self.factors:
                nn.init.kaiming_uniform_(f) 
        else:
            self.factors = weight
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
  
    
class TTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, rank, kernel_size=3, 
                 stride=1, padding=1, dilation=1,
                  **kwargs):
    # increase beta to decrease rank
        #  in_channels[0  in_channels[1]               in_channels[k]
        #     |             |                               |
        #     *---rank[0]---*---rank[1]---...---rank[k-1]---*-----kernel_size
        #     |             |                               |
        # out_channels[0] out_channels[1]             out_channels[k]
        #
        # increase beta to decrease rank
        super(TTConv2d, self).__init__()
        assert(len(in_channels) == len(in_channels))
        assert(len(rank) == len(in_channels) - 1)
        self.in_channels = list(in_channels)
        self.out_channels = list(out_channels)
        self.rank = list(rank)
        self.factors = ParameterList()
        
        r1 = [1] + self.rank[:-1]
        r2 = self.rank
        for ri, ro, si, so in zip(r1, r2, in_channels[:-1], out_channels[:-1]):
            p = Parameter(torch.Tensor(ri, si, so, ro))
            self.factors.append(p)
        self.bias = Parameter(torch.Tensor(np.prod(out_channels)))
        
        self.conv = nn.Conv2d(
                in_channels=self.rank[-1] * in_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False)
        self._initialize_weights()
        
    def forward(self, x):
            
        (b, c, h, w) = x.shape
        x = x.reshape((x.shape[0], 1, *self.in_channels, h, w))
        
        for (i, f) in enumerate(self.factors):
            x = mode2_dot(x, f, i+1)
        x = x.reshape((b * np.prod(self.out_channels[:-1]), 
                       self.rank[-1] * self.in_channels[-1], h, w))
        
        pdb.set_trace()
        x = self.conv(x)
        
        x = x.reshape((b, np.prod(self.out_channels), h, w))
        x = x + self.bias.reshape((1, -1, 1, 1))
        return x
    
    def _initialize_weights(self):
        for f in self.factors:
            nn.init.kaiming_uniform_(f)
        nn.init.constant_(self.bias, 0)
        
    