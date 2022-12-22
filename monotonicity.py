"""
This file contains custom PyTorch modules for imposing monotonicity constraints on neural networks.
"""

import torch.nn.functional as F

from diffusers import ModelMixin
from torch.nn import Linear, Sigmoid, init


class MonotonicLinear(Linear):
    """
    A simple monotonic dense layer.
    """

    def forward(self, x):
        return F.linear(x, self.weight.abs(), self.bias)


class MonotonicNet(ModelMixin):
    """
    A numerically stable monotonic net.
    """

    def __init__(self, min, max, n_features=1024):
        super().__init__()
        self.n_features = n_features
        self.mon_lin1 = MonotonicLinear(1, 1)
        self.mon_lin2 = MonotonicLinear(1, n_features)
        self.mon_lin3 = MonotonicLinear(n_features, 1, bias=False)
        self.expit = Sigmoid()

        init.constant_(self.mon_lin1.weight, max - min)
        init.constant_(self.mon_lin1.bias, min)
        init.normal_(self.mon_lin2.weight)
        init.normal_(self.mon_lin2.bias)
        init.normal_(self.mon_lin3.weight)

    def forward(self, t):
        t = t.unsqueeze(dim=1).type(self.dtype)
        h = self.mon_lin1(t)
        _h = 2. * (t - .5)  # scale input to [-1, +1]
        _h = self.mon_lin2(_h)
        _h = 2 * (self.expit(_h) - .5)  # scale input to [-1, +1]
        _h = self.mon_lin3(_h)
        _h /= self.n_features  # scale input to [-1, +1]
        h += _h
        return h.squeeze()
