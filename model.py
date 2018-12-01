from typing import List, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F



class VariationalNet(nn.Module):

    def __init__(self,
                 in_size: Tuple[int, int, int], nclasses: int,
                 channels: List[int],
                 units: List[int],
                 kernels: List[int] = None,
                 strides: List[int] = None,
                 use_bias: bool = False,
                 use_dropout: bool = True,
                 use_pooling: bool = False,
                 use_batch_norm: bool = False) -> None:
        super(VariationalNet, self).__init__()

        if use_batch_norm:
            raise NotImplementedError

        nin, h, w = in_size
        kernels = [5 for c in channels] if kernels is None else kernels
        strides = [1 for c in channels] if strides is None else strides

        self.nclasses = nclasses
        self.nconv = len(channels)
        self.nlinear = len(units)
        self.use_dropout = use_dropout
        self.use_pooling = use_pooling

        self.params = OrderedDict({})

        for idx, (nout, k, s) in enumerate(zip(channels, kernels, strides)):
            self.__add_param(f"c{idx:d}_weight", (nout, nin, k, k))
            if use_bias:
                self.__add_param(f"c{idx:d}_bias", (nout,))

            nin, h, w = nout, (h - k) // s + 1, (w - k) // s + 1
            if use_pooling:
                h, w = h // 2, w // 2

        nin = nin * h * w
        units = units + [nclasses]

        for idx, nout in enumerate(units):
            self.__add_param(f"l{idx:d}_weight", (nout, nin))
            if use_bias:
                self.__add_param(f"l{idx:d}_bias", (nout,))
            nin = nout

    def __add_param(self, name, size):
        mean = nn.Parameter(torch.randn(size))
        logvar = nn.Parameter(torch.randn(size) * 0.001)
        self.params[name] = (mean, logvar)
        setattr(self, name + "_mean", mean)
        setattr(self, name + "_logvar", logvar)

    def __get_params(self, name):
        wname, bname = f"{name:s}_weight", f"{name:s}_bias"
        weight_mean, weight_logvar = self.params[wname]
        weight = self.reparameterize(weight_mean, weight_logvar)
        bias = self.params.get(bname, None)
        if bias is not None:
            bias_mean, bias_logvar = bias
            bias = self.reparameterize(bias_mean, bias_logvar)
        return weight, bias

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        for idx in range(self.nconv):
            weight, bias = self.__get_params(f"c{idx:d}")
            x = F.conv2d(x, weight, bias)
            if self.use_pooling:
                x = F.max_pool2d(x, 2)
            x = F.relu(x)

        x = x.view(x.size(0), -1)

        for idx in range(self.nlinear):
            weight, bias = self.__get_params(f"l{idx:d}")
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
            x = F.linear(x, weight, bias)
        return x

    def average_probs(self, x, nsamples=50):
        avg_output = None
        for _ in range(nsamples):
            output = F.softmax(self.forward(x), dim=1)
            if avg_output is None:
                avg_output = output
            else:
                avg_output += output / nsamples
        return avg_output.max(dim=1)[1]

    def kldiv_layer(self, layer):
        wname, bname = f"{layer:s}_weight", f"{layer:s}_bias"
        weight_mean, weight_logvar = self.params[wname]
        kldiv = torch.sum(weight_mean.pow(2) + weight_logvar.exp() -
                          1 - weight_logvar)
        bias = self.params.get(bname, None)
        if bias is not None:
            bias_mean, bias_logvar = bias
            kldiv += torch.sum(bias_mean.pow(2) + bias_logvar.exp() -
                               1 - bias_logvar)
        return .5 * kldiv

    def kldiv(self):
        kldiv = 0
        for idx in range(self.nconv):
            kldiv += self.kldiv_layer(f"c{idx:d}")
        for idx in range(self.nlinear):
            kldiv += self.kldiv_layer(f"l{idx:d}")
        return kldiv
