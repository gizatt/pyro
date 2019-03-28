from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.iaf import clamp_preserve_gradients

@copy_docs_from(TransformModule)
class MaskedAutoregressiveFlow(TransformModule):
    """
    An implementation of Masked Autoregressive Flow from Papamakarios Et Al., 2018,

    :param autoregressive_nn: an autoregressive neural network whose forward call returns a real-valued
        mean and logit-scale as a tuple
    :type autoregressive_nn: nn.Module
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_max_clip: float

    References:

    1. Masked Autoregressive Flow for Density Estimation [arXiv:1705.07057]
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, autoregressive_nn, log_scale_min_clip=-5., log_scale_max_clip=3.):
        super(MaskedAutoregressiveFlow, self).__init__(cache_size=0)
        self.arn = autoregressive_nn
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        y_size = x.size()[:-1]
        perm = self.arn.permutation
        input_dim = x.size(-1)
        y = [torch.zeros(y_size, device=x.device)] * input_dim

        # NOTE: Forward pass is an expensive operation that scales in the dimension of the input
        for idx in perm:
            mean, log_scale = self.arn(torch.stack(y, dim=-1))
            scale = torch.exp(clamp_preserve_gradients(log_scale[..., idx],
                              min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            y[idx] = scale * x[..., idx] + mean[..., idx]

        y = torch.stack(y, dim=-1)
        log_scale = clamp_preserve_gradients(log_scale, min=self.log_scale_min_clip, max=self.log_scale_max_clip)
        self._cached_log_scale = log_scale
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        mean, log_scale = self.arn(y)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        inverse_scale = torch.exp(-clamp_preserve_gradients(
                log_scale, min=self.log_scale_min_clip,
                max=self.log_scale_max_clip))

        x = (y - mean) * inverse_scale
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        else:
            _, log_scale = self.arn(y)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)
