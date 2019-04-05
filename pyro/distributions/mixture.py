from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class MaskedConstraint(constraints.Constraint):
    """
    Combines two constraints interleaved elementwise by a mask.

    :param torch.Tensor mask: boolean mask tensor (of dtype ``torch.uint8``)
    :param torch.constraints.Constraint constraint0: constraint that holds
        wherever ``mask == 0``
    :param torch.constraints.Constraint constraint1: constraint that holds
        wherever ``mask == 1``
    """
    def __init__(self, mask, constraint0, constraint1):
        self.mask = mask
        self.constraint0 = constraint0
        self.constraint1 = constraint1

    def check(self, value):
        result = self.constraint0.check(value)
        mask = self.mask.expand(result.shape) if result.shape != self.mask.shape else self.mask
        result[mask] = self.constraint1.check(value)[mask]
        return result


class MaskedMixture(TorchDistribution):
    """
    A masked deterministic mixture of two distributions.

    This is useful when the mask is sampled from another distribution,
    possibly correlated across the batch. Often the mask can be
    marginalized out via enumeration.

    Example::

        change_point = pyro.sample("change_point",
                                   dist.Categorical(torch.ones(len(data) + 1)),
                                   infer={'enumerate': 'parallel'})
        mask = torch.arange(len(data), dtype=torch.long) >= changepoint
        with pyro.plate("data", len(data)):
            pyro.sample("obs", MaskedMixture(mask, dist1, dist2), obs=data)

    :param torch.Tensor mask: A byte tensor toggling between ``component0``
        and ``component1``.
    :param pyro.distributions.TorchDistribution component0: a distribution
        for batch elements ``mask == 0``.
    :param pyro.distributions.TorchDistribution component1: a distribution
        for batch elements ``mask == 1``.
    """
    arg_constraints = {}  # nothing can be constrained

    def __init__(self, mask, component0, component1, validate_args=None):
        if not torch.is_tensor(mask) or mask.dtype != torch.uint8:
            raise ValueError('Expected mask to be a ByteTensor but got {}'.format(type(mask)))
        if component0.event_shape != component1.event_shape:
            raise ValueError('components event_shape disagree: {} vs {}'
                             .format(component0.event_shape, component1.event_shape))
        batch_shape = broadcast_shape(mask.shape, component0.batch_shape, component1.batch_shape)
        if mask.shape != batch_shape:
            mask = mask.expand(batch_shape)
        if component0.batch_shape != batch_shape:
            component0 = component0.expand(batch_shape)
        if component1.batch_shape != batch_shape:
            component1 = component1.expand(batch_shape)

        self.mask = mask
        self.component0 = component0
        self.component1 = component1
        super(MaskedMixture, self).__init__(batch_shape, component0.event_shape, validate_args)

        # We need to disable _validate_sample on each component since samples are only valid on the
        # component from which they are drawn. Instead we perform validation using a MaskedConstraint.
        self.component0._validate_args = False
        self.component1._validate_args = False

    @property
    def has_rsample(self):
        return self.component0.has_rsample and self.component1.has_rsample

    @constraints.dependent_property
    def support(self):
        if self.component0.support is self.component1.support:
            return self.component0.support
        return MaskedConstraint(self.mask, self.component0.support, self.component1.support)

    def expand(self, batch_shape):
        try:
            return super(MaskedMixture, self).expand(batch_shape)
        except NotImplementedError:
            mask = self.mask.expand(batch_shape)
            component0 = self.component0.expand(batch_shape)
            component1 = self.component1.expand(batch_shape)
            return type(self)(mask, component0, component1)

    def sample(self, sample_shape=torch.Size()):
        mask = self.mask.expand(sample_shape + self.batch_shape) if sample_shape else self.mask
        result = self.component0.sample(sample_shape)
        result[mask] = self.component1.sample(sample_shape)[mask]
        return result

    def rsample(self, sample_shape=torch.Size()):
        mask = self.mask.expand(sample_shape + self.batch_shape) if sample_shape else self.mask
        result = self.component0.rsample(sample_shape)
        result[mask] = self.component1.rsample(sample_shape)[mask]
        return result

    def log_prob(self, value):
        value_shape = broadcast_shape(value.shape, self.batch_shape + self.event_shape)
        if value.shape != value_shape:
            value = value.expand(value_shape)
        if self._validate_args:
            self._validate_sample(value)
        mask_shape = value_shape[:len(value_shape) - len(self.event_shape)]
        mask = self.mask
        if mask.shape != mask_shape:
            mask = mask.expand(mask_shape)
        result = self.component0.log_prob(value)
        result[mask] = self.component1.log_prob(value)[mask]
        return result

    @lazy_property
    def mean(self):
        result = self.component0.mean.clone()
        result[self.mask] = self.component1.mean[self.mask]
        return result

    @lazy_property
    def variance(self):
        result = self.component0.variance.clone()
        result[self.mask] = self.component1.variance[self.mask]
        return result


class ConcatenatedConstraint(constraints.Constraint):
    """
    Combines two constraints by concatenating them.
    Indices index into event shape at dim.
    """
    def __init__(self, constraint0, constraint1, dim, constraint0_inds, constraint1_inds):
        self.constraint0 = constraint0
        self.constraint1 = constraint1
        self.constraint0_inds = constraint0_inds
        self.constraint1_inds = constraint1_inds
        self.dim = dim

    def check(self, value):
        result0 = self.constraint0.check(torch.index_select(value, self.dim, self.constraint0_inds))
        result1 = self.constraint1.check(torch.index_select(value, self.dim, self.constraint1_inds))
        return torch.all(result0) and torch.all(result1)


class ConcatenatedMixture(TorchDistribution):
    """
    A concatenated deterministic mixture of two distributions.
    """
    arg_constraints = {}  # nothing can be constrained

    def __init__(self, component0, component1, dim=0, validate_args=None):
        batch_shape = component0.batch_shape
        if component1.batch_shape != batch_shape:
            component1 = component1.expand(batch_shape)

        # If either component is univariate, convert it to multivariate
        # with event_size = (1,).
        if len(component0.event_shape) == 0:
            raise NotImplementedError
            # Doesn't always work if component0 has batch dimensions already?
            #component0 = component0.expand((1,)).to_event(1)

        if len(component1.event_shape) == 0:
            raise NotImplementedError()
            # Doesn't always work if component0 has batch dimensions already?
            #component1 = component1.expand((1,)).to_event(1)

        if len(component0.event_shape) != len(component1.event_shape):
            raise ValueError("Components must have same # of event dims.")

        for event_dim in range(len(component0.event_shape)):
            if (event_dim != dim and
                    component0.event_shape[event_dim] != component1.event_shape[event_dim]):
                raise ValueError("Components must have matching dimensions in non-concatenated dimensions.")

        event_shape = list(component0.event_shape)
        event_shape[dim] += component1.event_shape[dim]
        event_shape = tuple(event_shape)

        self.component0 = component0
        self.component1 = component1
        if dim < 0:
            self.reverse_dim = dim
        else:
            self.reverse_dim = dim - len(event_shape)
        self.component0_inds = torch.arange(component0.event_shape[dim])
        self.component1_inds = torch.arange(component0.event_shape[dim],
                                            component0.event_shape[dim] +
                                            component1.event_shape[dim])
        super(ConcatenatedMixture, self).__init__(batch_shape, event_shape, validate_args)

        # We need to disable _validate_sample on each component since samples are only valid on the
        # component from which they are drawn. Instead we perform validation using
        # a ConcatenatedConstraint.
        self.component0._validate_args = False
        self.component1._validate_args = False

    @property
    def has_rsample(self):
        return self.component0.has_rsample and self.component1.has_rsample

    @constraints.dependent_property
    def support(self):
        if self.component0.support is self.component1.support:
            return self.component0.support
        return ConcatenatedConstraint(
            self.component0.support, self.component1.support,
            self.reverse_dim, self.component0_inds, self.component1_inds)

    def expand(self, batch_shape):
        try:
            return super(ConcatenatedMixture, self).expand(batch_shape)
        except NotImplementedError:
            component0 = self.component0.expand(batch_shape)
            component1 = self.component1.expand(batch_shape)
            return type(self)(component0, component1)

    def sample(self, sample_shape=torch.Size()):
        result = torch.cat([self.component0.sample(sample_shape),
                            self.component1.sample(sample_shape)], dim=self.reverse_dim)
        return result

    def rsample(self, sample_shape=torch.Size()):
        result = torch.cat([self.component0.rsample(sample_shape),
                            self.component1.rsample(sample_shape)], dim=self.reverse_dim)
        return result

    def log_prob(self, value):
        value_shape = broadcast_shape(value.shape, self.batch_shape + self.event_shape)
        if value.shape != value_shape:
            value = value.expand(value_shape)
        if self._validate_args:
            self._validate_sample(value)
        result0 = self.component0.log_prob(
            torch.index_select(value, self.reverse_dim, self.component0_inds))
        result1 = self.component1.log_prob(
            torch.index_select(value, self.reverse_dim, self.component1_inds))
        return result0 + result1
