"""
Testing utilities module
"""

from typing import Self
from unittest import TestCase

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal, kl_divergence
from torch.testing import assert_close


def log_softmax(inpt: Tensor) -> Tensor:
    """
    Applies log_softmax over the last dimension of a batch of scalars,
    following broadcasting rules: leading dimensions are batch dimensions.

    Args:
        inpt: Input tensor of scalar values.

    Returns:
        Tensor: Log-softmax values, normalized over the last dimension.
    """

    return F.log_softmax(inpt, dim=-1)


def get_random_full_factors(*shape: int) -> Tensor:
    """
    Returns a batch of random upper-triangular Cholesky factors, with
    positive diagonals bounded away from zero.

    Args:
        *shape: Batch shape followed by the dimensionality of the factors.

    Returns:
        Tensor: Random Cholesky factors, of shape (*shape, shape[-1]).
    """

    *batch_shape, dim = shape

    full_factors = torch.randn(*batch_shape, dim, dim).triu()
    diag = full_factors.diagonal(dim1=-2, dim2=-1).abs() + 0.5

    return full_factors.diagonal_scatter(diag, dim1=-2, dim2=-1)


def get_ref_kl_divs(
    means_1: Tensor,
    full_factors_1: Tensor,
    means_0: Tensor,
    full_factors_0: Tensor,
) -> Tensor:
    """
    Returns the reference pair-wise KL divergences KL(N₁ᵢ ‖ N₀ⱼ), computed
    with torch.distributions, for vector batches of means and matrix batches
    of Cholesky factors.

    Args:
        means_1: Mean vectors of the target Gaussian distributions.
        full_factors_1: Cholesky factors of the target Gaussian precisions.
        means_0: Mean vectors of the reference Gaussian distributions.
        full_factors_0: Cholesky factors of the reference Gaussian precisions.

    Returns:
        Tensor: KL-divergences, of shape (len(means_1), len(means_0)).
    """

    dists_1 = [
        MultivariateNormal(mean, precision_matrix=factor.mT @ factor)
        for mean, factor in zip(means_1, full_factors_1, strict=True)
    ]
    dists_0 = [
        MultivariateNormal(mean, precision_matrix=factor.mT @ factor)
        for mean, factor in zip(means_0, full_factors_0, strict=True)
    ]

    return torch.stack(
        [
            torch.stack([kl_divergence(dist_1, dist_0) for dist_0 in dists_0])
            for dist_1 in dists_1
        ]
    )


class TestLogSoftmax(TestCase):
    """
    'log_softmax' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'log_softmax' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor([0.0, 0.0])

        out_0 = log_softmax(inpt_0)

        try:
            assert_close(
                out_0.exp().sum(),
                torch.tensor(1.0),
                check_dtype=False,
                check_device=False,
            )
        except Exception:
            self.assertTrue(False)
