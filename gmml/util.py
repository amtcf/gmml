"""
Utilities.
"""

from math import log

from torch import Tensor, zeros, eye, device as Device, dtype as DType
from torch.nn.functional import softplus
from torch.distributions.multivariate_normal import MultivariateNormal


def sample_standard_multivariate_normal(
    *shape: int,
    device: str | Device | None = None,
    dtype: DType | None = None
) -> Tensor:
    """
    Returns a batch of vectors sampled from a multivariate normal
    distribution with "shape" batch shape.

    Args:
        *shape: Output shape where the last dimension is the vector size and
            the leading dimensions form the sample batch shape.
        device: Optional device on which to construct the distribution and
            returned samples.
        dtype: Optional dtype used to construct the distribution and returned
            samples.

    Returns:
        Tensor: Samples drawn from a standard multivariate normal
            distribution.
    """

    if not len(shape):
        raise ValueError("Argument (shape) is empty")

    dim = shape[-1]
    shape = shape[: -1]

    if dim <= 0:
        raise ValueError("Last entry of argument (shape) must be positive")

    if any(batch_dim < 0 for batch_dim in shape):
        raise ValueError(
            "Batch dimensions in argument (shape) must be non-negative"
        )

    mean = zeros(dim, device=device, dtype=dtype)
    cov = eye(dim, device=device, dtype=dtype)

    dist = MultivariateNormal(mean, cov)

    return dist.sample(shape)


def set_softplus_diag(inpt: Tensor) -> Tensor:
    """
    Returns argument with a softplused diagonal over the last dimensions.

    Args:
        inpt: Tensor whose trailing diagonal entries will be transformed with
            softplus.

    Returns:
        Tensor: Tensor with the same shape as the input and a softplused
            trailing diagonal.
    """

    if inpt.dim() < 2:
        raise ValueError("Argument (inpt) must have 2 or more dimensions")

    diag = inpt.diagonal(dim1=-2, dim2=-1)
    diag = softplus(diag) / log(2.0)

    return inpt.diagonal_scatter(diag, dim1=-2, dim2=-1)
