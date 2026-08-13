"""
Loss functions for Gaussian mixture models.
"""

from __future__ import annotations

from torch import Tensor, logsumexp

from gmml.codebook import _GaussianCodeBook, _WeightedCodeBook
from gmml.validation import Reduction, validate_reduction


def get_log_likelihood(
    inpt: Tensor,
    codebook: _GaussianCodeBook,
    reduction: Reduction = "mean"
) -> Tensor:
    """
    Returns the log-likelihood.

    Args:
        inpt: Input tensor passed to the Gaussian codebook.
        codebook: Gaussian codebook.
        reduction: Reduction mode applied to the log-likelihood values.

    Returns:
        Tensor: Log-likelihood values, optionally reduced.
    """

    validate_reduction(reduction)

    log_joint_prob = codebook.get_log_joint_prob(inpt)
    log_likelihood = logsumexp(log_joint_prob, dim=-1)

    if reduction == "mean":
        return log_likelihood.mean()
    if reduction == "sum":
        return log_likelihood.sum()

    return log_likelihood


def get_cond_entropy(
    inpt: Tensor,
    codebook: _GaussianCodeBook,
    reduction: Reduction = "mean"
) -> Tensor:
    """
    Returns the entropy of the categorical probability distributions.

    Args:
        inpt: Input tensor passed to the Gaussian codebook.
        codebook: Gaussian codebook.
        reduction: Reduction mode applied to the entropy values.

    Returns:
        Tensor: Conditional entropy values, optionally reduced.
    """

    validate_reduction(reduction)

    cond_entropy = codebook.get_cond_entropy(inpt)

    if reduction == "mean":
        return cond_entropy.mean()
    if reduction == "sum":
        return cond_entropy.sum()

    return cond_entropy


def get_importance(codebook: _WeightedCodeBook) -> Tensor:
    """
    Returns the entropy of the component weights.

    Args:
        codebook: Weighted codebook.

    Returns:
        Tensor: Entropy of the component weights.
    """

    norm_logit = codebook.norm_logit
    importance = - (norm_logit * norm_logit.exp()).sum(-1)

    return importance
