"""
Functional module
"""

import math


import torch
import torch.linalg as linalg

from torch import Tensor


import gmml.utils as utils
import gmml.validation as val


def _get_vb_pairwise_diffs(x: Tensor, y: Tensor) -> Tensor:
    """
    Returns the pair-wise difference between two vector batches.
    No input validation.

    Args:
        x: First batch of vectors.
        y: Second batch of vectors.

    Returns:
        Tensor: Pair-wise differences between vectors in x and y.
    """

    *x_batch_shape, _ = list(x.shape)
    *y_batch_shape, _ = list(y.shape)

    for _ in y_batch_shape:
        x = x.unsqueeze(-2)

    for _ in x_batch_shape:
        y = y.unsqueeze(0)

    diffs = x - y

    return diffs


def get_vb_pairwise_diffs(x: Tensor, y: Tensor) -> Tensor:
    """
    Returns the pair-wise difference between two vector batches.

    Args:
        x: First validated batch of vectors.
        y: Second validated batch of vectors.

    Returns:
        Tensor: Pair-wise differences between vectors in x and y.
    """

    val.validate_vector_batch(x)
    val.validate_vector_batch(y)

    return _get_vb_pairwise_diffs(x, y)


def _get_log_dets(full_factors: torch.Tensor) -> torch.Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive-diagonal
    triangular matrices.
    No input validation.

    Args:
        full_factors: Batch of triangular matrices.

    Returns:
        Tensor: Log-determinant for each matrix in the batch.
    """

    return full_factors.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def get_log_dets(full_factors: Tensor) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive-diagonal
    triangular matrices.

    Args:
        full_factors: Validated batch of triangular matrices.

    Returns:
        Tensor: Log-determinant for each matrix in the batch.
    """

    val.validate_full_factors(full_factors)

    return _get_log_dets(full_factors)


def _get_triu_mahalanobis_dists(
    inpt: Tensor,
    means: Tensor,
    full_factors: Tensor
) -> Tensor:
    """
    Returns the Mahalanobis distance.
    Broadcasting rules apply to the "means" and "full_factors" arguments.
    No input validation.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        full_factors: Upper-triangular Cholesky factors of component
            precisions.

    Returns:
        Tensor: Mahalanobis distances.
    """

    diffs = _get_vb_pairwise_diffs(inpt, means)

    return ((full_factors.triu() * diffs.unsqueeze(-2)).sum(-1) ** 2).sum(-1)


def get_triu_mahalanobis_dists(
    inpt: Tensor,
    means: Tensor,
    full_factors: Tensor
) -> Tensor:
    """
    Returns the Mahalanobis distance.
    Broadcasting rules apply to the "means" and "full_factors" arguments.

    Args:
        inpt: Validated input batch of vectors.
        means: Validated mean vectors of Gaussian components.
        full_factors: Validated upper-triangular Cholesky factors.

    Returns:
        Tensor: Mahalanobis distances.
    """

    val.validate_full_gaussian_specification(inpt, means, full_factors)

    return _get_triu_mahalanobis_dists(inpt, means, full_factors)


def _get_tr_triu_triu(
    full_factors_0: Tensor, full_factors_1: Tensor
) -> Tensor:
    """
    Returns the covariance similarity term present in the KL divergence
    between two sets of multivariate Gaussians.
    No input validation.

    Args:
        full_factors_0: Cholesky factors for the first Gaussian set.
        full_factors_1: Cholesky factors for the second Gaussian set.

    Returns:
        Tensor: Trace term used in the KL-divergence expression.
    """

    *full_factors_0_batch_shape, _, _ = list(full_factors_0.shape)
    *full_factors_1_batch_shape, _, _ = list(full_factors_1.shape)

    for _ in full_factors_1_batch_shape:
        full_factors_0 = full_factors_0.unsqueeze(-3)

    for _ in full_factors_0_batch_shape:
        full_factors_1 = full_factors_1.unsqueeze(0)

    sys_solution = linalg.solve_triangular(
        full_factors_0.triu(),
        full_factors_1.triu(),
        upper=True,
        left=False
    )

    return (sys_solution.flatten(start_dim=-2) ** 2).sum(-1)


def get_tr_triu_triu(full_factors_0: Tensor, full_factors_1: Tensor) -> Tensor:
    """
    Returns the covariance similarity term present in the KL divergence
    between two sets of multivariate Gaussians.

    Args:
        full_factors_0: First validated batch of Cholesky factors.
        full_factors_1: Second validated batch of Cholesky factors.

    Returns:
        Tensor: Trace term used in the KL-divergence expression.
    """

    val.validate_square_matrix_batches(full_factors_0, full_factors_1)

    return _get_tr_triu_triu(full_factors_0, full_factors_1)


def _get_kl_divs(
    means_1: Tensor,
    full_factors_1: Tensor,
    means_0: Tensor,
    full_factors_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the multivariate Gaussians parameterized
    by "means_0" and "full_factors_0" to the multivariate Gaussians
    parameterized by "means_1" and "full_factors_1".
    Broadcasting rules between the corresponding means and full_factors
    arguments apply.
    No input validation.

    Args:
        means_1: Mean vectors of the target Gaussian distributions.
        full_factors_1: Cholesky factors of the target Gaussian precisions.
        means_0: Mean vectors of the reference Gaussian distributions.
        full_factors_0: Cholesky factors of the reference Gaussian precisions.

    Returns:
        Tensor: KL-divergence.
    """

    *_, dim = list(means_0.shape)

    log_dets_0 = _get_log_dets(full_factors_0)
    log_dets_1 = _get_log_dets(full_factors_1)

    log_dets_0_batch_shape = list(log_dets_0.shape)
    log_dets_1_batch_shape = list(log_dets_1.shape)

    for _ in log_dets_0_batch_shape:
        log_dets_1 = log_dets_1.unsqueeze(-1)

    for _ in log_dets_1_batch_shape:
        log_dets_0 = log_dets_0.unsqueeze(0)

    log_dets_diffs = log_dets_1 - log_dets_0

    tr_triu_triu = _get_tr_triu_triu(full_factors_1, full_factors_0)

    dists = _get_triu_mahalanobis_dists(means_1, means_0, full_factors_0)

    return tr_triu_triu / 2 + dists / 2 + log_dets_diffs - dim / 2


def get_kl_divs(
    means_1: Tensor,
    full_factors_1: Tensor,
    means_0: Tensor,
    full_factors_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the multivariate Gaussians parameterized
    by "means_0" and "full_factors_0" to the multivariate Gaussians
    parameterized by "means_1" and "full_factors_1".
    Broadcasting rules between the corresponding means and full_factors
    arguments apply.

    Args:
        means_1: Validated mean vectors of the target Gaussian
            distributions.
        full_factors_1: Validated Cholesky factors of the target precisions.
        means_0: Validated mean vectors of the reference Gaussian
            distributions.
        full_factors_0: Validated Cholesky factors of the reference
            precisions.

    Returns:
        Tensor: KL-divergence.
    """

    val.validate_full_gaussian_parametrizations(
        means_1,
        full_factors_1,
        means_0,
        full_factors_0
    )

    return _get_kl_divs(means_1, full_factors_1, means_0, full_factors_0)


def get_log_joint_probs(
    inpt: Tensor,
    means: Tensor,
    full_factors: Tensor,
    norm_logits: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by a general
    Gaussian mixture model.
    Broadcasting rules between the "means" and "full_factors" arguments apply.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        full_factors: Cholesky factors of component precisions.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    val.validate_full_gaussian_specification(
        inpt, means, full_factors, norm_logits
    )

    _, dim = list(means.shape)

    log_dets = _get_log_dets(full_factors)

    dists = _get_triu_mahalanobis_dists(inpt, means, full_factors)

    return norm_logits + log_dets - dists / 2 - dim * math.log(2 * math.pi) / 2


def get_samples(
    means: Tensor,
    full_factors: Tensor,
    *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "means" and "full_factors".
    Applies a multivariate reparameterization trick.

    Args:
        means: Mean vectors of Gaussian components.
        full_factors: Cholesky factors of component precisions.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    val.validate_full_gaussian_parametrization(means, full_factors)

    device = means.device
    dtype = means.dtype

    identity = torch.ones_like(means).diag_embed()
    inv = linalg.solve_triangular(full_factors, identity, upper=True)

    samples = utils.sample_standard_multivariate_normal(
        *sample_batch_shape,
        *means.shape,
        device=device,
        dtype=dtype
    )

    return means + (inv * samples.unsqueeze(-2)).sum(-1)


def _get_modal_entropies(full_factors: Tensor) -> Tensor:
    """
    Returns the entropy of the multivariate Gaussians parameterized in
    "full_factors".
    No input validation.

    Args:
        full_factors: Cholesky factors parameterizing Gaussian precisions.

    Returns:
        Tensor: Entropy value for each Gaussian component.
    """

    *_, dim = list(full_factors.shape)

    return dim * (1 + math.log(2 * math.pi)) / 2 - _get_log_dets(full_factors)


def get_joint_entropy_upper_bound(
    means: Tensor,
    full_factors: Tensor,
    norm_logits: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the Gaussian mixture distribution entropy.

    Args:
        means: Mean vectors of Gaussian components.
        full_factors: Cholesky factors of component precisions.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    val.validate_full_gaussian_parametrization(
        means, full_factors, norm_logits
    )

    weights = norm_logits.exp()

    weighted_modal_entropy = weights @ _get_modal_entropies(full_factors)

    kl_divs = _get_kl_divs(means, full_factors, means, full_factors)
    pairwise = weights @ ((- kl_divs).exp() @ weights).log()

    return weighted_modal_entropy - pairwise


def get_diag_kl_divs(
    means_1: Tensor,
    diag_factors_1: Tensor,
    means_0: Tensor,
    diag_factors_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the diagonally-constrained multivariate
    Gaussians parameterized by "means_0" and "diag_factors_0" to the
    diagonally-constrained multivariate Gaussians parameterized by "means_1"
    and "diag_factors_1".
    Broadcasting rules between the corresponding means and diag_factors
    arguments apply.

    Args:
        means_1: Mean vectors of the target Gaussian distributions.
        diag_factors_1: Diagonal precision factors of the target
            distributions.
        means_0: Mean vectors of the reference Gaussian distributions.
        diag_factors_0: Diagonal precision factors of the reference
            distributions.

    Returns:
        Tensor: KL-divergence.
    """

    val.validate_diag_gaussian_parametrizations(
        means_1,
        diag_factors_1,
        means_0,
        diag_factors_0
    )

    *_, dim = list(means_1.shape)

    diffs = _get_vb_pairwise_diffs(means_1, means_0)
    dists = ((diag_factors_0 * diffs) ** 2).sum(-1)

    *diag_factors_0_batch_shape, _ = list(diag_factors_0.shape)
    *diag_factors_1_batch_shape, _ = list(diag_factors_1.shape)

    for _ in diag_factors_0_batch_shape:
        diag_factors_1 = diag_factors_1.unsqueeze(-2)

    for _ in diag_factors_1_batch_shape:
        diag_factors_0 = diag_factors_0.unsqueeze(0)

    log_dets = (diag_factors_1 / diag_factors_0).log().sum(-1)

    tr_tril_triu = ((diag_factors_1 / diag_factors_0) ** 2).sum(-1)

    return tr_tril_triu / 2 + dists / 2 + log_dets - dim / 2


def get_diag_log_joint_probs(
    inpt: Tensor,
    means: Tensor,
    diag_factors: Tensor,
    norm_logits: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by a
    diagonally-constrained Gaussian mixture model.
    Broadcasting rules between the "means" and "diag_factors" arguments apply.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        diag_factors: Diagonal precision factors of Gaussian components.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    val.validate_diag_gaussian_specification(
        inpt, means, diag_factors, norm_logits
    )

    *_, dim = list(means.shape)

    log_dets = diag_factors.log().sum(-1)

    diffs = _get_vb_pairwise_diffs(inpt, means)
    dists = ((diag_factors * diffs) ** 2).sum(-1)

    return norm_logits + log_dets - dists / 2 - dim * math.log(2 * math.pi) / 2


def get_diag_samples(
    means: Tensor,
    diag_factors: Tensor,
    *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "means" and "diag_factors".
    Applies the reparameterization trick.

    Args:
        means: Mean vectors of Gaussian components.
        diag_factors: Diagonal precision factors of Gaussian components.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    val.validate_diag_gaussian_parametrization(means, diag_factors)

    device = means.device
    dtype = means.dtype

    samples = utils.sample_standard_multivariate_normal(
        *sample_batch_shape,
        *means.shape,
        device=device,
        dtype=dtype
    )

    return means + samples / diag_factors


def get_iso_kl_divs(
    means_1: Tensor,
    iso_factors_1: Tensor,
    means_0: Tensor,
    iso_factors_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the isometrically-constrained multivariate
    Gaussians parameterized by "means_0" and "iso_factors_0" to the
    isometrically-constrained multivariate Gaussians parameterized by
    "means_1" and "iso_factors_1".
    Broadcasting rules between the corresponding means and iso_factors
    arguments apply.

    Args:
        means_1: Mean vectors of the target Gaussian distributions.
        iso_factors_1: Isometric precision factors of the target
            distributions.
        means_0: Mean vectors of the reference Gaussian distributions.
        iso_factors_0: Isometric precision factors of the reference
            distributions.

    Returns:
        Tensor: KL-divergence.
    """

    val.validate_iso_gaussian_parametrizations(
        means_1,
        iso_factors_1,
        means_0,
        iso_factors_0
    )

    *_, dim = list(means_0.shape)

    diffs = _get_vb_pairwise_diffs(means_1, means_0)
    dists = (iso_factors_0 ** 2) * (diffs ** 2).sum(-1)

    iso_factors_0_batch_shape = list(iso_factors_0.shape)
    iso_factors_1_batch_shape = list(iso_factors_1.shape)

    for _ in iso_factors_0_batch_shape:
        iso_factors_1 = iso_factors_1.unsqueeze(-1)

    for _ in iso_factors_1_batch_shape:
        iso_factors_0 = iso_factors_0.unsqueeze(0)

    log_dets = dim * (iso_factors_1.log() - iso_factors_0.log())

    tr_tril_triu = dim * (iso_factors_1 / iso_factors_0) ** 2

    return tr_tril_triu / 2 + dists / 2 + log_dets - dim / 2


def get_iso_log_joint_probs(
    inpt: Tensor,
    means: Tensor,
    iso_factors: Tensor,
    norm_logits: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by an
    isometrically-constrained Gaussian mixture model.
    Broadcasting rules between the "means" and "iso_factors" arguments apply.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        iso_factors: Isometric precision factors of Gaussian components.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    val.validate_iso_gaussian_specification(
        inpt, means, iso_factors, norm_logits
    )

    _, dim = list(means.shape)

    log_dets = dim * iso_factors.log()

    diffs = _get_vb_pairwise_diffs(inpt, means)
    dists = ((iso_factors.unsqueeze(-1) * diffs) ** 2).sum(-1)

    return norm_logits + log_dets - dists / 2 - dim * math.log(2 * math.pi) / 2


def get_iso_samples(
    means: Tensor,
    iso_factors: Tensor,
    *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "means" and "iso_factors".
    Applies the reparameterization trick.

    Args:
        means: Mean vectors of Gaussian components.
        iso_factors: Isometric precision factors of Gaussian components.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    val.validate_iso_gaussian_parametrization(means, iso_factors)

    device = means.device
    dtype = means.dtype

    samples = utils.sample_standard_multivariate_normal(
        *sample_batch_shape,
        *means.shape,
        device=device,
        dtype=dtype
    )

    return means + samples / iso_factors.unsqueeze(-1)
