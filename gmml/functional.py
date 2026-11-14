"""
Differentiable functional forms for Gaussian mixture models.
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


def _get_full_log_dets(full_factors: Tensor) -> Tensor:
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


def get_full_log_dets(full_factors: Tensor) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive-diagonal
    triangular matrices.

    Args:
        full_factors: Batch of triangular matrices.

    Returns:
        Tensor: Log-determinant for each matrix in the batch.
    """

    val.validate_full_factors(full_factors)

    return _get_full_log_dets(full_factors)


def _get_triu_mahalanobis_dists(
    inpt: Tensor, means: Tensor, full_factors: Tensor
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

    # U (x − μ) as a contraction: never materializes a (..., d, d) product.
    sols = torch.einsum("...ij,...j->...i", full_factors, diffs)

    return (sols**2).sum(-1)


def get_triu_mahalanobis_dists(
    inpt: Tensor, means: Tensor, full_factors: Tensor
) -> Tensor:
    """
    Returns the Mahalanobis distance.
    Broadcasting rules apply to the "means" and "full_factors" arguments.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        full_factors: Upper-triangular Cholesky factors.

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

    *full_factors_0_batch_shape, dim, _ = list(full_factors_0.shape)
    *full_factors_1_batch_shape, _, _ = list(full_factors_1.shape)

    full_factors_0 = full_factors_0.reshape(-1, dim, dim)
    full_factors_1 = full_factors_1.reshape(-1, dim, dim)

    # ‖U₁ U₀⁻¹‖²_F = ‖U₀⁻ᵀ U₁ᵀ‖²_F: each U₀ᵀ is solved once against all U₁ᵀ,
    # stacked side by side as a (d, n₁ d) right-hand side, instead of
    # broadcasting (and copying) both factors to every pair.
    rhs = full_factors_1.mT.transpose(0, 1).reshape(dim, -1)

    sys_solution = linalg.solve_triangular(full_factors_0.mT, rhs, upper=False)
    sys_solution = sys_solution.reshape(-1, dim, len(full_factors_1), dim)

    return (
        (sys_solution**2)
        .sum((1, 3))
        .reshape(*full_factors_0_batch_shape, *full_factors_1_batch_shape)
    )


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
    val.validate_full_factors(full_factors_0)
    val.validate_full_factors(full_factors_1)

    return _get_tr_triu_triu(full_factors_0, full_factors_1)


def _get_full_kl_divs(
    means_1: Tensor,
    full_factors_1: Tensor,
    means_0: Tensor,
    full_factors_0: Tensor,
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

    log_dets_0 = _get_full_log_dets(full_factors_0)
    log_dets_1 = _get_full_log_dets(full_factors_1)

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


def get_full_kl_divs(
    means_1: Tensor,
    full_factors_1: Tensor,
    means_0: Tensor,
    full_factors_0: Tensor,
) -> Tensor:
    """
    Returns the KL divergence from the multivariate Gaussians parameterized
    by "means_0" and "full_factors_0" to the multivariate Gaussians
    parameterized by "means_1" and "full_factors_1".
    Broadcasting rules between the corresponding means and full_factors
    arguments apply.

    Args:
        means_1: Mean vectors of the target Gaussian distributions.
        full_factors_1: Cholesky factors of the target precisions.
        means_0: Mean vectors of the reference Gaussian distributions.
        full_factors_0: Cholesky factors of the reference precisions.

    Returns:
        Tensor: KL-divergence.
    """

    val.validate_full_gaussian_parametrizations(
        means_1, full_factors_1, means_0, full_factors_0
    )

    return _get_full_kl_divs(means_1, full_factors_1, means_0, full_factors_0)


def _get_full_log_joint_probs(
    inpt: Tensor, means: Tensor, full_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by a general
    Gaussian mixture model.
    Broadcasting rules between the "means" and "full_factors" arguments apply.
    No input validation.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        full_factors: Cholesky factors of component precisions.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    *_, dim = list(means.shape)

    log_dets = _get_full_log_dets(full_factors)

    dists = _get_triu_mahalanobis_dists(inpt, means, full_factors)

    return norm_logits + log_dets - dists / 2 - dim * math.log(2 * math.pi) / 2


def get_full_log_joint_probs(
    inpt: Tensor, means: Tensor, full_factors: Tensor, norm_logits: Tensor
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

    return _get_full_log_joint_probs(inpt, means, full_factors, norm_logits)


def _get_full_samples(
    means: Tensor, full_factors: Tensor, *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "means" and "full_factors".
    Applies a multivariate reparameterization trick.
    No input validation.

    Args:
        means: Mean vectors of Gaussian components.
        full_factors: Cholesky factors of component precisions.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    *means_batch_shape, dim = list(means.shape)
    *full_factors_batch_shape, _, _ = list(full_factors.shape)

    # Noise follows the broadcast batch shape, so that no two components
    # share the same draw.
    batch_shape = torch.broadcast_shapes(
        means_batch_shape, full_factors_batch_shape
    )

    samples = utils.sample_standard_multivariate_normal(
        *sample_batch_shape,
        *batch_shape,
        dim,
        device=means.device,
        dtype=means.dtype,
    )

    # x = μ + U⁻¹ z, solved against z instead of inverting U.
    sols = linalg.solve_triangular(
        full_factors, samples.unsqueeze(-1), upper=True
    )

    return means + sols.squeeze(-1)


def get_full_samples(
    means: Tensor, full_factors: Tensor, *sample_batch_shape: int
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

    return _get_full_samples(means, full_factors, *sample_batch_shape)


def _get_full_modal_entropies(full_factors: Tensor) -> Tensor:
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

    log_dets = _get_full_log_dets(full_factors)

    return dim * (1 + math.log(2 * math.pi)) / 2 - log_dets


def get_full_modal_entropies(full_factors: Tensor) -> Tensor:
    """
    Returns the entropy of the multivariate Gaussians parameterized in
    "full_factors".

    Args:
        full_factors: Cholesky factors parameterizing Gaussian precisions.

    Returns:
        Tensor: Entropy value for each Gaussian component.
    """

    val.validate_full_factors(full_factors)

    return _get_full_modal_entropies(full_factors)


def _get_joint_entropy_upper_bound(
    modal_entropies: Tensor, kl_divs: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns the pair-wise KL divergence upper-bound on the entropy of a
    Gaussian mixture distribution, Σᵢ πᵢ Hᵢ − Σᵢ πᵢ log Σⱼ πⱼ exp(−KLᵢⱼ).
    No input validation.

    Args:
        modal_entropies: Entropy of each Gaussian component.
        kl_divs: Pair-wise KL divergences between the Gaussian components.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    weights = norm_logits.exp()

    weighted_modal_entropy = (weights * modal_entropies).sum(-1)

    # log Σⱼ πⱼ exp(−KLᵢⱼ) = logsumexpⱼ(log πⱼ − KLᵢⱼ), stable for large KL.
    pairwise = weights @ torch.logsumexp(norm_logits - kl_divs, dim=-1)

    return weighted_modal_entropy - pairwise


def get_joint_entropy_upper_bound(
    modal_entropies: Tensor, kl_divs: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns the pair-wise KL divergence upper-bound on the entropy of a
    Gaussian mixture distribution, Σᵢ πᵢ Hᵢ − Σᵢ πᵢ log Σⱼ πⱼ exp(−KLᵢⱼ).

    Args:
        modal_entropies: Entropy of each Gaussian component, broadcastable
            to the shape of norm_logits.
        kl_divs: Pair-wise KL divergences between the Gaussian components,
            broadcastable to a square matrix of the size of norm_logits.
        norm_logits: Normalized component logits, a vector.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    val.validate_joint_entropy_specification(
        modal_entropies, kl_divs, norm_logits
    )

    return _get_joint_entropy_upper_bound(
        modal_entropies, kl_divs, norm_logits
    )


def _get_full_joint_entropy_upper_bound(
    means: Tensor, full_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the Gaussian mixture distribution entropy.
    No input validation.

    Args:
        means: Mean vectors of Gaussian components.
        full_factors: Cholesky factors of component precisions.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    modal_entropies = _get_full_modal_entropies(full_factors)

    kl_divs = _get_full_kl_divs(means, full_factors, means, full_factors)

    return _get_joint_entropy_upper_bound(
        modal_entropies, kl_divs, norm_logits
    )


def get_full_joint_entropy_upper_bound(
    means: Tensor, full_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the Gaussian mixture distribution entropy.

    Args:
        means: Mean vectors of Gaussian components.
        full_factors: Cholesky factors of component precisions.
        norm_logits: Normalized component logits, a vector.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    val.validate_vector(norm_logits)
    val.validate_full_gaussian_parametrization(
        means, full_factors, norm_logits
    )

    return _get_full_joint_entropy_upper_bound(
        means, full_factors, norm_logits
    )


def _get_diag_log_dets(diag_factors: Tensor) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive diagonal
    matrices, represented by their diagonals.
    No input validation.

    Args:
        diag_factors: Batch of diagonals.

    Returns:
        Tensor: Log-determinant for each diagonal matrix in the batch.
    """

    return diag_factors.log().sum(-1)


def get_diag_log_dets(diag_factors: Tensor) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive diagonal
    matrices, represented by their diagonals.

    Args:
        diag_factors: Batch of diagonals.

    Returns:
        Tensor: Log-determinant for each diagonal matrix in the batch.
    """

    val.validate_diag_factors(diag_factors)

    return _get_diag_log_dets(diag_factors)


def _get_diag_kl_divs(
    means_1: Tensor,
    diag_factors_1: Tensor,
    means_0: Tensor,
    diag_factors_0: Tensor,
) -> Tensor:
    """
    Returns the KL divergence from the diagonally-constrained multivariate
    Gaussians parameterized by "means_0" and "diag_factors_0" to the
    diagonally-constrained multivariate Gaussians parameterized by "means_1"
    and "diag_factors_1".
    Broadcasting rules between the corresponding means and diag_factors
    arguments apply.
    No input validation.

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

    *_, dim = list(means_0.shape)

    diffs = _get_vb_pairwise_diffs(means_1, means_0)
    dists = ((diag_factors_0 * diffs) ** 2).sum(-1)

    *diag_factors_0_batch_shape, _ = list(diag_factors_0.shape)
    *diag_factors_1_batch_shape, _ = list(diag_factors_1.shape)

    log_dets_0 = _get_diag_log_dets(diag_factors_0)
    log_dets_1 = _get_diag_log_dets(diag_factors_1)

    for _ in diag_factors_0_batch_shape:
        log_dets_1 = log_dets_1.unsqueeze(-1)

    log_dets_diffs = log_dets_1 - log_dets_0

    # Σₖ (γ₀ₖ / γ₁ₖ)² for all pairs, as one matrix product.
    tr_triu_triu = (
        (diag_factors_1**-2).reshape(-1, dim)
        @ (diag_factors_0**2).reshape(-1, dim).mT
    ).reshape(*diag_factors_1_batch_shape, *diag_factors_0_batch_shape)

    return tr_triu_triu / 2 + dists / 2 + log_dets_diffs - dim / 2


def get_diag_kl_divs(
    means_1: Tensor,
    diag_factors_1: Tensor,
    means_0: Tensor,
    diag_factors_0: Tensor,
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
        diag_factors_1: Diagonal precision factors of the target distributions.
        means_0: Mean vectors of the reference Gaussian distributions.
        diag_factors_0: Diagonal precision factors of the reference
            distributions.

    Returns:
        Tensor: KL-divergence.
    """

    val.validate_diag_gaussian_parametrizations(
        means_1, diag_factors_1, means_0, diag_factors_0
    )

    return _get_diag_kl_divs(means_1, diag_factors_1, means_0, diag_factors_0)


def _get_diag_log_joint_probs(
    inpt: Tensor, means: Tensor, diag_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by a
    diagonally-constrained Gaussian mixture model.
    Broadcasting rules between the "means" and "diag_factors" arguments apply.
    No input validation.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        diag_factors: Diagonal precision factors of Gaussian components.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    *inpt_batch_shape, dim = list(inpt.shape)

    log_dets = _get_diag_log_dets(diag_factors)

    means, diag_factors = torch.broadcast_tensors(means, diag_factors)
    *batch_shape, _ = list(means.shape)

    # Centering on the mean of the means curbs the cancellation of the
    # expansion below, without changing the distances.
    center = means.detach().reshape(-1, dim).mean(0)
    inpt = (inpt - center).reshape(-1, dim)
    means = (means - center).reshape(-1, dim)
    sq_diag_factors = (diag_factors**2).reshape(-1, dim)

    # ‖γ ⊙ (x − μ)‖² = (x ⊙ x)·γ² − 2 x·(γ² ⊙ μ) + (μ ⊙ μ)·γ², computed as
    # matrix products: no (..., components, d) tensor is materialized.
    dists = (
        (inpt**2) @ sq_diag_factors.mT
        - 2 * inpt @ (sq_diag_factors * means).mT
        + (sq_diag_factors * means**2).sum(-1)
    )
    dists = dists.clamp_min(0).reshape(*inpt_batch_shape, *batch_shape)

    return norm_logits + log_dets - dists / 2 - dim * math.log(2 * math.pi) / 2


def get_diag_log_joint_probs(
    inpt: Tensor, means: Tensor, diag_factors: Tensor, norm_logits: Tensor
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

    return _get_diag_log_joint_probs(inpt, means, diag_factors, norm_logits)


def _get_diag_samples(
    means: Tensor, diag_factors: Tensor, *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "means" and "diag_factors".
    Applies the reparameterization trick.
    No input validation.

    Args:
        means: Mean vectors of Gaussian components.
        diag_factors: Diagonal precision factors of Gaussian components.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    # Noise follows the broadcast batch shape, so that no two components
    # share the same draw.
    shape = torch.broadcast_shapes(means.shape, diag_factors.shape)

    samples = utils.sample_standard_multivariate_normal(
        *sample_batch_shape, *shape, device=means.device, dtype=means.dtype
    )

    return means + samples / diag_factors


def get_diag_samples(
    means: Tensor, diag_factors: Tensor, *sample_batch_shape: int
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

    return _get_diag_samples(means, diag_factors, *sample_batch_shape)


def _get_diag_modal_entropies(diag_factors: Tensor) -> Tensor:
    """
    Returns the entropy of the diagonally-constrained multivariate Gaussians
    parameterized in "diag_factors".
    No input validation.

    Args:
        diag_factors: Diagonal precision factors of Gaussian components.

    Returns:
        Tensor: Entropy value for each Gaussian component.
    """

    *_, dim = list(diag_factors.shape)

    log_dets = _get_diag_log_dets(diag_factors)

    return dim * (1 + math.log(2 * math.pi)) / 2 - log_dets


def get_diag_modal_entropies(diag_factors: Tensor) -> Tensor:
    """
    Returns the entropy of the diagonally-constrained multivariate Gaussians
    parameterized in "diag_factors".

    Args:
        diag_factors: Diagonal precision factors of Gaussian components.

    Returns:
        Tensor: Entropy value for each Gaussian component.
    """

    val.validate_diag_factors(diag_factors)

    return _get_diag_modal_entropies(diag_factors)


def _get_diag_joint_entropy_upper_bound(
    means: Tensor, diag_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the diagonally-constrained Gaussian mixture
    distribution entropy.
    No input validation.

    Args:
        means: Mean vectors of Gaussian components.
        diag_factors: Diagonal precision factors of Gaussian components.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    modal_entropies = _get_diag_modal_entropies(diag_factors)

    kl_divs = _get_diag_kl_divs(means, diag_factors, means, diag_factors)

    return _get_joint_entropy_upper_bound(
        modal_entropies, kl_divs, norm_logits
    )


def get_diag_joint_entropy_upper_bound(
    means: Tensor, diag_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the diagonally-constrained Gaussian mixture
    distribution entropy.

    Args:
        means: Mean vectors of Gaussian components.
        diag_factors: Diagonal precision factors of Gaussian components.
        norm_logits: Normalized component logits, a vector.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    val.validate_vector(norm_logits)
    val.validate_diag_gaussian_parametrization(
        means, diag_factors, norm_logits
    )

    return _get_diag_joint_entropy_upper_bound(
        means, diag_factors, norm_logits
    )


def _get_iso_log_dets(iso_factors: Tensor, dim: int) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive scalar
    matrices, represented by their scale factors.
    No input validation.

    Args:
        iso_factors: Batch of scale factors.
        dim: Dimensionality of the scalar matrices.

    Returns:
        Tensor: Log-determinant for each scalar matrix in the batch.
    """

    return dim * iso_factors.log()


def get_iso_log_dets(iso_factors: Tensor, dim: int) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive scalar
    matrices, represented by their scale factors.

    Args:
        iso_factors: Batch of scale factors.
        dim: Dimensionality of the scalar matrices.

    Returns:
        Tensor: Log-determinant for each scalar matrix in the batch.
    """

    val.validate_iso_factors(iso_factors)
    val.validate_dim(dim)

    return _get_iso_log_dets(iso_factors, dim)


def _get_iso_kl_divs(
    means_1: Tensor,
    iso_factors_1: Tensor,
    means_0: Tensor,
    iso_factors_0: Tensor,
) -> Tensor:
    """
    Returns the KL divergence from the isometrically-constrained multivariate
    Gaussians parameterized by "means_0" and "iso_factors_0" to the
    isometrically-constrained multivariate Gaussians parameterized by
    "means_1" and "iso_factors_1".
    Broadcasting rules between the corresponding means and iso_factors
    arguments apply.
    No input validation.

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

    *_, dim = list(means_0.shape)

    diffs = _get_vb_pairwise_diffs(means_1, means_0)
    dists = (iso_factors_0**2) * (diffs**2).sum(-1)

    iso_factors_0_batch_shape = list(iso_factors_0.shape)
    iso_factors_1_batch_shape = list(iso_factors_1.shape)

    for _ in iso_factors_0_batch_shape:
        iso_factors_1 = iso_factors_1.unsqueeze(-1)

    for _ in iso_factors_1_batch_shape:
        iso_factors_0 = iso_factors_0.unsqueeze(0)

    log_dets_diffs = _get_iso_log_dets(iso_factors_1 / iso_factors_0, dim)

    tr_triu_triu = dim * (iso_factors_0 / iso_factors_1) ** 2

    return tr_triu_triu / 2 + dists / 2 + log_dets_diffs - dim / 2


def get_iso_kl_divs(
    means_1: Tensor,
    iso_factors_1: Tensor,
    means_0: Tensor,
    iso_factors_0: Tensor,
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
        iso_factors_1: Isometric precision factors of the target distributions.
        means_0: Mean vectors of the reference Gaussian distributions.
        iso_factors_0: Isometric precision factors of the reference
            distributions.

    Returns:
        Tensor: KL-divergence.
    """

    val.validate_iso_gaussian_parametrizations(
        means_1, iso_factors_1, means_0, iso_factors_0
    )

    return _get_iso_kl_divs(means_1, iso_factors_1, means_0, iso_factors_0)


def _get_iso_log_joint_probs(
    inpt: Tensor, means: Tensor, iso_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by an
    isometrically-constrained Gaussian mixture model.
    Broadcasting rules between the "means" and "iso_factors" arguments apply.
    No input validation.

    Args:
        inpt: Input batch of vectors.
        means: Mean vectors of Gaussian components.
        iso_factors: Isometric precision factors of Gaussian components.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    *inpt_batch_shape, dim = list(inpt.shape)
    *means_batch_shape, _ = list(means.shape)

    log_dets = _get_iso_log_dets(iso_factors, dim)

    # Centering on the mean of the means curbs the cancellation of the
    # expansion below, without changing the distances.
    center = means.detach().reshape(-1, dim).mean(0)
    inpt = (inpt - center).reshape(-1, dim)
    means = (means - center).reshape(-1, dim)

    # ‖x − μ‖² = ‖x‖² − 2 x·μ + ‖μ‖², computed as a matrix product: no
    # (..., components, d) tensor is materialized.
    sq_dists = (
        (inpt**2).sum(-1, keepdim=True)
        - 2 * inpt @ means.mT
        + (means**2).sum(-1)
    )
    sq_dists = sq_dists.clamp_min(0)
    sq_dists = sq_dists.reshape(*inpt_batch_shape, *means_batch_shape)

    dists = (iso_factors**2) * sq_dists

    return norm_logits + log_dets - dists / 2 - dim * math.log(2 * math.pi) / 2


def get_iso_log_joint_probs(
    inpt: Tensor, means: Tensor, iso_factors: Tensor, norm_logits: Tensor
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

    return _get_iso_log_joint_probs(inpt, means, iso_factors, norm_logits)


def _get_iso_samples(
    means: Tensor, iso_factors: Tensor, *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "means" and "iso_factors".
    Applies the reparameterization trick.
    No input validation.

    Args:
        means: Mean vectors of Gaussian components.
        iso_factors: Isometric precision factors of Gaussian components.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    *means_batch_shape, dim = list(means.shape)

    # Noise follows the broadcast batch shape, so that no two components
    # share the same draw.
    batch_shape = torch.broadcast_shapes(means_batch_shape, iso_factors.shape)

    samples = utils.sample_standard_multivariate_normal(
        *sample_batch_shape,
        *batch_shape,
        dim,
        device=means.device,
        dtype=means.dtype,
    )

    return means + samples / iso_factors.unsqueeze(-1)


def get_iso_samples(
    means: Tensor, iso_factors: Tensor, *sample_batch_shape: int
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

    return _get_iso_samples(means, iso_factors, *sample_batch_shape)


def _get_iso_modal_entropies(iso_factors: Tensor, dim: int) -> Tensor:
    """
    Returns the entropy of the isometrically-constrained multivariate
    Gaussians parameterized in "iso_factors".
    No input validation.

    Args:
        iso_factors: Isometric precision factors of Gaussian components.
        dim: Dimensionality of the Gaussian components.

    Returns:
        Tensor: Entropy value for each Gaussian component.
    """

    log_dets = _get_iso_log_dets(iso_factors, dim)

    return dim * (1 + math.log(2 * math.pi)) / 2 - log_dets


def get_iso_modal_entropies(iso_factors: Tensor, dim: int) -> Tensor:
    """
    Returns the entropy of the isometrically-constrained multivariate
    Gaussians parameterized in "iso_factors".

    Args:
        iso_factors: Isometric precision factors of Gaussian components.
        dim: Dimensionality of the Gaussian components.

    Returns:
        Tensor: Entropy value for each Gaussian component.
    """

    val.validate_iso_factors(iso_factors)
    val.validate_dim(dim)

    return _get_iso_modal_entropies(iso_factors, dim)


def _get_iso_joint_entropy_upper_bound(
    means: Tensor, iso_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the isometrically-constrained Gaussian mixture
    distribution entropy.
    No input validation.

    Args:
        means: Mean vectors of Gaussian components.
        iso_factors: Isometric precision factors of Gaussian components.
        norm_logits: Normalized component logits.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    *_, dim = list(means.shape)

    modal_entropies = _get_iso_modal_entropies(iso_factors, dim)

    kl_divs = _get_iso_kl_divs(means, iso_factors, means, iso_factors)

    return _get_joint_entropy_upper_bound(
        modal_entropies, kl_divs, norm_logits
    )


def get_iso_joint_entropy_upper_bound(
    means: Tensor, iso_factors: Tensor, norm_logits: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the isometrically-constrained Gaussian mixture
    distribution entropy.

    Args:
        means: Mean vectors of Gaussian components.
        iso_factors: Isometric precision factors of Gaussian components.
        norm_logits: Normalized component logits, a vector.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    val.validate_vector(norm_logits)
    val.validate_iso_gaussian_parametrization(means, iso_factors, norm_logits)

    return _get_iso_joint_entropy_upper_bound(means, iso_factors, norm_logits)
