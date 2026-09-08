"""
Functional module
"""

from math import pi, log

from torch import Tensor, ones_like
from torch.linalg import solve_triangular

from gmml.util import sample_standard_multivariate_normal

from gmml.validation import validate_vector_batch, \
    validate_square_matrix_batches, \
    validate_full_fct

from gmml.validation import validate_full_gaussian_specification, \
    validate_full_gaussian_parametrization, \
    validate_full_gaussian_parametrizations

from gmml.validation import validate_diag_gaussian_specification, \
    validate_diag_gaussian_parametrization, \
    validate_diag_gaussian_parametrizations

from gmml.validation import validate_iso_gaussian_specification, \
    validate_iso_gaussian_parametrization, \
    validate_iso_gaussian_parametrizations


def _get_vb_pairwise_diff(x: Tensor, y: Tensor) -> Tensor:
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

    diff = x - y

    return diff


def get_vb_pairwise_diff(x: Tensor, y: Tensor) -> Tensor:
    """
    Returns the pair-wise difference between two vector batches.

    Args:
        x: First validated batch of vectors.
        y: Second validated batch of vectors.

    Returns:
        Tensor: Pair-wise differences between vectors in x and y.
    """

    validate_vector_batch(x)
    validate_vector_batch(y)

    return _get_vb_pairwise_diff(x, y)


def _get_log_det(full_fct: Tensor) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive-diagonal
    triangular matrices.
    No input validation.

    Args:
        full_fct: Batch of triangular matrices.

    Returns:
        Tensor: Log-determinant for each matrix in the batch.
    """

    return full_fct.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def get_log_det(full_fct: Tensor) -> Tensor:
    """
    Returns the logarithm of the determinant of a batch of positive-diagonal
    triangular matrices.

    Args:
        full_fct: Validated batch of triangular matrices.

    Returns:
        Tensor: Log-determinant for each matrix in the batch.
    """

    validate_full_fct(full_fct)

    return _get_log_det(full_fct)


def _get_triu_mahalanobis_dist(
    inpt: Tensor,
    mean: Tensor,
    full_fct: Tensor
) -> Tensor:
    """
    Returns the Mahalanobis distance.
    Broadcasting rules apply to the "mean" and "full_fct" arguments.
    No input validation.

    Args:
        inpt: Input batch of vectors.
        mean: Mean vectors of Gaussian components.
        full_fct: Upper-triangular Cholesky factors of component precisions.

    Returns:
        Tensor: Mahalanobis distances.
    """

    diff = _get_vb_pairwise_diff(inpt, mean)

    return ((full_fct.triu() * diff.unsqueeze(-2)).sum(-1) ** 2).sum(-1)


def get_triu_mahalanobis_dist(
    inpt: Tensor,
    mean: Tensor,
    full_fct: Tensor
) -> Tensor:
    """
    Returns the Mahalanobis distance.
    Broadcasting rules apply to the "mean" and "full_fct" arguments.

    Args:
        inpt: Validated input batch of vectors.
        mean: Validated mean vectors of Gaussian components.
        full_fct: Validated upper-triangular Cholesky factors.

    Returns:
        Tensor: Mahalanobis distances.
    """

    validate_full_gaussian_specification(inpt, mean, full_fct)

    return _get_triu_mahalanobis_dist(inpt, mean, full_fct)


def _get_tr_triu_triu(full_fct_0: Tensor, full_fct_1: Tensor) -> Tensor:
    """
    Returns the covariance similarity term present in the KL divergence
    between two sets of multivariate Gaussians.
    No input validation.

    Args:
        full_fct_0: Cholesky factors for the first Gaussian set.
        full_fct_1: Cholesky factors for the second Gaussian set.

    Returns:
        Tensor: Trace term used in the KL-divergence expression.
    """

    *full_fct_0_batch_shape, _, _ = list(full_fct_0.shape)
    *full_fct_1_batch_shape, _, _ = list(full_fct_1.shape)

    for _ in full_fct_1_batch_shape:
        full_fct_0 = full_fct_0.unsqueeze(-3)

    for _ in full_fct_0_batch_shape:
        full_fct_1 = full_fct_1.unsqueeze(0)

    sys_sol = solve_triangular(
        full_fct_0.triu(),
        full_fct_1.triu(),
        upper=True,
        left=False
    )

    return (sys_sol.flatten(start_dim=-2) ** 2).sum(-1)


def get_tr_triu_triu(full_fct_0: Tensor, full_fct_1: Tensor) -> Tensor:
    """
    Returns the covariance similarity term present in the KL divergence
    between two sets of multivariate Gaussians.

    Args:
        full_fct_0: First validated batch of Cholesky factors.
        full_fct_1: Second validated batch of Cholesky factors.

    Returns:
        Tensor: Trace term used in the KL-divergence expression.
    """

    validate_square_matrix_batches(full_fct_0, full_fct_1)

    return _get_tr_triu_triu(full_fct_0, full_fct_1)


def _get_kl_div(
    mean_1: Tensor,
    full_fct_1: Tensor,
    mean_0: Tensor,
    full_fct_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the multivariate Gaussians parameterized by
    "mean_0" and "full_fct_0" to the multivariate Gaussians parameterized by
    "mean_1" and "full_fct_1".
    Broadcasting rules between the corresponding means and full_fct arguments
    apply.
    No input validation.

    Args:
        mean_1: Mean vectors of the target Gaussian distributions.
        full_fct_1: Cholesky factors of the target Gaussian precisions.
        mean_0: Mean vectors of the reference Gaussian distributions.
        full_fct_0: Cholesky factors of the reference Gaussian precisions.

    Returns:
        Tensor: KL-divergence.
    """

    *_, dim = list(mean_0.shape)

    log_det_0 = _get_log_det(full_fct_0)
    log_det_1 = _get_log_det(full_fct_1)

    log_det_0_batch_shape = list(log_det_0.shape)
    log_det_1_batch_shape = list(log_det_1.shape)

    for _ in log_det_0_batch_shape:
        log_det_1 = log_det_1.unsqueeze(-1)

    for _ in log_det_1_batch_shape:
        log_det_0 = log_det_0.unsqueeze(0)

    log_det = log_det_1 - log_det_0

    tr_triu_triu = _get_tr_triu_triu(full_fct_1, full_fct_0)

    dist = _get_triu_mahalanobis_dist(mean_1, mean_0, full_fct_0)

    return tr_triu_triu / 2 + dist / 2 + log_det - dim / 2


def get_kl_div(
    mean_1: Tensor,
    full_fct_1: Tensor,
    mean_0: Tensor,
    full_fct_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the multivariate Gaussians parameterized by
    "mean_0" and "full_fct_0" to the multivariate Gaussians parameterized by
    "mean_1" and "full_fct_1".
    Broadcasting rules between the corresponding means and full_fct arguments
    apply.

    Args:
        mean_1: Validated mean vectors of the target Gaussian distributions.
        full_fct_1: Validated Cholesky factors of the target precisions.
        mean_0: Validated mean vectors of the reference Gaussian distributions.
        full_fct_0: Validated Cholesky factors of the reference precisions.

    Returns:
        Tensor: KL-divergence.
    """

    validate_full_gaussian_parametrizations(
        mean_1,
        full_fct_1,
        mean_0,
        full_fct_0
    )

    return _get_kl_div(mean_1, full_fct_1, mean_0, full_fct_0)


def get_log_joint_prob(
    inpt: Tensor,
    mean: Tensor,
    full_fct: Tensor,
    norm_logit: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by a general
    Gaussian mixture model.
    Broadcasting rules between the "mean" and "full_fct" arguments apply.

    Args:
        inpt: Input batch of vectors.
        mean: Mean vectors of Gaussian components.
        full_fct: Cholesky factors of component precisions.
        norm_logit: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    validate_full_gaussian_specification(inpt, mean, full_fct, norm_logit)

    _, dim = list(mean.shape)

    log_det = _get_log_det(full_fct)

    dist = _get_triu_mahalanobis_dist(inpt, mean, full_fct)

    return norm_logit + log_det - dist / 2 - dim * log(2 * pi) / 2


def get_sample(
    mean: Tensor,
    full_fct: Tensor,
    *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "mean" and "full_fct".
    Applies a multivariate reparameterization trick.

    Args:
        mean: Mean vectors of Gaussian components.
        full_fct: Cholesky factors of component precisions.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    validate_full_gaussian_parametrization(mean, full_fct)

    device = mean.device
    dtype = mean.dtype

    identity = ones_like(mean).diag_embed()
    inv = solve_triangular(full_fct, identity, upper=True)

    sample = sample_standard_multivariate_normal(
        *sample_batch_shape,
        *mean.shape,
        device=device,
        dtype=dtype
    )

    return mean + (inv * sample.unsqueeze(-2)).sum(-1)


def _get_modal_entropy(full_fct: Tensor) -> Tensor:
    """
    Returns the entropy of the multivariate Gaussians parameterized in
    "full_fct".
    No input validation.

    Args:
        full_fct: Cholesky factors parameterizing Gaussian precisions.

    Returns:
        Tensor: Entropy value for each Gaussian component.
    """

    *_, dim = list(full_fct.shape)

    return dim * (1 + log(2 * pi)) / 2 - _get_log_det(full_fct)


def get_joint_entropy_upper_bound(
    mean: Tensor,
    full_fct: Tensor,
    norm_logit: Tensor
) -> Tensor:
    """
    Returns an upper-bound on the Gaussian mixture distribution entropy.

    Args:
        mean: Mean vectors of Gaussian components.
        full_fct: Cholesky factors of component precisions.
        norm_logit: Normalized component logits.

    Returns:
        Tensor: Upper bound on mixture entropy.
    """

    validate_full_gaussian_parametrization(mean, full_fct, norm_logit)

    weight = norm_logit.exp()

    modal = weight @ _get_modal_entropy(full_fct)

    kl_div = _get_kl_div(mean, full_fct, mean, full_fct)
    pairwise = weight @ ((- kl_div).exp() @ weight).log()

    return modal - pairwise


def get_diag_kl_div(
    mean_1: Tensor,
    diag_fct_1: Tensor,
    mean_0: Tensor,
    diag_fct_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the diagonally-constrained multivariate
    Gaussians parameterized by "mean_0" and "diag_fct_0" to the
    diagonally-constrained multivariate Gaussians parameterized by "mean_1"
    and "diag_fct_1".
    Broadcasting rules between the corresponding means and diag_fct arguments
    apply.

    Args:
        mean_1: Mean vectors of the target Gaussian distributions.
        diag_fct_1: Diagonal precision factors of the target distributions.
        mean_0: Mean vectors of the reference Gaussian distributions.
        diag_fct_0: Diagonal precision factors of the reference distributions.

    Returns:
        Tensor: KL-divergence.
    """

    validate_diag_gaussian_parametrizations(
        mean_1,
        diag_fct_1,
        mean_0,
        diag_fct_0
    )

    *_, dim = list(mean_1.shape)

    diff = _get_vb_pairwise_diff(mean_1, mean_0)
    dist = ((diag_fct_0 * diff) ** 2).sum(-1)

    *diag_fct_0_batch_shape, _ = list(diag_fct_0.shape)
    *diag_fct_1_batch_shape, _ = list(diag_fct_1.shape)

    for _ in diag_fct_0_batch_shape:
        diag_fct_1 = diag_fct_1.unsqueeze(-2)

    for _ in diag_fct_1_batch_shape:
        diag_fct_0 = diag_fct_0.unsqueeze(0)

    log_det = (diag_fct_1 / diag_fct_0).log().sum(-1)

    tr_tril_triu = ((diag_fct_1 / diag_fct_0) ** 2).sum(-1)

    return tr_tril_triu / 2 + dist / 2 + log_det - dim / 2


def get_diag_log_joint_prob(
    inpt: Tensor,
    mean: Tensor,
    diag_fct: Tensor,
    norm_logit: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by a
    diagonally-constrained Gaussian mixture model.
    Broadcasting rules between the "mean" and "diag_fct" arguments apply.

    Args:
        inpt: Input batch of vectors.
        mean: Mean vectors of Gaussian components.
        diag_fct: Diagonal precision factors of Gaussian components.
        norm_logit: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    validate_diag_gaussian_specification(inpt, mean, diag_fct, norm_logit)

    *_, dim = list(mean.shape)

    log_det = diag_fct.log().sum(-1)

    diff = _get_vb_pairwise_diff(inpt, mean)
    dist = ((diag_fct * diff) ** 2).sum(-1)

    return norm_logit + log_det - dist / 2 - dim * log(2 * pi) / 2


def get_diag_sample(
    mean: Tensor,
    diag_fct: Tensor,
    *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "mean" and "diag_fct".
    Applies the reparameterization trick.

    Args:
        mean: Mean vectors of Gaussian components.
        diag_fct: Diagonal precision factors of Gaussian components.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    validate_diag_gaussian_parametrization(mean, diag_fct)

    device = mean.device
    dtype = mean.dtype

    sample = sample_standard_multivariate_normal(
        *sample_batch_shape,
        *mean.shape,
        device=device,
        dtype=dtype
    )

    return mean + sample / diag_fct


def get_iso_kl_div(
    mean_1: Tensor,
    iso_fct_1: Tensor,
    mean_0: Tensor,
    iso_fct_0: Tensor
) -> Tensor:
    """
    Returns the KL divergence from the isometrically-constrained multivariate
    Gaussians parameterized by "mean_0" and "iso_fct_0" to the
    isometrically-constrained multivariate Gaussians parameterized by "mean_1"
    and "iso_fct_1".
    Broadcasting rules between the corresponding means and iso_fct arguments
    apply.

    Args:
        mean_1: Mean vectors of the target Gaussian distributions.
        iso_fct_1: Isometric precision factors of the target distributions.
        mean_0: Mean vectors of the reference Gaussian distributions.
        iso_fct_0: Isometric precision factors of the reference distributions.

    Returns:
        Tensor: KL-divergence.
    """

    validate_iso_gaussian_parametrizations(
        mean_1,
        iso_fct_1,
        mean_0,
        iso_fct_0
    )

    *_, dim = list(mean_0.shape)

    diff = _get_vb_pairwise_diff(mean_1, mean_0)
    dist = (iso_fct_0 ** 2) * (diff ** 2).sum(-1)

    iso_fct_0_batch_shape = list(iso_fct_0.shape)
    iso_fct_1_batch_shape = list(iso_fct_1.shape)

    for _ in iso_fct_0_batch_shape:
        iso_fct_1 = iso_fct_1.unsqueeze(-1)

    for _ in iso_fct_1_batch_shape:
        iso_fct_0 = iso_fct_0.unsqueeze(0)

    log_det = dim * (iso_fct_1.log() - iso_fct_0.log())

    tr_tril_triu = dim * (iso_fct_1 / iso_fct_0) ** 2

    return tr_tril_triu / 2 + dist / 2 + log_det - dim / 2


def get_iso_log_joint_prob(
    inpt: Tensor,
    mean: Tensor,
    iso_fct: Tensor,
    norm_logit: Tensor
) -> Tensor:
    """
    Returns the logarithm of the joint probability specified by a
    isometrically-constrained Gaussian mixture model.
    Broadcasting rules between the "mean" and "iso_fct" arguments apply.

    Args:
        inpt: Input batch of vectors.
        mean: Mean vectors of Gaussian components.
        iso_fct: Isometric precision factors of Gaussian components.
        norm_logit: Normalized component logits.

    Returns:
        Tensor: Log joint probabilities per input and component.
    """

    validate_iso_gaussian_specification(inpt, mean, iso_fct, norm_logit)

    _, dim = list(mean.shape)

    log_det = dim * iso_fct.log()

    diff = _get_vb_pairwise_diff(inpt, mean)
    dist = ((iso_fct.unsqueeze(-1) * diff) ** 2).sum(-1)

    return norm_logit + log_det - dist / 2 - dim * log(2 * pi) / 2


def get_iso_sample(
    mean: Tensor,
    iso_fct: Tensor,
    *sample_batch_shape: int
) -> Tensor:
    """
    Returns a batch of samples withdrawn from each of the multivariate
    Gaussians parameterized by "mean" and "iso_fct".
    Applies the reparameterization trick.

    Args:
        mean: Mean vectors of Gaussian components.
        iso_fct: Isometric precision factors of Gaussian components.
        *sample_batch_shape: Leading shape of independent samples to draw.

    Returns:
        Tensor: Samples drawn from each Gaussian component.
    """

    validate_iso_gaussian_parametrization(mean, iso_fct)

    device = mean.device
    dtype = mean.dtype

    sample = sample_standard_multivariate_normal(
        *sample_batch_shape,
        *mean.shape,
        device=device,
        dtype=dtype
    )

    return mean + sample / iso_fct.unsqueeze(-1)
