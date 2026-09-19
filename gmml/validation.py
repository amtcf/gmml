"""
Validation helpers and custom errors.
"""

from typing import Literal, Optional


import torch

from torch import Tensor
from torch.testing import assert_close


Shape = list[int] | tuple[int, ...]


class HaveIncompatibleDims(ValueError):
    """Raised when tensor dimensions are incompatible."""
    pass

class HaveNonMatchingShapes(ValueError):
    """Raised when broadcasted shapes do not match."""
    pass

class HaveNonBroadcastableShapes(ValueError):
    """Raised when shapes cannot be broadcast together."""
    pass


class HasNonPositiveElement(ValueError):
    """Raised when a tensor contains a non-positive element."""
    pass

class HasNonPositiveDiagonalElement(ValueError):
    """Raised when a matrix diagonal contains a non-positive element."""
    pass

class IsNotNormalized(ValueError):
    """Raised when a log-probability tensor is not normalized."""
    pass

class IsNotUpperTriangular(ValueError):
    """Raised when a tensor is not upper triangular."""
    pass


class IsEmpty(ValueError):
    """Raised when a tensor is empty."""
    pass

class IsNotVector(ValueError):
    """Raised when a tensor is not one-dimensional."""
    pass

class IsNotScalarBatch(ValueError):
    """Raised when a tensor is not a batch of scalars."""
    pass

class IsNotVectorBatch(ValueError):
    """Raised when a tensor is not a batch of vectors."""
    pass

class IsNotMatrixBatch(ValueError):
    """Raised when a tensor is not a batch of matrices."""
    pass

class IsNotSquareMatrixBatch(ValueError):
    """Raised when a tensor is not a batch of square matrices."""
    pass


def are_broadcastable(*shapes: Shape) -> bool:
    """
    Determines if the shapes are broadcastable.

    Args:
        *shapes: Sequence of shapes to test for broadcast compatibility.

    Returns:
        bool: True when all shapes can be broadcast together, otherwise False.
    """

    try:
        torch.broadcast_shapes(*shapes)
        return True
    except RuntimeError:
        return False


def validate_non_empty(x: Tensor) -> None:
    """
    Validates a non-empty tensor.

    Args:
        x: Tensor expected to contain at least one element.

    Returns:
        None: This function returns only after successful validation.
    """

    if x.numel() == 0:
        raise IsEmpty("Argument tensor (x) is empty")


def validate_scalar_batch(x: Tensor) -> None:
    """
    Validates a batch of scalars.

    Args:
        x: Tensor expected to represent a batch of scalar values.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_non_empty(x)

    if x.dim() < 1:
        raise IsNotScalarBatch("Argument tensor (x) is not a scalar batch")


def validate_vector(x: Tensor) -> None:
    """
    Validates a vector.

    Args:
        x: Tensor expected to be one-dimensional.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_non_empty(x)

    if x.dim() != 1:
        raise IsNotVector("Argument tensor (x) is not a vector")


def validate_vector_batch(x: Tensor) -> None:
    """
    Validates a vector batch.

    Args:
        x: Tensor expected to represent a batch of vectors.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_non_empty(x)

    if x.dim() < 2:
        raise IsNotVectorBatch("Argument tensor (x) is not a vector batch")


def validate_vector_batches(x: Tensor, y: Tensor) -> None:
    """
    Validates a pair of vector batches.

    Args:
        x: First tensor expected to represent a batch of vectors.
        y: Second tensor expected to represent a batch of vectors.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(x)
    validate_vector_batch(y)

    *_, x_last_dim = list(x.shape)
    *_, y_last_dim = list(y.shape)

    if x_last_dim != y_last_dim:
        raise HaveIncompatibleDims(
            "Trailing dim of first tensor (x) and last tensor (y) are "
            "incompatible"
        )


def validate_square_matrix_batch(sq_mtrxs: Tensor) -> None:
    """
    Validates a square matrix.

    Args:
        sq_mtrxs: Tensor expected to represent a batch of square matrices.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_non_empty(sq_mtrxs)

    if sq_mtrxs.dim() < 3:
        raise IsNotMatrixBatch(
            "Argument tensor (sq_mtrxs) is not a batch of matrices"
        )

    *_, sq_mtrxs_slast_dim, sq_mtrxs_last_dim = list(sq_mtrxs.shape)

    if sq_mtrxs_last_dim < 2 or sq_mtrxs_slast_dim < 2:
        raise IsNotMatrixBatch(
            "Argument tensor (sq_mtrxs) is not a batch of matrices"
        )

    if sq_mtrxs_slast_dim != sq_mtrxs_last_dim:
        raise IsNotSquareMatrixBatch(
            "Argument tensor (sq_mtrxs) is not a batch of square matrices"
        )


def validate_square_matrix_batches(
    sq_mtrxs_0: Tensor,
    sq_mtrxs_1: Tensor
) -> None:
    """
    Validates a pair of square matrices.

    Args:
        sq_mtrxs_0: First tensor expected to represent a batch of square
            matrices.
        sq_mtrxs_1: Second tensor expected to represent a batch of square
            matrices.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_square_matrix_batch(sq_mtrxs_0)
    validate_square_matrix_batch(sq_mtrxs_1)

    *_, _, sq_mtrxs_0_last_dim = list(sq_mtrxs_0.shape)
    *_, _, sq_mtrxs_1_last_dim = list(sq_mtrxs_1.shape)

    if sq_mtrxs_0_last_dim != sq_mtrxs_1_last_dim:
        raise HaveIncompatibleDims(
            "Trailing dims of first tensor (sq_mtrxs_0) and last tensor "
            "(sq_mtrxs_1) are incompatible"
        )


def validate_trius(trius: Tensor) -> None:
    """
    Validates a batch of upper triangular tensors.

    Args:
        trius: Tensor expected to be upper triangular over its trailing
            dimensions.

    Returns:
        None: This function returns only after successful validation.
    """

    if trius.tril(diagonal=-1).any():
        raise IsNotUpperTriangular(
            "Argument tensor (trius) is not upper triangular"
        )


def validate_full_factors(full_factors: Tensor) -> None:
    """
    Validates a batch of general Cholesky factors.

    Args:
        full_factors: Tensor expected to contain upper-triangular Cholesky
            factors with positive diagonal entries.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_square_matrix_batch(full_factors)
    validate_trius(full_factors)

    if (full_factors.diagonal(dim1=-2, dim2=-1) <= 0).any():
        raise HasNonPositiveDiagonalElement(
            "Argument tensor (full_factors) has a " \
            "non-positive diagonal element"
        )


def validate_norm_logits(norm_logits: Tensor) -> None:
    """
    Validates a batch of normalized logit values.

    Args:
        norm_logits: Tensor expected to encode normalized log-probabilities.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_scalar_batch(norm_logits)

    try:
        assert_close(
            norm_logits.exp().sum(),
            torch.tensor(1.0),
            check_dtype=False,
            check_device=False
        )
    except AssertionError:
        raise IsNotNormalized(
            "Argument tensor (norm_logits) is not normalized"
        )


def validate_full_gaussian_parametrization(
    means: Tensor,
    full_factors: Tensor,
    norm_logits: Optional[Tensor] = None
) -> None:
    """
    Validates a general Gaussian parametrization.

    Args:
        means: Tensor containing Gaussian mean vectors.
        full_factors: Tensor containing upper-triangular Cholesky factors.
        norm_logits: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(means)
    validate_full_factors(full_factors)

    *means_batch_shape, means_last_dim = list(means.shape)
    *full_factors_batch_shape, _, full_factors_last_dim = list(
        full_factors.shape
    )

    if means_last_dim != full_factors_last_dim:
        raise HaveIncompatibleDims(
            "Trailing dims of first tensor (means) and second tensor "
            "(full_factors) are incompatible"
        )

    if not are_broadcastable(means_batch_shape, full_factors_batch_shape):
        raise HaveNonBroadcastableShapes(
            "Batch dims of first tensor (means) and second tensor "
            "(full_factors) are not broadcastable"
        )

    if norm_logits is not None:
        validate_norm_logits(norm_logits)

        norm_logits_batch_shape = list(norm_logits.shape)
        batch_shape = torch.broadcast_shapes(
            means_batch_shape, full_factors_batch_shape
        )
        batch_shape = list(batch_shape)

        if batch_shape != norm_logits_batch_shape:
            raise HaveNonMatchingShapes(
                "Broadcasting shape of first (means) and second tensor "
                "(full_factors) does not match the shape of third tensor "
                "(norm_logits)"
            )


def validate_full_gaussian_specification(
    inpt: Tensor,
    means: Tensor,
    full_factors: Tensor,
    norm_logits: Optional[Tensor] = None
) -> None:
    """
    Validates a general Gaussian parametrization and its relation with the
    input tensor.

    Args:
        inpt: Input tensor expected to be compatible with the Gaussian means.
        means: Tensor containing Gaussian mean vectors.
        full_factors: Tensor containing upper-triangular Cholesky factors.
        norm_logits: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batches(inpt, means)
    validate_full_gaussian_parametrization(means, full_factors, norm_logits)


def validate_full_gaussian_parametrizations(
    means_0: Tensor,
    full_factors_0: Tensor,
    means_1: Tensor,
    full_factors_1: Tensor,
) -> None:
    """
    Validates a pair of general Gaussian parametrizations.

    Args:
        means_0: Mean vectors for the first Gaussian parametrization.
        full_factors_0: Cholesky factors for the first Gaussian
            parametrization.
        means_1: Mean vectors for the second Gaussian parametrization.
        full_factors_1: Cholesky factors for the second Gaussian
            parametrization.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_full_gaussian_parametrization(means_0, full_factors_0)
    validate_full_gaussian_parametrization(means_1, full_factors_1)

    *_, means_0_last_dim = list(means_0.shape)
    *_, means_1_last_dim = list(means_1.shape)

    if means_0_last_dim != means_1_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (means_0) and third tensor "
            "(means_1) are not compatible"
        )


def validate_diag_factors(diag_factors: Tensor) -> None:
    """
    Validates a diagonally-restricted batch of Cholesky factors.

    Args:
        diag_factors: Tensor expected to contain positive diagonal Cholesky
            factors.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(diag_factors)

    if (diag_factors <= 0).any():
        raise HasNonPositiveElement(
            "Argument tensor (diag_factors) has a non-positive element"
        )


def validate_diag_gaussian_parametrization(
    means: Tensor,
    diag_factors: Tensor,
    norm_logits: Optional[Tensor] = None
) -> None:
    """
    Validates a diagonally-constrained Gaussian parametrization.

    Args:
        means: Tensor containing Gaussian mean vectors.
        diag_factors: Tensor containing diagonal Cholesky factors.
        norm_logits: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(means)
    validate_diag_factors(diag_factors)

    *means_batch_shape, means_last_dim = list(means.shape)
    *diag_factors_batch_shape, diag_factors_last_dim = list(diag_factors.shape)

    if means_last_dim != diag_factors_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (means) and second tensor "
            "(diag_factors) are not compatible"
        )

    if not are_broadcastable(means_batch_shape, diag_factors_batch_shape):
        raise HaveNonBroadcastableShapes(
            "Batch dims of first tensor (means) and second tensor "
            "(diag_factors) are not broadcastable"
        )

    if norm_logits is not None:
        validate_norm_logits(norm_logits)

        norm_logits_batch_shape = list(norm_logits.shape)

        batch_shape = torch.broadcast_shapes(
            means_batch_shape, diag_factors_batch_shape
        )
        batch_shape = list(batch_shape)

        if batch_shape != norm_logits_batch_shape:
            raise HaveNonMatchingShapes(
                "Broadcasting shape of first (means) and second tensor "
                "(diag_factors) does not match the shape of third tensor "
                "(norm_logits)"
            )


def validate_diag_gaussian_specification(
    inpt: Tensor,
    means: Tensor,
    diag_factors: Tensor,
    norm_logits: Optional[Tensor] = None
) -> None:
    """
    Validates a diagonally-constrained Gaussian parametrization and its
    relation with the input tensor.

    Args:
        inpt: Input tensor expected to be compatible with the Gaussian means.
        means: Tensor containing Gaussian mean vectors.
        diag_factors: Tensor containing diagonal Cholesky factors.
        norm_logits: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batches(inpt, means)
    validate_diag_gaussian_parametrization(means, diag_factors, norm_logits)


def validate_diag_gaussian_parametrizations(
    means_0: Tensor,
    diag_factors_0: Tensor,
    means_1: Tensor,
    diag_factors_1: Tensor,
) -> None:
    """
    Validates a pair of diagonally-constrained Gaussian parametrizations.

    Args:
        means_0: Mean vectors for the first Gaussian parametrization.
        diag_factors_0: Diagonal factors for the first Gaussian
            parametrization.
        means_1: Mean vectors for the second Gaussian parametrization.
        diag_factors_1: Diagonal factors for the second Gaussian
            parametrization.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_diag_gaussian_parametrization(means_0, diag_factors_0)
    validate_diag_gaussian_parametrization(means_1, diag_factors_1)

    *_, means_0_last_dim = list(means_0.shape)
    *_, means_1_last_dim = list(means_1.shape)

    if means_0_last_dim != means_1_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (means_0) and third tensor "
            "(means_1) are not compatible"
        )


def validate_iso_factors(iso_factors: Tensor) -> None:
    """
    Validates an isometrically-restricted batch of Cholesky factors.

    Args:
        iso_factors: Tensor expected to contain positive isometric scale
            factors.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_scalar_batch(iso_factors)

    if (iso_factors <= 0).any():
        raise HasNonPositiveElement(
            "Argument tensor (iso_factors) has a non-positive element"
        )


def validate_iso_gaussian_parametrization(
    means: Tensor,
    iso_factors: Tensor,
    norm_logits: Optional[Tensor] = None
) -> None:
    """
    Validates an isometrically-constrained Gaussian parametrization.

    Args:
        means: Tensor containing Gaussian mean vectors.
        iso_factors: Tensor containing isometric scale factors.
        norm_logits: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(means)
    validate_iso_factors(iso_factors)

    *means_batch_shape, _ = list(means.shape)
    iso_factors_batch_shape = list(iso_factors.shape)

    if not are_broadcastable(means_batch_shape, iso_factors_batch_shape):
        raise HaveNonBroadcastableShapes(
            "Batch dims of first tensor (means) and second tensor "
            "(iso_factors) are not broadcastable"
        )

    if norm_logits is not None:
        validate_norm_logits(norm_logits)

        norm_logits_batch_shape = list(norm_logits.shape)

        batch_shape = torch.broadcast_shapes(
            means_batch_shape, iso_factors_batch_shape
        )
        batch_shape = list(batch_shape)

        if batch_shape != norm_logits_batch_shape:
            raise HaveNonMatchingShapes(
                "Broadcasting shape of first (means) and second tensor "
                "(iso_factors) does not match the shape of third tensor "
                "(norm_logits)"
            )


def validate_iso_gaussian_specification(
    inpt: Tensor,
    means: Tensor,
    iso_factors: Tensor,
    norm_logits: Optional[Tensor] = None
) -> None:
    """
    Validates an isometrically-constrained Gaussian parametrization and its
    relation with the input tensor.

    Args:
        inpt: Input tensor expected to be compatible with the Gaussian means.
        means: Tensor containing Gaussian mean vectors.
        iso_factors: Tensor containing isometric scale factors.
        norm_logits: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batches(inpt, means)
    validate_iso_gaussian_parametrization(means, iso_factors, norm_logits)


def validate_iso_gaussian_parametrizations(
    means_0: Tensor,
    iso_factors_0: Tensor,
    means_1: Tensor,
    iso_factors_1: Tensor,
) -> None:
    """
    Validates a pair of isometrically-constrained Gaussian parametrization.

    Args:
        means_0: Mean vectors for the first Gaussian parametrization.
        iso_factors_0: Isometric factors for the first Gaussian
            parametrization.
        means_1: Mean vectors for the second Gaussian parametrization.
        iso_factors_1: Isometric factors for the second Gaussian
            parametrization.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_iso_gaussian_parametrization(means_0, iso_factors_0)
    validate_iso_gaussian_parametrization(means_1, iso_factors_1)

    *_, means_0_last_dim = list(means_0.shape)
    *_, means_1_last_dim = list(means_1.shape)

    if means_0_last_dim != means_1_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (means_0) and third tensor "
            "(means_1) are not compatible"
        )
