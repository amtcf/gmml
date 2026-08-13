"""
Validation helpers and custom errors.
"""

from typing import Literal, Optional

from torch import Tensor, tensor, broadcast_shapes
from torch.testing import assert_close


Reduction = Literal["mean", "sum"] | None
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
        broadcast_shapes(*shapes)
        return True
    except RuntimeError:
        return False


def validate_reduction(reduction: Reduction) -> None:
    """
    Validates a loss reduction parameter.

    Args:
        reduction: Reduction mode to validate.

    Returns:
        None: This function returns only after successful validation.
    """

    if reduction is not None and reduction not in {"mean", "sum"}:
        raise ValueError(
            'Argument (reduction) can only be "mean", "sum" or None'
        )


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


def validate_square_matrix_batch(sq_mtrx: Tensor) -> None:
    """
    Validates a square matrix.

    Args:
        sq_mtrx: Tensor expected to represent a batch of square matrices.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_non_empty(sq_mtrx)

    if sq_mtrx.dim() < 3:
        raise IsNotMatrixBatch(
            "Argument tensor (sq_mtrx) is not a batch of matrices"
        )

    *_, sq_mtrx_slast_dim, sq_mtrx_last_dim = list(sq_mtrx.shape)

    if sq_mtrx_last_dim < 2 or sq_mtrx_slast_dim < 2:
        raise IsNotMatrixBatch(
            "Argument tensor (sq_mtrx) is not a batch of matrices"
        )

    if sq_mtrx_slast_dim != sq_mtrx_last_dim:
        raise IsNotSquareMatrixBatch(
            "Argument tensor (sq_mtrx) is not a batch of square matrices"
        )


def validate_square_matrix_batches(
    sq_mtrx_0: Tensor,
    sq_mtrx_1: Tensor
) -> None:
    """
    Validates a pair of square matrices.

    Args:
        sq_mtrx_0: First tensor expected to represent a batch of square
            matrices.
        sq_mtrx_1: Second tensor expected to represent a batch of square
            matrices.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_square_matrix_batch(sq_mtrx_0)
    validate_square_matrix_batch(sq_mtrx_1)

    *_, _, sq_mtrx_0_last_dim = list(sq_mtrx_0.shape)
    *_, _, sq_mtrx_1_last_dim = list(sq_mtrx_1.shape)

    if sq_mtrx_0_last_dim != sq_mtrx_1_last_dim:
        raise HaveIncompatibleDims(
            "Trailing dims of first tensor (sq_mtrx_0) and last tensor "
            "(sq_mtrx_1) are incompatible"
        )


def validate_triu(triu: Tensor) -> None:
    """
    Validates an upper triangular tensor.

    Args:
        triu: Tensor expected to be upper triangular over its trailing
            dimensions.

    Returns:
        None: This function returns only after successful validation.
    """

    if triu.tril(diagonal=-1).any():
        raise IsNotUpperTriangular(
            "Argument tensor (triu) is not upper triangular"
        )


def validate_full_fct(full_fct: Tensor) -> None:
    """
    Validates a batch of general Cholesky factors.

    Args:
        full_fct: Tensor expected to contain upper-triangular Cholesky
            factors with positive diagonal entries.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_square_matrix_batch(full_fct)
    validate_triu(full_fct)

    if (full_fct.diagonal(dim1=-2, dim2=-1) <= 0).any():
        raise HasNonPositiveDiagonalElement(
            "Argument tensor (full_fct) has a non-positive diagonal element"
        )


def validate_norm_logit(norm_logit: Tensor) -> None:
    """
    Validates a batch of normalized logit values.

    Args:
        norm_logit: Tensor expected to encode normalized log-probabilities.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_scalar_batch(norm_logit)

    try:
        assert_close(
            norm_logit.exp().sum(),
            tensor(1.0),
            check_dtype=False,
            check_device=False
        )
    except:
        raise IsNotNormalized(
            "Argument tensor (norm_logit) is not normalized"
        )


def validate_full_gaussian_parametrization(
    mean: Tensor,
    full_fct: Tensor,
    norm_logit: Optional[Tensor] = None
) -> None:
    """
    Validates a general Gaussian parametrization.

    Args:
        mean: Tensor containing Gaussian mean vectors.
        full_fct: Tensor containing upper-triangular Cholesky factors.
        norm_logit: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(mean)
    validate_full_fct(full_fct)

    *mean_batch_shape, mean_last_dim = list(mean.shape)
    *full_fct_batch_shape, _, full_fct_last_dim = list(full_fct.shape)

    if mean_last_dim != full_fct_last_dim:
        raise HaveIncompatibleDims(
            "Trailing dims of first tensor (mean) and second tensor "
            "(full_fct) are incompatible"
        )

    if not are_broadcastable(mean_batch_shape, full_fct_batch_shape):
        raise HaveNonBroadcastableShapes(
            "Batch dims of first tensor (mean) and second tensor "
            "(full_fct) are not broadcastable"
        )

    if norm_logit is not None:
        validate_norm_logit(norm_logit)

        norm_logit_batch_shape = list(norm_logit.shape)
        batch_shape = broadcast_shapes(mean_batch_shape, full_fct_batch_shape)
        batch_shape = list(batch_shape)

        if batch_shape != norm_logit_batch_shape:
            raise HaveNonMatchingShapes(
                "Broadcasting shape of first (mean) and second tensor "
                "(full_fct) does not match the shape of third tensor "
                "(norm_logit)"
            )


def validate_full_gaussian_specification(
    inpt: Tensor,
    mean: Tensor,
    full_fct: Tensor,
    norm_logit: Optional[Tensor] = None
) -> None:
    """
    Validates a general Gaussian parametrization and its relation with the
    input tensor.

    Args:
        inpt: Input tensor expected to be compatible with the Gaussian means.
        mean: Tensor containing Gaussian mean vectors.
        full_fct: Tensor containing upper-triangular Cholesky factors.
        norm_logit: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batches(inpt, mean)
    validate_full_gaussian_parametrization(mean, full_fct, norm_logit)


def validate_full_gaussian_parametrizations(
    mean_0: Tensor,
    full_fct_0: Tensor,
    mean_1: Tensor,
    full_fct_1: Tensor,
) -> None:
    """
    Validates a pair of general Gaussian parametrizations.

    Args:
        mean_0: Mean vectors for the first Gaussian parametrization.
        full_fct_0: Cholesky factors for the first Gaussian parametrization.
        mean_1: Mean vectors for the second Gaussian parametrization.
        full_fct_1: Cholesky factors for the second Gaussian parametrization.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_full_gaussian_parametrization(mean_0, full_fct_0)
    validate_full_gaussian_parametrization(mean_1, full_fct_1)

    *_, mean_0_last_dim = list(mean_0.shape)
    *_, mean_1_last_dim = list(mean_1.shape)

    if mean_0_last_dim != mean_1_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (mean_0) and third tensor (mean_1) are "
            "not compatible"
        )


def validate_diag_fct(diag_fct: Tensor) -> None:
    """
    Validates a diagonally-restricted batch of Cholesky factors.

    Args:
        diag_fct: Tensor expected to contain positive diagonal Cholesky
            factors.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(diag_fct)

    if (diag_fct <= 0).any():
        raise HasNonPositiveElement(
            "Argument tensor (diag_fct) has a non-positive element"
        )


def validate_diag_gaussian_parametrization(
    mean: Tensor,
    diag_fct: Tensor,
    norm_logit: Optional[Tensor] = None
) -> None:
    """
    Validates a diagonally-constrained Gaussian parametrization.

    Args:
        mean: Tensor containing Gaussian mean vectors.
        diag_fct: Tensor containing diagonal Cholesky factors.
        norm_logit: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(mean)
    validate_diag_fct(diag_fct)

    *mean_batch_shape, mean_last_dim = list(mean.shape)
    *diag_fct_batch_shape, diag_fct_last_dim = list(diag_fct.shape)

    if mean_last_dim != diag_fct_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (mean) and second tensor (diag_fct) "
            "are not compatible"
        )

    if not are_broadcastable(mean_batch_shape, diag_fct_batch_shape):
        raise HaveNonBroadcastableShapes(
            "Batch dims of first tensor (mean) and second tensor "
            "(diag_fct) are not broadcastable"
        )

    if norm_logit is not None:
        validate_norm_logit(norm_logit)

        norm_logit_batch_shape = list(norm_logit.shape)

        batch_shape = broadcast_shapes(mean_batch_shape, diag_fct_batch_shape)
        batch_shape = list(batch_shape)

        if batch_shape != norm_logit_batch_shape:
            raise HaveNonMatchingShapes(
                "Broadcasting shape of first (mean) and second tensor "
                "(diag_fct) does not match the shape of third tensor "
                "(norm_logit)"
            )


def validate_diag_gaussian_specification(
    inpt: Tensor,
    mean: Tensor,
    diag_fct: Tensor,
    norm_logit: Optional[Tensor] = None
) -> None:
    """
    Validates a diagonally-constrained Gaussian parametrization and its
    relation with the input tensor.

    Args:
        inpt: Input tensor expected to be compatible with the Gaussian means.
        mean: Tensor containing Gaussian mean vectors.
        diag_fct: Tensor containing diagonal Cholesky factors.
        norm_logit: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batches(inpt, mean)
    validate_diag_gaussian_parametrization(mean, diag_fct, norm_logit)


def validate_diag_gaussian_parametrizations(
    mean_0: Tensor,
    diag_fct_0: Tensor,
    mean_1: Tensor,
    diag_fct_1: Tensor,
) -> None:
    """
    Validates a pair of diagonally-constrained Gaussian parametrizations.

    Args:
        mean_0: Mean vectors for the first Gaussian parametrization.
        diag_fct_0: Diagonal factors for the first Gaussian parametrization.
        mean_1: Mean vectors for the second Gaussian parametrization.
        diag_fct_1: Diagonal factors for the second Gaussian parametrization.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_diag_gaussian_parametrization(mean_0, diag_fct_0)
    validate_diag_gaussian_parametrization(mean_1, diag_fct_1)

    *_, mean_0_last_dim = list(mean_0.shape)
    *_, mean_1_last_dim = list(mean_1.shape)

    if mean_0_last_dim != mean_1_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (mean_0) and third tensor (mean_1) are "
            "not compatible"
        )


def validate_iso_fct(iso_fct: Tensor) -> None:
    """
    Validates an isometrically-restricted batch of Cholesky factors.

    Args:
        iso_fct: Tensor expected to contain positive isometric scale factors.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_scalar_batch(iso_fct)

    if (iso_fct <= 0).any():
        raise HasNonPositiveElement(
            "Argument tensor (iso_fct) has a non-positive element"
        )


def validate_iso_gaussian_parametrization(
    mean: Tensor,
    iso_fct: Tensor,
    norm_logit: Optional[Tensor] = None
) -> None:
    """
    Validates an isometrically-constrained Gaussian parametrization.

    Args:
        mean: Tensor containing Gaussian mean vectors.
        iso_fct: Tensor containing isometric scale factors.
        norm_logit: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batch(mean)
    validate_iso_fct(iso_fct)

    *mean_batch_shape, _ = list(mean.shape)
    iso_fct_batch_shape = list(iso_fct.shape)

    if not are_broadcastable(mean_batch_shape, iso_fct_batch_shape):
        raise HaveNonBroadcastableShapes(
            "Batch dims of first tensor (mean) and second tensor "
            "(iso_fct) are not broadcastable"
        )

    if norm_logit is not None:
        validate_norm_logit(norm_logit)

        norm_logit_batch_shape = list(norm_logit.shape)

        batch_shape = broadcast_shapes(mean_batch_shape, iso_fct_batch_shape)
        batch_shape = list(batch_shape)

        if batch_shape != norm_logit_batch_shape:
            raise HaveNonMatchingShapes(
                "Broadcasting shape of first (mean) and second tensor "
                "(iso_fct) does not match the shape of third tensor "
                "(norm_logit)"
            )


def validate_iso_gaussian_specification(
    inpt: Tensor,
    mean: Tensor,
    iso_fct: Tensor,
    norm_logit: Optional[Tensor] = None
) -> None:
    """
    Validates an isometrically-constrained Gaussian parametrization and its
    relation with the input tensor.

    Args:
        inpt: Input tensor expected to be compatible with the Gaussian means.
        mean: Tensor containing Gaussian mean vectors.
        iso_fct: Tensor containing isometric scale factors.
        norm_logit: Optional tensor containing normalized component logits.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_vector_batches(inpt, mean)
    validate_iso_gaussian_parametrization(mean, iso_fct, norm_logit)


def validate_iso_gaussian_parametrizations(
    mean_0: Tensor,
    iso_fct_0: Tensor,
    mean_1: Tensor,
    iso_fct_1: Tensor,
) -> None:
    """
    Validates a pair of isometrically-constrained Gaussian parametrization.

    Args:
        mean_0: Mean vectors for the first Gaussian parametrization.
        iso_fct_0: Isometric factors for the first Gaussian parametrization.
        mean_1: Mean vectors for the second Gaussian parametrization.
        iso_fct_1: Isometric factors for the second Gaussian parametrization.

    Returns:
        None: This function returns only after successful validation.
    """

    validate_iso_gaussian_parametrization(mean_0, iso_fct_0)
    validate_iso_gaussian_parametrization(mean_1, iso_fct_1)

    *_, mean_0_last_dim = list(mean_0.shape)
    *_, mean_1_last_dim = list(mean_1.shape)

    if mean_0_last_dim != mean_1_last_dim:
        raise HaveIncompatibleDims(
            "Last dim of first tensor (mean_0) and third tensor (mean_1) are "
            "not compatible"
        )
