"""
Validation testing module
"""

from typing import Self
from unittest import TestCase

import torch

import gmml.validation as val

from .utils import log_softmax


class TestAreBroadcastable(TestCase):
    """
    'are_broadcastable' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.shape_0 = [1]
        self.shape_1 = [2]
        self.shape_2 = [1, 2]

    def test_01(self: Self) -> None:
        """
        Checks the output shape of 'are_broadcastable' on case 01.
        """

        if not val.are_broadcastable(self.shape_0, self.shape_1):
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks the output shape of 'are_broadcastable' on case 10.
        """

        if not val.are_broadcastable(self.shape_1, self.shape_0):
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks the output shape of 'are_broadcastable' on case 02.
        """

        if not val.are_broadcastable(self.shape_0, self.shape_2):
            self.assertTrue(False)

    def test_12(self: Self) -> None:
        """
        Checks the output shape of 'are_broadcastable' on case 12.
        """

        if not val.are_broadcastable(self.shape_1, self.shape_2):
            self.assertTrue(False)

    def test_012(self: Self) -> None:
        """
        Checks the output shape of 'are_broadcastable' on case 012.
        """

        if not val.are_broadcastable(self.shape_0, self.shape_1, self.shape_2):
            self.assertTrue(False)


class TestValidateNonEmpty(TestCase):
    """
    'validate_non_empty' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_non_empty' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor([0.0])

        try:
            val.validate_non_empty(inpt_0)
        except val.IsEmpty:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_non_empty' raises IsEmpty on case 1.
        """

        inpt_1 = torch.tensor([])

        try:
            val.validate_non_empty(inpt_1)
        except val.IsEmpty:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_non_empty' raises IsEmpty on case 2.
        """

        inpt_2 = torch.tensor([]).reshape(0, 2)

        try:
            val.validate_non_empty(inpt_2)
        except val.IsEmpty:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateShape(TestCase):
    """
    'validate_shape' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Accepts a tensor with the expected shape.
        """

        inpt_0 = torch.zeros(2, 3)

        try:
            val.validate_shape(inpt_0, (2, 3))
        except val.HasUnexpectedShape:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Accepts a list as the expected shape.
        """

        inpt_1 = torch.zeros(1)

        try:
            val.validate_shape(inpt_1, [1])
        except val.HasUnexpectedShape:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Rejects a tensor whose shape differs from the expected one.
        """

        inpt_2 = torch.zeros(3, 2)

        try:
            val.validate_shape(inpt_2, (2, 3))
        except val.HasUnexpectedShape:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Rejects a tensor that is broadcastable to the expected shape.
        """

        inpt_3 = torch.zeros(3)

        try:
            val.validate_shape(inpt_3, (2, 3))
        except val.HasUnexpectedShape:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateScalarBatch(TestCase):
    """
    'validate_scalar_batch' testing unit
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_scalar_batch' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor([0.0])

        try:
            val.validate_scalar_batch(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_scalar_batch' accepts the valid input of case 1.
        """

        inpt_1 = torch.tensor([0.0, 0.0])

        try:
            val.validate_scalar_batch(inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_scalar_batch' accepts the valid input of case 2.
        """

        inpt_2 = torch.tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        try:
            val.validate_scalar_batch(inpt_2)
        except Exception:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks that 'validate_scalar_batch' raises IsNotScalarBatch on case 3.
        """

        inpt_3 = torch.tensor(0.0)

        try:
            val.validate_scalar_batch(inpt_3)
        except val.IsNotScalarBatch:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVector(TestCase):
    """
    'validate_vector' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_vector' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor([0.0])

        try:
            val.validate_vector(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_vector' accepts the valid input of case 1.
        """

        inpt_1 = torch.tensor([0.0, 0.0])

        try:
            val.validate_vector(inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_vector' raises IsNotVector on case 2.
        """

        inpt_2 = torch.tensor(0.0)

        try:
            val.validate_vector(inpt_2)
        except val.IsNotVector:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks that 'validate_vector' raises IsNotVector on case 3.
        """

        inpt_3 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_vector(inpt_3)
        except val.IsNotVector:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVectorBatch(TestCase):
    """
    'validate_vector_batch' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_vector_batch' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_vector_batch(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_vector_batch' accepts the valid input of case 1.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_vector_batch(inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_vector_batch' raises IsNotVectorBatch on case 2.
        """

        inpt_2 = torch.tensor([0.0, 0.0])

        try:
            val.validate_vector_batch(inpt_2)
        except val.IsNotVectorBatch:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVectorBatches(TestCase):
    """
    'validate_vector_batches' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        self.inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        self.inpt_2 = torch.tensor(
            [
                [0.0],
            ]
        )
        self.inpt_3 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.inpt_4 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_vector_batches' accepts the valid input of case
        01.
        """

        try:
            val.validate_vector_batches(self.inpt_0, self.inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_vector_batches' accepts the valid input of case
        10.
        """

        try:
            val.validate_vector_batches(self.inpt_1, self.inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'validate_vector_batches' raises HaveIncompatibleDims on
        case 02.
        """

        try:
            val.validate_vector_batches(self.inpt_0, self.inpt_2)
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks that 'validate_vector_batches' raises HaveIncompatibleDims on
        case 20.
        """

        try:
            val.validate_vector_batches(self.inpt_2, self.inpt_0)
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateSquareMatrixBatch(TestCase):
    """
    'validate_square_matrix_batch' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batch' accepts the valid input of
        case 0.
        """

        inpt_0 = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            val.validate_square_matrix_batch(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batch' raises IsNotMatrixBatch on
        case 1.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_square_matrix_batch(inpt_1)
        except val.IsNotMatrixBatch:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batch' accepts the valid input of
        case 2, a batch of 1 × 1 matrices.
        """

        inpt_2 = torch.tensor(
            [
                [
                    [0.0],
                ]
            ]
        )

        try:
            val.validate_square_matrix_batch(inpt_2)
        except Exception:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batch' raises
        IsNotSquareMatrixBatch on case 3.
        """

        inpt_3 = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            val.validate_square_matrix_batch(inpt_3)
        except val.IsNotSquareMatrixBatch:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateSquareMatrixBatches(TestCase):
    """
    'validate_square_matrix_batches' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.inpt_0 = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )
        self.inpt_1 = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )
        self.inpt_2 = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batches' accepts the valid input of
        case 01.
        """

        try:
            val.validate_square_matrix_batches(self.inpt_0, self.inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batches' accepts the valid input of
        case 10.
        """

        try:
            val.validate_square_matrix_batches(self.inpt_1, self.inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batches' raises
        HaveIncompatibleDims on case 02.
        """

        try:
            val.validate_square_matrix_batches(self.inpt_0, self.inpt_2)
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks that 'validate_square_matrix_batches' raises
        HaveIncompatibleDims on case 20.
        """

        try:
            val.validate_square_matrix_batches(self.inpt_2, self.inpt_0)
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateTrius(TestCase):
    """
    'validate_trius' unit testing
    'validate_trius' is always applied after 'validate_square_matrix_batch'
    Only square_matrix_batches are considered (temporarily)
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_trius' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_trius(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_trius' accepts the valid input of case 1.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_trius(inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_trius' raises IsNotUpperTriangular on case 2.
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

        try:
            val.validate_trius(inpt_2)
        except val.IsNotUpperTriangular:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullFactors(TestCase):
    """
    'validate_full_factors' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_full_factors' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        )

        try:
            val.validate_full_factors(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_full_factors' raises
        HasNonPositiveDiagonalElement on case 1.
        """

        inpt_1 = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            val.validate_full_factors(inpt_1)
        except val.HasNonPositiveDiagonalElement:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_full_factors' raises
        HasNonPositiveDiagonalElement on case 2.
        """

        inpt_2 = torch.tensor(
            [
                [
                    [-1.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            val.validate_full_factors(inpt_2)
        except val.HasNonPositiveDiagonalElement:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateNormLogits(TestCase):
    """
    'validate_norm_logits' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_norm_logits' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor([0.0, 0.0])
        inpt_0 = log_softmax(inpt_0)

        try:
            val.validate_norm_logits(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_norm_logits' accepts the valid input of case 1.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        inpt_1 = log_softmax(inpt_1)

        try:
            val.validate_norm_logits(inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_norm_logits' raises IsNotNormalized on case 2.
        """

        inpt_2 = torch.tensor([1.0, 1.0])

        try:
            val.validate_norm_logits(inpt_2)
        except val.IsNotNormalized:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Rejects a batch normalized over all its entries, rather than over
        its last dimension.
        """

        inpt_3 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        inpt_3 = inpt_3.flatten().log_softmax(-1).unflatten(-1, (2, 2))

        try:
            val.validate_norm_logits(inpt_3)
        except val.IsNotNormalized:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullGaussianParametrization(TestCase):
    """
    'validate_full_gaussian_parametrization' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_factors_0 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )
        self.norm_logits_0 = torch.tensor([0.0, 0.0])
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrization' accepts the valid
        input of case 00.
        """

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0, self.full_factors_0
            )
        except Exception:
            self.assertTrue(False)

    def test_000(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrization' accepts the valid
        input of case 000.
        """

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0, self.full_factors_0, self.norm_logits_0
            )
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrization' raises
        HaveIncompatibleDims on case 10.
        """

        means_1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        try:
            val.validate_full_gaussian_parametrization(
                means_1, self.full_factors_0
            )
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrization' raises
        HaveIncompatibleDims on case 01.
        """

        full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ]
        )

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0, full_factors_1
            )
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrization' raises
        HaveNonBroadcastableShapes on case 20.
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_full_gaussian_parametrization(
                means_2, self.full_factors_0
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrization' raises
        HaveNonBroadcastableShapes on case 02.
        """

        full_factors_2 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0, full_factors_2
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_002(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrization' raises
        HaveNonMatchingShapes on case 002.
        """

        norm_logits_2 = torch.tensor([0.0, 0.0, 0.0])
        norm_logits_2 = log_softmax(norm_logits_2)

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0, self.full_factors_0, norm_logits_2
            )
        except val.HaveNonMatchingShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullGaussianParametrizations(TestCase):
    """
    'validate_full_gaussian_parametrizations' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_factors_0 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrizations' accepts the valid
        input of case 01.
        """

        try:
            val.validate_full_gaussian_parametrizations(
                self.means_0,
                self.full_factors_0,
                self.means_1,
                self.full_factors_1,
            )
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrizations' accepts the valid
        input of case 10.
        """

        try:
            val.validate_full_gaussian_parametrizations(
                self.means_1,
                self.full_factors_1,
                self.means_0,
                self.full_factors_0,
            )
        except Exception:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'validate_full_gaussian_parametrizations' raises
        HaveIncompatibleDims on case 02.
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        full_factors_2 = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ]
        )

        try:
            val.validate_full_gaussian_parametrizations(
                self.means_0, self.full_factors_0, means_2, full_factors_2
            )
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagFactors(TestCase):
    """
    'validate_diag_factors' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_diag_factors' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor(
            [
                [1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_factors(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_diag_factors' accepts the valid input of case 1.
        """

        inpt_1 = torch.tensor(
            [
                [1.0],
            ]
        )

        try:
            val.validate_diag_factors(inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_diag_factors' raises HasNonPositiveElement on
        case 2.
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 1.0],
            ]
        )

        try:
            val.validate_diag_factors(inpt_2)
        except val.HasNonPositiveElement:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks that 'validate_diag_factors' raises HasNonPositiveElement on
        case 3.
        """

        inpt_3 = torch.tensor(
            [
                [-1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_factors(inpt_3)
        except val.HasNonPositiveElement:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagGaussianParametrization(TestCase):
    """
    'validate_diag_gaussian_parametrization' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.means_0 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        self.diag_factors_0 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.norm_logits_0 = torch.tensor([0.0, 0.0])
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrization' accepts the valid
        input of case 00.
        """

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0, self.diag_factors_0
            )
        except Exception:
            self.assertTrue(False)

    def test_000(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrization' accepts the valid
        input of case 000.
        """

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0, self.diag_factors_0, self.norm_logits_0
            )
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrization' raises
        HaveIncompatibleDims on case 10.
        """

        means_1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        try:
            val.validate_diag_gaussian_parametrization(
                means_1, self.diag_factors_0
            )
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrization' raises
        HaveIncompatibleDims on case 01.
        """

        diag_factors_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0, diag_factors_1
            )
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrization' raises
        HaveNonBroadcastableShapes on case 20.
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_diag_gaussian_parametrization(
                means_2, self.diag_factors_0
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrization' raises
        HaveNonBroadcastableShapes on case 02.
        """

        diag_factors_2 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0, diag_factors_2
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_002(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrization' raises
        HaveNonMatchingShapes on case 002.
        """

        norm_logits_2 = torch.tensor([0.0, 0.0, 0.0])
        norm_logits_2 = log_softmax(norm_logits_2)

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0, self.diag_factors_0, norm_logits_2
            )
        except val.HaveNonMatchingShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagGaussianParametrizations(TestCase):
    """
    'validate_diag_gaussian_parametrizations' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.diag_factors_0 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.diag_factors_1 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrizations' accepts the valid
        input of case 01.
        """

        try:
            val.validate_diag_gaussian_parametrizations(
                self.means_0,
                self.diag_factors_0,
                self.means_1,
                self.diag_factors_1,
            )
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrizations' accepts the valid
        input of case 10.
        """

        try:
            val.validate_diag_gaussian_parametrizations(
                self.means_1,
                self.diag_factors_1,
                self.means_0,
                self.diag_factors_0,
            )
        except Exception:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'validate_diag_gaussian_parametrizations' raises
        HaveIncompatibleDims on case 02.
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        diag_factors_2 = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_gaussian_parametrizations(
                self.means_0, self.diag_factors_0, means_2, diag_factors_2
            )
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoFactors(TestCase):
    """
    'validate_is_fct' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_iso_factors' accepts the valid input of case 0.
        """

        inpt_0 = torch.tensor([1.0, 1.0])

        try:
            val.validate_iso_factors(inpt_0)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_iso_factors' accepts the valid input of case 1.
        """

        inpt_1 = torch.tensor([1.0])

        try:
            val.validate_iso_factors(inpt_1)
        except Exception:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_iso_factors' accepts the valid input of case 2.
        """

        inpt_2 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        try:
            val.validate_iso_factors(inpt_2)
        except Exception:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks that 'validate_iso_factors' raises HasNonPositiveElement on case
        3.
        """

        inpt_3 = torch.tensor([0.0, 1.0])

        try:
            val.validate_iso_factors(inpt_3)
        except val.HasNonPositiveElement:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_4(self: Self) -> None:
        """
        Checks that 'validate_iso_factors' raises HasNonPositiveElement on case
        4.
        """

        inpt_4 = torch.tensor([-1.0, 1.0])

        try:
            val.validate_iso_factors(inpt_4)
        except val.HasNonPositiveElement:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoGaussianParametrization(TestCase):
    """
    'validate_iso_gaussian_parametrization' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.means_0 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        self.iso_factors_0 = torch.tensor([1.0, 1.0])
        self.norm_logits_0 = torch.tensor([0.0, 0.0])
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrization' accepts the valid
        input of case 00.
        """

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0, self.iso_factors_0
            )
        except Exception:
            self.assertTrue(False)

    def test_000(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrization' accepts the valid
        input of case 000.
        """

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0, self.iso_factors_0, self.norm_logits_0
            )
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrization' raises
        HaveNonBroadcastableShapes on case 10.
        """

        means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_iso_gaussian_parametrization(
                means_1, self.iso_factors_0
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrization' raises
        HaveNonBroadcastableShapes on case 01.
        """

        iso_factors_1 = torch.tensor([1.0, 1.0, 1.0])

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0, iso_factors_1
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_001(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrization' raises
        HaveNonMatchingShapes on case 001.
        """

        norm_logits_1 = torch.tensor([0.0, 0.0, 0.0])
        norm_logits_1 = log_softmax(norm_logits_1)

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0, self.iso_factors_0, norm_logits_1
            )
        except val.HaveNonMatchingShapes:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoGaussianParametrizations(TestCase):
    """
    'validate_iso_gaussian_parametrizations' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_factors_0 = torch.tensor([1.0, 1.0])

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_factors_1 = torch.tensor([1.0, 1.0])

    def test_01(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrizations' accepts the valid
        input of case 01.
        """

        try:
            val.validate_iso_gaussian_parametrizations(
                self.means_0,
                self.iso_factors_0,
                self.means_1,
                self.iso_factors_1,
            )
        except Exception:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrizations' accepts the valid
        input of case 10.
        """

        try:
            val.validate_iso_gaussian_parametrizations(
                self.means_1,
                self.iso_factors_1,
                self.means_0,
                self.iso_factors_0,
            )
        except Exception:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'validate_iso_gaussian_parametrizations' raises
        HaveIncompatibleDims on case 02.
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        iso_factors_2 = torch.tensor([1.0, 1.0])

        try:
            val.validate_iso_gaussian_parametrizations(
                self.means_0, self.iso_factors_0, means_2, iso_factors_2
            )
        except val.HaveIncompatibleDims:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestIsBroadcastableTo(TestCase):
    """
    'is_broadcastable_to' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'is_broadcastable_to' accepts shapes broadcasting to the
        target shape.
        """

        self.assertTrue(val.is_broadcastable_to([1], [3]))
        self.assertTrue(val.is_broadcastable_to([3], [2, 3]))
        self.assertTrue(val.is_broadcastable_to([2, 3], [2, 3]))

    def test_1(self: Self) -> None:
        """
        Checks that 'is_broadcastable_to' rejects shapes that would change
        the target shape, or that are not broadcastable with it.
        """

        self.assertFalse(val.is_broadcastable_to([3], [1]))
        self.assertFalse(val.is_broadcastable_to([2, 3], [3]))
        self.assertFalse(val.is_broadcastable_to([2], [3]))


class TestValidateDim(TestCase):
    """
    'validate_dim' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_dim' accepts a positive integer.
        """

        try:
            val.validate_dim(1)
        except Exception:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_dim' raises IsNotPositiveInteger on zero, a
        negative integer, a boolean and a float.
        """

        for dim in [0, -1, True, 2.0]:
            with self.subTest(dim=dim):
                try:
                    val.validate_dim(dim)
                except val.IsNotPositiveInteger:
                    pass
                except Exception:
                    self.assertTrue(False)
                else:
                    self.assertTrue(False)


class TestValidateJointEntropySpecification(TestCase):
    """
    'validate_joint_entropy_specification' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'validate_joint_entropy_specification' accepts per
        component and shared modal entropies.
        """

        norm_logits_0 = log_softmax(torch.zeros(3))

        for modal_entropies in [torch.zeros(3), torch.zeros(1)]:
            with self.subTest(shape=tuple(modal_entropies.shape)):
                try:
                    val.validate_joint_entropy_specification(
                        modal_entropies, torch.zeros(3, 3), norm_logits_0
                    )
                except Exception:
                    self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'validate_joint_entropy_specification' raises
        HaveNonBroadcastableShapes on mismatching modal entropies or KL
        divergences.
        """

        norm_logits_1 = log_softmax(torch.zeros(3))

        for modal_entropies, kl_divs in [
            (torch.zeros(2), torch.zeros(3, 3)),
            (torch.zeros(3), torch.zeros(3, 2)),
            (torch.zeros(3), torch.zeros(2, 3, 3)),
        ]:
            with self.subTest(kl_divs=tuple(kl_divs.shape)):
                try:
                    val.validate_joint_entropy_specification(
                        modal_entropies, kl_divs, norm_logits_1
                    )
                except val.HaveNonBroadcastableShapes:
                    pass
                except Exception:
                    self.assertTrue(False)
                else:
                    self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'validate_joint_entropy_specification' raises
        IsNotVector on a batch of normalized logits.
        """

        norm_logits_2 = log_softmax(torch.zeros(2, 3))

        try:
            val.validate_joint_entropy_specification(
                torch.zeros(3), torch.zeros(3, 3), norm_logits_2
            )
        except val.IsNotVector:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)
