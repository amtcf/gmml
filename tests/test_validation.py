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
        """

        self.shape_0 = [1]
        self.shape_1 = [2]
        self.shape_2 = [1, 2]

    def test_01(self: Self) -> None:
        """
        """

        if not val.are_broadcastable(self.shape_0, self.shape_1):
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        if not val.are_broadcastable(self.shape_1, self.shape_0):
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        """

        if not val.are_broadcastable(self.shape_0, self.shape_2):
            self.assertTrue(False)

    def test_12(self: Self) -> None:
        """
        """

        if not val.are_broadcastable(self.shape_1, self.shape_2):
            self.assertTrue(False)

    def test_012(self: Self) -> None:
        """
        """

        if not val.are_broadcastable(
            self.shape_0,
            self.shape_1,
            self.shape_2
        ):
            self.assertTrue(False)


class TestValidateNonEmpty(TestCase):
    """
    'validate_non_empty' unit testing
    """

    def test_0(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [0.0]
        )

        try:
            val.validate_non_empty(inpt_0)
        except val.IsEmpty:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor([])

        try:
            val.validate_non_empty(inpt_1)
        except val.IsEmpty:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor([]).reshape(0, 2)

        try:
            val.validate_non_empty(inpt_2)
        except val.IsEmpty:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateScalarBatch(TestCase):
    """
    'validate_scalar_batch' testing unit
    """

    def test_0(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [0.0]
        )

        try:
            val.validate_scalar_batch(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [0.0, 0.0]
        )

        try:
            val.validate_scalar_batch(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        try:
            val.validate_scalar_batch(inpt_2)
        except:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        """

        inpt_3 = torch.tensor(0.0)

        try:
            val.validate_scalar_batch(inpt_3)
        except val.IsNotScalarBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVector(TestCase):
    """
    'validate_vector' unit testing
    """

    def test_0(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [0.0]
        )

        try:
            val.validate_vector(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [0.0, 0.0]
        )

        try:
            val.validate_vector(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(0.0)

        try:
            val.validate_vector(inpt_2)
        except val.IsNotVector:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVectorBatch(TestCase):
    """
    'validate_vector_batch' unit testing
    """

    def test_0(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_vector_batch(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_vector_batch(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [0.0, 0.0]
        )

        try:
            val.validate_vector_batch(inpt_2)
        except val.IsNotVectorBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVectorBatches(TestCase):
    """
    'validate_vector_batches' unit testing
    """

    def setUp(self: Self) -> None:
        """
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
        """

        try:
            val.validate_vector_batches(self.inpt_0, self.inpt_1)
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        try:
            val.validate_vector_batches(self.inpt_1, self.inpt_0)
        except:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        """

        try:
            val.validate_vector_batches(self.inpt_0, self.inpt_2)
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        try:
            val.validate_vector_batches(self.inpt_2, self.inpt_0)
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateSquareMatrixBatch(TestCase):
    """
    'validate_square_matrix_batch' unit testing
    """

    def test_0(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
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
        except val.IsNotMatrixBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateSquareMatrixBatches(TestCase):
    """
    'validate_square_matrix_batches' unit testing
    """

    def setUp(self: Self) -> None:
        """
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
        """

        try:
            val.validate_square_matrix_batches(self.inpt_0, self.inpt_1)
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        try:
            val.validate_square_matrix_batches(self.inpt_1, self.inpt_0)
        except:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        """

        try:
            val.validate_square_matrix_batches(self.inpt_0, self.inpt_2)
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        try:
            val.validate_square_matrix_batches(self.inpt_2, self.inpt_0)
        except val.HaveIncompatibleDims:
            pass
        except:
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
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_trius(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

        try:
            val.validate_trius(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullFactors(TestCase):
    """
    'validate_full_factors' unit testing
    """


    def test_0(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [
                [
                    [- 1.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            val.validate_full_factors(inpt_2)
        except val.HasNonPositiveDiagonalElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateNormLogits(TestCase):
    """
    'validate_norm_logits' unit testing
    """

    def test_0(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [0.0, 0.0]
        )
        inpt_0 = log_softmax(inpt_0)

        try:
            val.validate_norm_logits(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [1.0, 1.0]
        )

        try:
            val.validate_norm_logits(inpt_2)
        except val.IsNotNormalized:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullGaussianParametrization(TestCase):
    """
    'validate_full_gaussian_parametrization' unit testing
    """

    def setUp(self: Self) -> None:
        """
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
        self.norm_logits_0 = torch.tensor(
            [0.0, 0.0]
        )
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        """

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0,
                self.full_factors_0
            )
        except:
            self.assertTrue(False)

    def test_000(self: Self) -> None:
        """
        """

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0,
                self.full_factors_0,
                self.norm_logits_0
            )
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        means_1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        try:
            val.validate_full_gaussian_parametrization(
                means_1,
                self.full_factors_0
            )
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
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
                self.means_0,
                full_factors_1
            )
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
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
                means_2,
                self.full_factors_0
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
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
                self.means_0,
                full_factors_2
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_002(self: Self) -> None:
        """
        """

        norm_logits_2 = torch.tensor(
            [0.0, 0.0, 0.0]
        )
        norm_logits_2 = log_softmax(norm_logits_2)

        try:
            val.validate_full_gaussian_parametrization(
                self.means_0,
                self.full_factors_0,
                norm_logits_2
            )
        except val.HaveNonMatchingShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullGaussianParametrizations(TestCase):
    """
    'validate_full_gaussian_parametrizations' unit testing
    """

    def setUp(self: Self) -> None:
        """
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
                ]
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
                ]
            ]
        )

    def test_01(self: Self) -> None:
        """
        """

        try:
            val.validate_full_gaussian_parametrizations(
                self.means_0,
                self.full_factors_0,
                self.means_1,
                self.full_factors_1
            )
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        try:
            val.validate_full_gaussian_parametrizations(
                self.means_1,
                self.full_factors_1,
                self.means_0,
                self.full_factors_0
            )
        except:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
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
                self.means_0,
                self.full_factors_0,
                means_2,
                full_factors_2
            )
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagFactors(TestCase):
    """
    'validate_diag_factors' unit testing
    """

    def test_0(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [
                [1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_factors(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [
                [1.0],
            ]
        )

        try:
            val.validate_diag_factors(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
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
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        """

        inpt_3 = torch.tensor(
            [
                [- 1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_factors(inpt_3)
        except val.HasNonPositiveElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagGaussianParametrization(TestCase):
    """
    'validate_diag_gaussian_parametrization' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        )
        self.diag_factors_0 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.norm_logits_0 = torch.tensor(
            [0.0, 0.0]
        )
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        """

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0,
                self.diag_factors_0
            )
        except:
            self.assertTrue(False)

    def test_000(self: Self) -> None:
        """
        """

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0,
                self.diag_factors_0,
                self.norm_logits_0
            )
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        means_1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        try:
            val.validate_diag_gaussian_parametrization(
                means_1,
                self.diag_factors_0
            )
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        """

        diag_factors_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0,
                diag_factors_1
            )
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
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
                means_2,
                self.diag_factors_0
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
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
                self.means_0,
                diag_factors_2
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_002(self: Self) -> None:
        """
        """

        norm_logits_2 = torch.tensor(
            [0.0, 0.0, 0.0]
        )
        norm_logits_2 = log_softmax(norm_logits_2)

        try:
            val.validate_diag_gaussian_parametrization(
                self.means_0,
                self.diag_factors_0,
                norm_logits_2
            )
        except val.HaveNonMatchingShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagGaussianParametrizations(TestCase):
    """
    'validate_diag_gaussian_parametrizations' unit testing
    """

    def setUp(self: Self) -> None:
        """
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
        """

        try:
            val.validate_diag_gaussian_parametrizations(
                self.means_0,
                self.diag_factors_0,
                self.means_1,
                self.diag_factors_1
            )
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        try:
            val.validate_diag_gaussian_parametrizations(
                self.means_1,
                self.diag_factors_1,
                self.means_0,
                self.diag_factors_0
            )
        except:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
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
                self.means_0,
                self.diag_factors_0,
                means_2,
                diag_factors_2
            )
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoFactors(TestCase):
    """
    'validate_is_fct' unit testing
    """

    def test_0(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [1.0, 1.0]
        )

        try:
            val.validate_iso_factors(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [1.0]
        )

        try:
            val.validate_iso_factors(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        try:
            val.validate_iso_factors(inpt_2)
        except:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        """

        inpt_3 = torch.tensor(
            [0.0, 1.0]
        )

        try:
            val.validate_iso_factors(inpt_3)
        except val.HasNonPositiveElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_4(self: Self) -> None:
        """
        """

        inpt_4 = torch.tensor(
            [- 1.0, 1.0]
        )

        try:
            val.validate_iso_factors(inpt_4)
        except val.HasNonPositiveElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoGaussianParametrization(TestCase):
    """
    'validate_iso_gaussian_parametrization' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        )
        self.iso_factors_0 = torch.tensor(
            [1.0, 1.0]
        )
        self.norm_logits_0 = torch.tensor(
            [0.0, 0.0]
        )
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        """

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0,
                self.iso_factors_0
            )
        except:
            self.assertTrue(False)

    def test_000(self: Self) -> None:
        """
        """

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0,
                self.iso_factors_0,
                self.norm_logits_0
            )
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
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
                means_1,
                self.iso_factors_0
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        """

        iso_factors_1 = torch.tensor(
            [1.0, 1.0, 1.0]
        )

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0,
                iso_factors_1
            )
        except val.HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_001(self: Self) -> None:
        """
        """

        norm_logits_1 = torch.tensor(
            [0.0, 0.0, 0.0]
        )
        norm_logits_1 = log_softmax(norm_logits_1)

        try:
            val.validate_iso_gaussian_parametrization(
                self.means_0,
                self.iso_factors_0,
                norm_logits_1
            )
        except val.HaveNonMatchingShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoGaussianParametrizations(TestCase):
    """
    'validate_iso_gaussian_parametrizations' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_factors_0 = torch.tensor(
            [1.0, 1.0]
        )

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_factors_1 = torch.tensor(
            [1.0, 1.0]
        )

    def test_01(self: Self) -> None:
        """
        """

        try:
            val.validate_iso_gaussian_parametrizations(
                self.means_0,
                self.iso_factors_0,
                self.means_1,
                self.iso_factors_1
            )
        except:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        try:
            val.validate_iso_gaussian_parametrizations(
                self.means_1,
                self.iso_factors_1,
                self.means_0,
                self.iso_factors_0
            )
        except:
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        iso_factors_2 = torch.tensor(
            [1.0, 1.0]
        )

        try:
            val.validate_iso_gaussian_parametrizations(
                self.means_0,
                self.iso_factors_0,
                means_2,
                iso_factors_2
            )
        except val.HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)
