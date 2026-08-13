"""
Validation testing module
"""

from __future__ import annotations

from unittest import TestCase

from torch import tensor

from gmml.validation import are_broadcastable
from gmml.validation import validate_non_empty, IsEmpty
from gmml.validation import validate_scalar_batch, IsNotScalarBatch
from gmml.validation import validate_vector, IsNotVector
from gmml.validation import validate_vector_batch, IsNotVectorBatch
from gmml.validation import validate_vector_batches
from gmml.validation import HaveIncompatibleDims, HaveNonBroadcastableShapes
from gmml.validation import validate_square_matrix_batch
from gmml.validation import IsNotMatrixBatch, IsNotSquareMatrixBatch
from gmml.validation import validate_square_matrix_batches
from gmml.validation import validate_triu, IsNotUpperTriangular
from gmml.validation import validate_full_fct, HasNonPositiveDiagonalElement
from gmml.validation import validate_norm_logit, IsNotNormalized
from gmml.validation import validate_full_gaussian_parametrization
from gmml.validation import HaveNonMatchingShapes
from gmml.validation import validate_full_gaussian_parametrizations
from gmml.validation import validate_diag_fct, HasNonPositiveElement
from gmml.validation import validate_diag_gaussian_parametrization
from gmml.validation import validate_diag_gaussian_parametrizations
from gmml.validation import validate_iso_fct
from gmml.validation import validate_iso_gaussian_parametrization
from gmml.validation import validate_iso_gaussian_parametrizations

from .util import log_softmax


class TestAreBroadcastable(TestCase):
    """
    'are_broadcastable' unit testing
    """

    def setUp(self: TestAreBroadcastable) -> None:
        """
        """

        self.shape_0 = [1]
        self.shape_1 = [2]
        self.shape_2 = [1, 2]

    def test_01(self: TestAreBroadcastable) -> None:
        """
        """

        if not are_broadcastable(self.shape_0, self.shape_1):
            self.assertTrue(False)

    def test_10(self: TestAreBroadcastable) -> None:
        """
        """

        if not are_broadcastable(self.shape_1, self.shape_0):
            self.assertTrue(False)

    def test_02(self: TestAreBroadcastable) -> None:
        """
        """

        if not are_broadcastable(self.shape_0, self.shape_2):
            self.assertTrue(False)

    def test_12(self: TestAreBroadcastable) -> None:
        """
        """

        if not are_broadcastable(self.shape_1, self.shape_2):
            self.assertTrue(False)

    def test_012(self: TestAreBroadcastable) -> None:
        """
        """

        if not are_broadcastable(
            self.shape_0,
            self.shape_1,
            self.shape_2
        ):
            self.assertTrue(False)


class TestValidateNonEmpty(TestCase):
    """
    'validate_non_empty' unit testing
    """

    def test_0(self: TestValidateNonEmpty) -> None:
        """
        """

        inpt_0 = tensor(
            [0.0]
        )

        try:
            validate_non_empty(inpt_0)
        except IsEmpty:
            self.assertTrue(False)

    def test_1(self: TestValidateNonEmpty) -> None:
        """
        """

        inpt_1 = tensor([])

        try:
            validate_non_empty(inpt_1)
        except IsEmpty:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: TestValidateNonEmpty) -> None:
        """
        """

        inpt_2 = tensor([]).reshape(0, 2)

        try:
            validate_non_empty(inpt_2)
        except IsEmpty:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateScalarBatch(TestCase):
    """
    'validate_scalar_batch' testing unit
    """

    def test_0(self: TestValidateScalarBatch) -> None:
        """
        """

        inpt_0 = tensor(
            [0.0]
        )

        try:
            validate_scalar_batch(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateScalarBatch) -> None:
        """
        """

        inpt_1 = tensor(
            [0.0, 0.0]
        )

        try:
            validate_scalar_batch(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: TestValidateScalarBatch) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        try:
            validate_scalar_batch(inpt_2)
        except:
            self.assertTrue(False)

    def test_3(self: TestValidateScalarBatch) -> None:
        """
        """

        inpt_3 = tensor(0.0)

        try:
            validate_scalar_batch(inpt_3)
        except IsNotScalarBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVector(TestCase):
    """
    'validate_vector' unit testing
    """

    def test_0(self: TestValidateVector) -> None:
        """
        """

        inpt_0 = tensor(
            [0.0]
        )

        try:
            validate_vector(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateVector) -> None:
        """
        """

        inpt_1 = tensor(
            [0.0, 0.0]
        )

        try:
            validate_vector(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: TestValidateVector) -> None:
        """
        """

        inpt_2 = tensor(0.0)

        try:
            validate_vector(inpt_2)
        except IsNotVector:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: TestValidateVector) -> None:
        """
        """

        inpt_3 = tensor(
            [
                [0.0, 0.0],
            ]
        )

        try:
            validate_vector(inpt_3)
        except IsNotVector:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVectorBatch(TestCase):
    """
    'validate_vector_batch' unit testing
    """

    def test_0(self: TestValidateVectorBatch) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )

        try:
            validate_vector_batch(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateVectorBatch) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            validate_vector_batch(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: TestValidateVectorBatch) -> None:
        """
        """

        inpt_2 = tensor(
            [0.0, 0.0]
        )

        try:
            validate_vector_batch(inpt_2)
        except IsNotVectorBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateVectorBatches(TestCase):
    """
    'validate_vector_batches' unit testing
    """

    def setUp(self: TestValidateVectorBatches) -> None:
        """
        """

        self.inpt_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )

        self.inpt_1 = tensor(
            [
                [0.0, 0.0],
            ]
        )

        self.inpt_2 = tensor(
            [
                [0.0],
            ]
        )
        self.inpt_3 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.inpt_4 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

    def test_01(self: TestValidateVectorBatches) -> None:
        """
        """

        try:
            validate_vector_batches(self.inpt_0, self.inpt_1)
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateVectorBatches) -> None:
        """
        """

        try:
            validate_vector_batches(self.inpt_1, self.inpt_0)
        except:
            self.assertTrue(False)

    def test_02(self: TestValidateVectorBatches) -> None:
        """
        """

        try:
            validate_vector_batches(self.inpt_0, self.inpt_2)
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: TestValidateVectorBatches) -> None:
        """
        """

        try:
            validate_vector_batches(self.inpt_2, self.inpt_0)
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateSquareMatrixBatch(TestCase):
    """
    'validate_square_matrix_batch' unit testing
    """

    def test_0(self: TestValidateSquareMatrixBatch) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            validate_square_matrix_batch(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateSquareMatrixBatch) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            validate_square_matrix_batch(inpt_1)
        except IsNotMatrixBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: TestValidateSquareMatrixBatch) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [
                    [0.0],
                ]
            ]
        )

        try:
            validate_square_matrix_batch(inpt_2)
        except IsNotMatrixBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: TestValidateSquareMatrixBatch) -> None:
        """
        """

        inpt_3 = tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            validate_square_matrix_batch(inpt_3)
        except IsNotSquareMatrixBatch:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateSquareMatrixBatches(TestCase):
    """
    'validate_square_matrix_batches' unit testing
    """

    def setUp(self: TestValidateSquareMatrixBatches) -> None:
        """
        """

        self.inpt_0 = tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )
        self.inpt_1 = tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )
        self.inpt_2 = tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )

    def test_01(self: TestValidateSquareMatrixBatches) -> None:
        """
        """

        try:
            validate_square_matrix_batches(self.inpt_0, self.inpt_1)
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateSquareMatrixBatches) -> None:
        """
        """

        try:
            validate_square_matrix_batches(self.inpt_1, self.inpt_0)
        except:
            self.assertTrue(False)

    def test_02(self: TestValidateSquareMatrixBatches) -> None:
        """
        """

        try:
            validate_square_matrix_batches(self.inpt_0, self.inpt_2)
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: TestValidateSquareMatrixBatches) -> None:
        """
        """

        try:
            validate_square_matrix_batches(self.inpt_2, self.inpt_0)
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateTriu(TestCase):
    """
    'validate_triu' unit testing
    'validate_triu' is always applied after 'validate_square_matrix_batch'
    Only square_matrix_batches are considered (temporarily)
    """

    def test_0(self: TestValidateTriu) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            validate_triu(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateTriu) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

        try:
            validate_triu(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: TestValidateTriu) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

        try:
            validate_triu(inpt_2)
        except IsNotUpperTriangular:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullFct(TestCase):
    """
    'validate_full_fct' unit testing
    """


    def test_0(self: TestValidateFullFct) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        )

        try:
            validate_full_fct(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateFullFct) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            validate_full_fct(inpt_1)
        except HasNonPositiveDiagonalElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_2(self: TestValidateFullFct) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [
                    [- 1.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        try:
            validate_full_fct(inpt_2)
        except HasNonPositiveDiagonalElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateNormLogit(TestCase):
    """
    'validate_norm_logit' unit testing
    """

    def test_0(self: TestValidateNormLogit) -> None:
        """
        """

        inpt_0 = tensor(
            [0.0, 0.0]
        )
        inpt_0 = log_softmax(inpt_0)

        try:
            validate_norm_logit(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateNormLogit) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        inpt_1 = log_softmax(inpt_1)

        try:
            validate_norm_logit(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: TestValidateNormLogit) -> None:
        """
        """

        inpt_2 = tensor(
            [1.0, 1.0]
        )

        try:
            validate_norm_logit(inpt_2)
        except IsNotNormalized:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullGaussianParametrization(TestCase):
    """
    'validate_full_gaussian_parametrization' unit testing
    """

    def setUp(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_fct_0 = tensor(
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
        self.norm_logit_0 = tensor(
            [0.0, 0.0]
        )
        self.norm_logit_0 = log_softmax(self.norm_logit_0)

    def test_00(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        try:
            validate_full_gaussian_parametrization(
                self.mean_0,
                self.full_fct_0
            )
        except:
            self.assertTrue(False)

    def test_000(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        try:
            validate_full_gaussian_parametrization(
                self.mean_0,
                self.full_fct_0,
                self.norm_logit_0
            )
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        mean_1 = tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        try:
            validate_full_gaussian_parametrization(
                mean_1,
                self.full_fct_0
            )
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        full_fct_1 = tensor(
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
            validate_full_gaussian_parametrization(
                self.mean_0,
                full_fct_1
            )
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        mean_2 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            validate_full_gaussian_parametrization(
                mean_2,
                self.full_fct_0
            )
        except HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_02(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        full_fct_2 = tensor(
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
            validate_full_gaussian_parametrization(
                self.mean_0,
                full_fct_2
            )
        except HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_002(self: TestValidateFullGaussianParametrization) -> None:
        """
        """

        norm_logit_2 = tensor(
            [0.0, 0.0, 0.0]
        )
        norm_logit_2 = log_softmax(norm_logit_2)

        try:
            validate_full_gaussian_parametrization(
                self.mean_0,
                self.full_fct_0,
                norm_logit_2
            )
        except HaveNonMatchingShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateFullGaussianParametrizations(TestCase):
    """
    'validate_full_gaussian_parametrizations' unit testing
    """

    def setUp(self: TestValidateFullGaussianParametrizations) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_fct_0 = tensor(
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

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_fct_1 = tensor(
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

    def test_01(self: TestValidateFullGaussianParametrizations) -> None:
        """
        """

        try:
            validate_full_gaussian_parametrizations(
                self.mean_0,
                self.full_fct_0,
                self.mean_1,
                self.full_fct_1
            )
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateFullGaussianParametrizations) -> None:
        """
        """

        try:
            validate_full_gaussian_parametrizations(
                self.mean_1,
                self.full_fct_1,
                self.mean_0,
                self.full_fct_0
            )
        except:
            self.assertTrue(False)

    def test_02(self: TestValidateFullGaussianParametrizations) -> None:
        """
        """

        mean_2 = tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        full_fct_2 = tensor(
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
            validate_full_gaussian_parametrizations(
                self.mean_0,
                self.full_fct_0,
                mean_2,
                full_fct_2
            )
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagFct(TestCase):
    """
    'validate_diag_fct' unit testing
    """

    def test_0(self: TestValidateDiagFct) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [1.0, 1.0],
            ]
        )

        try:
            validate_diag_fct(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateDiagFct) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [1.0],
            ]
        )

        try:
            validate_diag_fct(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: TestValidateDiagFct) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [0.0, 1.0],
            ]
        )

        try:
            validate_diag_fct(inpt_2)
        except HasNonPositiveElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: TestValidateDiagFct) -> None:
        """
        """

        inpt_3 = tensor(
            [
                [- 1.0, 1.0],
            ]
        )

        try:
            validate_diag_fct(inpt_3)
        except HasNonPositiveElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagGaussianParametrization(TestCase):
    """
    'validate_diag_gaussian_parametrization' unit testing
    """

    def setUp(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        )
        self.diag_fct_0 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.norm_logit_0 = tensor(
            [0.0, 0.0]
        )
        self.norm_logit_0 = log_softmax(self.norm_logit_0)

    def test_00(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        try:
            validate_diag_gaussian_parametrization(
                self.mean_0,
                self.diag_fct_0
            )
        except:
            self.assertTrue(False)

    def test_000(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        try:
            validate_diag_gaussian_parametrization(
                self.mean_0,
                self.diag_fct_0,
                self.norm_logit_0
            )
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        mean_1 = tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        try:
            validate_diag_gaussian_parametrization(
                mean_1,
                self.diag_fct_0
            )
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        diag_fct_1 = tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        try:
            validate_diag_gaussian_parametrization(
                self.mean_0,
                diag_fct_1
            )
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_20(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        mean_2 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            validate_diag_gaussian_parametrization(
                mean_2,
                self.diag_fct_0
            )
        except HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_02(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        diag_fct_2 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        try:
            validate_diag_gaussian_parametrization(
                self.mean_0,
                diag_fct_2
            )
        except HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_002(self: TestValidateDiagGaussianParametrization) -> None:
        """
        """

        norm_logit_2 = tensor(
            [0.0, 0.0, 0.0]
        )
        norm_logit_2 = log_softmax(norm_logit_2)

        try:
            validate_diag_gaussian_parametrization(
                self.mean_0,
                self.diag_fct_0,
                norm_logit_2
            )
        except HaveNonMatchingShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateDiagGaussianParametrizations(TestCase):
    """
    'validate_diag_gaussian_parametrizations' unit testing
    """

    def setUp(self: TestValidateDiagGaussianParametrizations) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.diag_fct_0 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.diag_fct_1 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

    def test_01(self: TestValidateDiagGaussianParametrizations) -> None:
        """
        """

        try:
            validate_diag_gaussian_parametrizations(
                self.mean_0,
                self.diag_fct_0,
                self.mean_1,
                self.diag_fct_1
            )
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateDiagGaussianParametrizations) -> None:
        """
        """

        try:
            validate_diag_gaussian_parametrizations(
                self.mean_1,
                self.diag_fct_1,
                self.mean_0,
                self.diag_fct_0
            )
        except:
            self.assertTrue(False)

    def test_02(self: TestValidateDiagGaussianParametrizations) -> None:
        """
        """

        mean_2 = tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        diag_fct_2 = tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        try:
            validate_diag_gaussian_parametrizations(
                self.mean_0,
                self.diag_fct_0,
                mean_2,
                diag_fct_2
            )
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoFct(TestCase):
    """
    'validate_is_fct' unit testing
    """

    def test_0(self: TestValidateIsoFct) -> None:
        """
        """

        inpt_0 = tensor(
            [1.0, 1.0]
        )

        try:
            validate_iso_fct(inpt_0)
        except:
            self.assertTrue(False)

    def test_1(self: TestValidateIsoFct) -> None:
        """
        """

        inpt_1 = tensor(
            [1.0]
        )

        try:
            validate_iso_fct(inpt_1)
        except:
            self.assertTrue(False)

    def test_2(self: TestValidateIsoFct) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        try:
            validate_iso_fct(inpt_2)
        except:
            self.assertTrue(False)

    def test_3(self: TestValidateIsoFct) -> None:
        """
        """

        inpt_3 = tensor(
            [0.0, 1.0]
        )

        try:
            validate_iso_fct(inpt_3)
        except HasNonPositiveElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_4(self: TestValidateIsoFct) -> None:
        """
        """

        inpt_4 = tensor(
            [- 1.0, 1.0]
        )

        try:
            validate_iso_fct(inpt_4)
        except HasNonPositiveElement:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoGaussianParametrization(TestCase):
    """
    'validate_iso_gaussian_parametrization' unit testing
    """

    def setUp(self: TestValidateIsoGaussianParametrization) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        )
        self.iso_fct_0 = tensor(
            [1.0, 1.0]
        )
        self.norm_logit_0 = tensor(
            [0.0, 0.0]
        )
        self.norm_logit_0 = log_softmax(self.norm_logit_0)

    def test_00(self: TestValidateIsoGaussianParametrization) -> None:
        """
        """

        try:
            validate_iso_gaussian_parametrization(
                self.mean_0,
                self.iso_fct_0
            )
        except:
            self.assertTrue(False)

    def test_000(self: TestValidateIsoGaussianParametrization) -> None:
        """
        """

        try:
            validate_iso_gaussian_parametrization(
                self.mean_0,
                self.iso_fct_0,
                self.norm_logit_0
            )
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateIsoGaussianParametrization) -> None:
        """
        """

        mean_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        try:
            validate_iso_gaussian_parametrization(
                mean_1,
                self.iso_fct_0
            )
        except HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_01(self: TestValidateIsoGaussianParametrization) -> None:
        """
        """

        iso_fct_1 = tensor(
            [1.0, 1.0, 1.0]
        )

        try:
            validate_iso_gaussian_parametrization(
                self.mean_0,
                iso_fct_1
            )
        except HaveNonBroadcastableShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_001(self: TestValidateIsoGaussianParametrization) -> None:
        """
        """

        norm_logit_1 = tensor(
            [0.0, 0.0, 0.0]
        )
        norm_logit_1 = log_softmax(norm_logit_1)

        try:
            validate_iso_gaussian_parametrization(
                self.mean_0,
                self.iso_fct_0,
                norm_logit_1
            )
        except HaveNonMatchingShapes:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestValidateIsoGaussianParametrizations(TestCase):
    """
    'validate_iso_gaussian_parametrizations' unit testing
    """

    def setUp(self: TestValidateIsoGaussianParametrizations) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_fct_0 = tensor(
            [1.0, 1.0]
        )

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_fct_1 = tensor(
            [1.0, 1.0]
        )

    def test_01(self: TestValidateIsoGaussianParametrizations) -> None:
        """
        """

        try:
            validate_iso_gaussian_parametrizations(
                self.mean_0,
                self.iso_fct_0,
                self.mean_1,
                self.iso_fct_1
            )
        except:
            self.assertTrue(False)

    def test_10(self: TestValidateIsoGaussianParametrizations) -> None:
        """
        """

        try:
            validate_iso_gaussian_parametrizations(
                self.mean_1,
                self.iso_fct_1,
                self.mean_0,
                self.iso_fct_0
            )
        except:
            self.assertTrue(False)

    def test_02(self: TestValidateIsoGaussianParametrizations) -> None:
        """
        """

        mean_2 = tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        iso_fct_2 = tensor(
            [1.0, 1.0]
        )

        try:
            validate_iso_gaussian_parametrizations(
                self.mean_0,
                self.iso_fct_0,
                mean_2,
                iso_fct_2
            )
        except HaveIncompatibleDims:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)
