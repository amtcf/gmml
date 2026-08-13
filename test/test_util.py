"""
Utilities testing module
"""

from __future__ import annotations

from unittest import TestCase

from torch import tensor, float64

from gmml.util import sample_standard_multivariate_normal
from gmml.util import set_softplus_diag


class TestSampleStandardMultivariateNormal(TestCase):
    """
    'sample_standard_multivariate_normal' unit testing
    """

    def test_0(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        true_out_0_shape = [2]
        out_0 = sample_standard_multivariate_normal(*true_out_0_shape)
        out_0_shape = list(out_0.shape)

        if out_0_shape != true_out_0_shape:
            self.assertTrue(False)

    def test_1(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        true_out_1_shape = [2, 2]
        out_1 = sample_standard_multivariate_normal(*true_out_1_shape)
        out_1_shape = list(out_1.shape)

        if out_1_shape != true_out_1_shape:
            self.assertTrue(False)

    def test_2(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        try:
            sample_standard_multivariate_normal()
        except ValueError:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_3(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        try:
            sample_standard_multivariate_normal(0)
        except ValueError:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_4(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        try:
            sample_standard_multivariate_normal(-1)
        except ValueError:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_5(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        try:
            sample_standard_multivariate_normal(-1, 2)
        except ValueError:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_6(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        out_6 = sample_standard_multivariate_normal(2, device="cpu")

        if out_6.device.type != "cpu":
            self.assertTrue(False)

    def test_7(self: TestSampleStandardMultivariateNormal) -> None:
        """
        """

        out_7 = sample_standard_multivariate_normal(2, dtype=float64)

        if out_7.dtype != float64:
            self.assertTrue(False)


class TestSetSoftplusDiag(TestCase):
    """
    'set_softplus_diag' unit testing
    """

    def setUp(self: TestSetSoftplusDiag) -> None:
        """
        """

        self.mask = tensor(
            [0.0, 0.0]
        )

    def test_0(self: TestSetSoftplusDiag) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        out_0 = set_softplus_diag(inpt_0)
        out_0_diag = out_0.diagonal(dim1=-2, dim2=-1)

        if (out_0_diag <= 0.0).any():
            self.assertTrue(False)

        diff = out_0 - inpt_0
        masked_diff = diff.diagonal_scatter(self.mask, dim1=-2, dim2=-1)
        if (masked_diff != 0.0).any():
            self.assertTrue(False)

    def test_1(self: TestSetSoftplusDiag) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        out_1 = set_softplus_diag(inpt_1)
        out_1_diag = out_1.diagonal(dim1=-2, dim2=-1)

        if (out_1_diag <= 0.0).any():
            self.assertTrue(False)

        diff = out_1 - inpt_1
        masked_diff = diff.diagonal_scatter(self.mask, dim1=-2, dim2=-1)
        if (masked_diff != 0.0).any():
            self.assertTrue(False)

    def test_2(self: TestSetSoftplusDiag) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [- 1.0,   0.0],
                [  0.0, - 1.0],
            ]
        )

        out_2 = set_softplus_diag(inpt_2)
        out_2_diag = out_2.diagonal(dim1=-2, dim2=-1)

        if (out_2_diag <= 0.0).any():
            self.assertTrue(False)

        diff = out_2 - inpt_2
        masked_diff = diff.diagonal_scatter(self.mask, dim1=-2, dim2=-1)
        if (masked_diff != 0.0).any():
            self.assertTrue(False)

    def test_3(self: TestSetSoftplusDiag) -> None:
        """
        """

        inpt_3 = tensor(
            [0.0, 0.0]
        )

        try:
            set_softplus_diag(inpt_3)
        except ValueError:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)

    def test_4(self: TestSetSoftplusDiag) -> None:
        """
        """

        inpt_4 = tensor(0.0)

        try:
            set_softplus_diag(inpt_4)
        except ValueError:
            pass
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(False)
