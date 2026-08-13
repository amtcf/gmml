"""
Functional testing module

All function which do not have a an associated testing suite but are contained
on the 'functional' module are related to tested functions either by
composition or through trivial operations.
"""

from __future__ import annotations

from unittest import TestCase

from torch import tensor

from gmml.functional import _get_vb_pairwise_diff
from gmml.functional import _get_log_det
from gmml.functional import _get_triu_mahalanobis_dist
from gmml.functional import _get_tr_triu_triu
from gmml.functional import _get_kl_div
from gmml.functional import get_log_joint_prob
from gmml.functional import get_sample
from gmml.functional import get_diag_kl_div
from gmml.functional import get_diag_log_joint_prob
from gmml.functional import get_diag_sample
from gmml.functional import get_iso_kl_div
from gmml.functional import get_iso_log_joint_prob
from gmml.functional import get_iso_sample

from .util import log_softmax


class Test_GetVBPairwiseDiff(TestCase):
    """
    '_get_vb_pairwise_diff' unit testing
    """

    def setUp(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        self.inpt_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.inpt_1 = tensor(
            [
                [1.0, 0.0],
            ]
        )
        self.inpt_2 = tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
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
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
            ]
        )

    def test_01(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_01 = tensor(
            [
                [
                    [- 1.0, 0.0],
                ],
            ]
        )

        out_01 = _get_vb_pairwise_diff(self.inpt_0, self.inpt_1)

        if (out_01 != true_out_01).any():
            self.assertTrue(False)

    def test_10(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_10 = tensor(
            [
                [
                    [1.0, 0.0],
                ],
            ]
        )

        out_10 = _get_vb_pairwise_diff(self.inpt_1, self.inpt_0)

        if (out_10 != true_out_10).any():
            self.assertTrue(False)

    def test_02(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_02 = tensor(
            [
                [
                    [0.0, - 1.0],
                    [- 1.0, 0.0],
                ]
            ]
        )

        out_02 = _get_vb_pairwise_diff(self.inpt_0, self.inpt_2)

        if (out_02 != true_out_02).any():
            self.assertTrue(False)

    def test_20(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_20 = tensor(
            [
                [
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                ]
            ]
        )

        out_20 = _get_vb_pairwise_diff(self.inpt_2, self.inpt_0)

        if (out_20 != true_out_20).any():
            self.assertTrue(False)

    def test_23(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_23 = tensor(
            [
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                ]
            ]
        )

        out_23 = _get_vb_pairwise_diff(self.inpt_2, self.inpt_3)

        if (out_23 != true_out_23).any():
            self.assertTrue(False)

    def test_32(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_32 = tensor(
            [
                [
                    [0.0, - 1.0],
                    [- 1.0, 0.0],
                ],
                [
                    [0.0, - 1.0],
                    [- 1.0, 0.0],
                ]
            ]
        )

        out_32 = _get_vb_pairwise_diff(self.inpt_3, self.inpt_2)

        if (out_32 != true_out_32).any():
            self.assertTrue(False)

    def test_34(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_34 = tensor(
            [
                [

                    [
                        [  0.0,   0.0],
                        [  0.0,   0.0],
                    ],
                    [
                        [- 1.0,   0.0],
                        [  0.0, - 1.0],
                    ],
                    [
                        [  0.0, - 1.0],
                        [- 1.0,   0.0],
                    ],
                ],
                [

                    [
                        [  0.0,   0.0],
                        [  0.0,   0.0],
                    ],
                    [
                        [- 1.0,   0.0],
                        [  0.0, - 1.0],
                    ],
                    [
                        [  0.0, - 1.0],
                        [- 1.0,   0.0],
                    ],
                ],
            ]
        )

        out_34 = _get_vb_pairwise_diff(self.inpt_3, self.inpt_4)

        if (out_34 != true_out_34).any():
            self.assertTrue(False)

    def test_43(self: Test_GetVBPairwiseDiff) -> None:
        """
        """

        true_out_43 = tensor(
            [
                [
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                ],
                [
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                    ],
                    [
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ],
                ],
                [
                    [
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ],
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                    ],
                ],
            ]

        )

        out_43 = _get_vb_pairwise_diff(self.inpt_4, self.inpt_3)

        if (out_43 != true_out_43).any():
            self.assertTrue(False)


class Test_GetLogDet(TestCase):
    """
    '_get_log_det' unit testing
    """

    def test_0(self: Test_GetLogDet) -> None:
        """
        """

        full_fct_0 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        true_out_0 = tensor(
            [0.0]
        )

        out_0 = _get_log_det(full_fct_0)

        if (out_0 != true_out_0).any():
            self.assertTrue(False)

    def test_1(self: Test_GetLogDet) -> None:
        """
        """

        full_fct_1 = tensor(
            [
                [
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            ]
        )

        true_out_1 = tensor(
            [0.0]
        )

        out_1 = _get_log_det(full_fct_1)

        if (out_1 != true_out_1).any():
            self.assertTrue(False)

    def test_2(self: Test_GetLogDet) -> None:
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
            ]
        )

        true_out_2 = tensor(
            [0.0, 0.0]
        )

        out_2 = _get_log_det(full_fct_2)

        if (out_2 != true_out_2).any():
            self.assertTrue(False)

    def test_3(self: Test_GetLogDet) -> None:
        """
        """

        full_fct_3 = tensor(
            [
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                ],
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                ],
            ]
        )

        true_out_3 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        out_3 = _get_log_det(full_fct_3)

        if (out_3 != true_out_3).any():
            self.assertTrue(False)


class Test_GetTriuMahalanobisDist(TestCase):
    """
    '_get_triu_mahalanobis_dist' unit testing
    """

    def setUp(self: Test_GetTriuMahalanobisDist) -> None:
        """
        """

        self.inpt_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.mean_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_fct_0 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        self.inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
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
                ],
            ]
        )

    def test_000(self: Test_GetTriuMahalanobisDist) -> None:
        """
        """

        true_out_000 = tensor(
            [
                [0.0],
            ]
        )

        out_000 = _get_triu_mahalanobis_dist(
            self.inpt_0,
            self.mean_0,
            self.full_fct_0,
        )

        if (out_000 != true_out_000).any():
            self.assertTrue(False)

    def test_100(self: Test_GetTriuMahalanobisDist) -> None:
        """
        """

        true_out_100 = tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        out_100 = _get_triu_mahalanobis_dist(
            self.inpt_1,
            self.mean_0,
            self.full_fct_0,
        )

        if (out_100 != true_out_100).any():
            self.assertTrue(False)

    def test_010(self: Test_GetTriuMahalanobisDist) -> None:
        """
        """

        true_out_010 = tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        out_010 = _get_triu_mahalanobis_dist(
            self.inpt_0,
            self.mean_1,
            self.full_fct_0,
        )

        if (out_010 != true_out_010).any():
            self.assertTrue(False)

    def test_110(self: Test_GetTriuMahalanobisDist) -> None:
        """
        """

        true_out_110 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        out_110 = _get_triu_mahalanobis_dist(
            self.inpt_1,
            self.mean_1,
            self.full_fct_0,
        )

        if (out_110 != true_out_110).any():
            self.assertTrue(False)

    def test_011(self: Test_GetTriuMahalanobisDist) -> None:
        """
        """

        true_out_011 = tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        out_011 = _get_triu_mahalanobis_dist(
            self.inpt_0,
            self.mean_1,
            self.full_fct_1,
        )

        if (out_011 != true_out_011).any():
            self.assertTrue(False)

    def test_111(self: Test_GetTriuMahalanobisDist) -> None:
        """
        """

        true_out_111 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        out_111 = _get_triu_mahalanobis_dist(
            self.inpt_1,
            self.mean_1,
            self.full_fct_1,
        )

        if (out_111 != true_out_111).any():
            self.assertTrue(False)


class Test_GetTrTriuTriu(TestCase):
    """
    '_get_tr_triu_triu' unit testing
    """

    def setUp(self: Test_GetTrTriuTriu) -> None:
        """
        """

        self.full_fct_0 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
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
                ],
            ]
        )

    def test_00(self: Test_GetTrTriuTriu) -> None:
        """
        """

        true_out_00 = tensor(
            [
                [2.0],
            ]
        )

        out_00 = _get_tr_triu_triu(self.full_fct_0, self.full_fct_0)

        if (out_00 != true_out_00).any():
            self.assertTrue(False)

    def test_10(self: Test_GetTrTriuTriu) -> None:
        """
        """

        true_out_10 = tensor(
            [
                [2.0],
                [2.0],
            ]
        )

        out_10 = _get_tr_triu_triu(self.full_fct_1, self.full_fct_0)

        if (out_10 != true_out_10).any():
            self.assertTrue(False)

    def test_01(self: Test_GetTrTriuTriu) -> None:
        """
        """

        true_out_01 = tensor(
            [
                [2.0, 2.0],
            ]
        )

        out_01 = _get_tr_triu_triu(self.full_fct_0, self.full_fct_1)

        if (out_01 != true_out_01).any():
            self.assertTrue(False)

    def test_11(self: Test_GetTrTriuTriu) -> None:
        """
        """

        true_out_11 = tensor(
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        )

        out_11 = _get_tr_triu_triu(self.full_fct_1, self.full_fct_1)

        if (out_11 != true_out_11).any():
            self.assertTrue(False)


class Test_GetKLDiv(TestCase):
    """
    '_get_kl_div' unit testing
    """

    def setUp(self: Test_GetKLDiv) -> None:
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
        self.mean_1 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_fct_1 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )


    def test_01(self: Test_GetKLDiv) -> None:
        """
        """

        true_out_01_shape = [2, 1]

        out_01 = _get_kl_div(
            self.mean_0,
            self.full_fct_0,
            self.mean_1,
            self.full_fct_1
        )
        out_01_shape = list(out_01.shape)

        if out_01_shape != true_out_01_shape:
            self.assertTrue(False)

    def test_10(self: Test_GetKLDiv) -> None:
        """
        """

        true_out_10_shape = [1, 2]

        out_10 = _get_kl_div(
            self.mean_1,
            self.full_fct_1,
            self.mean_0,
            self.full_fct_0
        )
        out_10_shape = list(out_10.shape)

        if out_10_shape != true_out_10_shape:
            self.assertTrue(False)


class TestGetLogJointProb(TestCase):
    """
    'get_log_joint_prob' unit testing
    """

    def setUp(self: TestGetLogJointProb) -> None:
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
        self.norm_logit_0 = tensor(
            [0.0, 0.0]
        )
        self.norm_logit_0 = log_softmax(self.norm_logit_0)

    def test_00(self: TestGetLogJointProb) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        true_out_00_shape = [1, 2]

        out_00 = get_log_joint_prob(
            inpt_0,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )
        out_00_shape = list(out_00.shape)

        if out_00_shape != true_out_00_shape:
            self.assertTrue(False)

    def test_10(self: TestGetLogJointProb) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        true_out_10_shape = [2, 2]

        out_10 = get_log_joint_prob(
            inpt_1,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )
        out_10_shape = list(out_10.shape)

        if out_10_shape != true_out_10_shape:
            self.assertTrue(False)

    def test_20(self: TestGetLogJointProb) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        true_out_20_shape = [3, 2]

        out_20 = get_log_joint_prob(
            inpt_2,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )
        out_20_shape = list(out_20.shape)

        if out_20_shape != true_out_20_shape:
            self.assertTrue(False)


class TestGetSample(TestCase):
    """
    'get_sample' unit testing
    """

    def setUp(self: TestGetSample) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_fct_0 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_fct_1 =  tensor(
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

    def test_00(self: TestGetSample) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = get_sample(
            self.mean_0,
            self.full_fct_0
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: TestGetSample) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = get_sample(
            self.mean_0,
            self.full_fct_0,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: TestGetSample) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = get_sample(
            self.mean_0,
            self.full_fct_0,
            *true_out_batch_shape_2,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: TestGetSample) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = get_sample(
            self.mean_1,
            self.full_fct_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: TestGetSample) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = get_sample(
            self.mean_1,
            self.full_fct_1,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: TestGetSample) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = get_sample(
            self.mean_1,
            self.full_fct_1,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)


class Test_GetDiagKLDiv(TestCase):
    """
    """

    def setUp(self: Test_GetDiagKLDiv) -> None:
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
        self.diag_fct_0 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_fct_1 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )
        self.diag_fct_1 = tensor(
            [
                [1.0, 1.0],
            ]
        )

    def test_01(self: Test_GetDiagKLDiv) -> None:
        """
        """

        full_out_01 = _get_kl_div(
            self.mean_0,
            self.full_fct_0,
            self.mean_1,
            self.full_fct_1,
        )

        diag_out_01 = get_diag_kl_div(
            self.mean_0,
            self.diag_fct_0,
            self.mean_1,
            self.diag_fct_1,
        )

        if (diag_out_01 != full_out_01).any():
            self.assertTrue(False)

    def test_10(self: Test_GetDiagKLDiv) -> None:
        """
        """

        full_out_10 = _get_kl_div(
            self.mean_1,
            self.full_fct_1,
            self.mean_0,
            self.full_fct_0,
        )

        diag_out_10 = get_diag_kl_div(
            self.mean_1,
            self.diag_fct_1,
            self.mean_0,
            self.diag_fct_0,
        )

        if (diag_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_02(self: Test_GetDiagKLDiv) -> None:
        """
        """

        mean_2 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
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
            ]
        )
        diag_fct_2 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        full_out_02 = _get_kl_div(
            self.mean_0,
            self.full_fct_0,
            mean_2,
            full_fct_2,
        )

        diag_out_02 = get_diag_kl_div(
            self.mean_0,
            self.diag_fct_0,
            mean_2,
            diag_fct_2,
        )

        if (diag_out_02 != full_out_02).any():
            self.assertTrue(False)

    def test_03(self: Test_GetDiagKLDiv) -> None:
        """
        """

        mean_3 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        full_fct_3 = tensor(
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
        diag_fct_3 = tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        full_out_03 = _get_kl_div(
            self.mean_0,
            self.full_fct_0,
            mean_3,
            full_fct_3,
        )

        diag_out_03 = get_diag_kl_div(
            self.mean_0,
            self.diag_fct_0,
            mean_3,
            diag_fct_3,
        )

        if (diag_out_03 != full_out_03).any():
            self.assertTrue(False)


class TestGetDiagLogJointProb(TestCase):
    """
    'get_diag_log_joint_prob' unit testing
    """

    def setUp(self: TestGetDiagLogJointProb) -> None:
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

    def test_00(self: TestGetDiagLogJointProb) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )

        full_out_00 = get_log_joint_prob(
            inpt_0,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )

        diag_out_00 = get_diag_log_joint_prob(
            inpt_0,
            self.mean_0,
            self.diag_fct_0,
            self.norm_logit_0
        )

        if (diag_out_00 != full_out_00).any():
            self.assertTrue(False)

    def test_10(self: TestGetDiagLogJointProb) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_10 = get_log_joint_prob(
            inpt_1,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )

        diag_out_10 = get_diag_log_joint_prob(
            inpt_1,
            self.mean_0,
            self.diag_fct_0,
            self.norm_logit_0
        )

        if (diag_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_20(self: TestGetDiagLogJointProb) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_20 = get_log_joint_prob(
            inpt_2,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )

        diag_out_20 = get_diag_log_joint_prob(
            inpt_2,
            self.mean_0,
            self.diag_fct_0,
            self.norm_logit_0
        )

        if (diag_out_20 != full_out_20).any():
            self.assertTrue(False)


class TestGetDiagSample(TestCase):
    """
    'get_diag_sample' unit testing
    """

    def setUp(self: TestGetDiagSample) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.diag_fct_0 = tensor(
            [
                [1.0, 1.0],
            ]
        )

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.diag_fct_1 =  tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

    def test_00(self: TestGetDiagSample) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = get_diag_sample(
            self.mean_0,
            self.diag_fct_0
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: TestGetDiagSample) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = get_diag_sample(
            self.mean_0,
            self.diag_fct_0,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: TestGetDiagSample) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = get_diag_sample(
            self.mean_0,
            self.diag_fct_0,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: TestGetDiagSample) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = get_diag_sample(
            self.mean_1,
            self.diag_fct_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: TestGetDiagSample) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = get_diag_sample(
            self.mean_1,
            self.diag_fct_1,
            *true_out_batch_shape_1,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: TestGetDiagSample) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = get_diag_sample(
            self.mean_1,
            self.diag_fct_1,
            *true_out_batch_shape_2,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)


class Test_GetIsoKLDiv(TestCase):
    """
    """

    def setUp(self: Test_GetIsoKLDiv) -> None:
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
        self.iso_fct_0 = tensor(
            [1.0, 1.0]
        )

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_fct_1 = tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )
        self.iso_fct_1 = tensor(
            [1.0]
        )

    def test_01(self: Test_GetIsoKLDiv) -> None:
        """
        """

        full_out_01 = _get_kl_div(
            self.mean_0,
            self.full_fct_0,
            self.mean_1,
            self.full_fct_1,
        )

        iso_out_01 = get_iso_kl_div(
            self.mean_0,
            self.iso_fct_0,
            self.mean_1,
            self.iso_fct_1,
        )

        if (iso_out_01 != full_out_01).any():
            self.assertTrue(False)

    def test_10(self: Test_GetIsoKLDiv) -> None:
        """
        """

        full_out_10 = _get_kl_div(
            self.mean_1,
            self.full_fct_1,
            self.mean_0,
            self.full_fct_0,
        )

        iso_out_10 = get_iso_kl_div(
            self.mean_1,
            self.iso_fct_1,
            self.mean_0,
            self.iso_fct_0,
        )

        if (iso_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_02(self: Test_GetIsoKLDiv) -> None:
        """
        """

        mean_2 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
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
            ]
        )
        iso_fct_2 = tensor(
            [1.0, 1.0]
        )

        full_out_02 = _get_kl_div(
            self.mean_0,
            self.full_fct_0,
            mean_2,
            full_fct_2,
        )

        iso_out_02 = get_iso_kl_div(
            self.mean_0,
            self.iso_fct_0,
            mean_2,
            iso_fct_2,
        )

        if (iso_out_02 != full_out_02).any():
            self.assertTrue(False)

    def test_03(self: Test_GetIsoKLDiv) -> None:
        """
        """

        mean_3 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        full_fct_3 = tensor(
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
        iso_fct_3 = tensor(
            [1.0, 1.0, 1.0]
        )

        full_out_03 = _get_kl_div(
            self.mean_0,
            self.full_fct_0,
            mean_3,
            full_fct_3,
        )

        iso_out_03 = get_iso_kl_div(
            self.mean_0,
            self.iso_fct_0,
            mean_3,
            iso_fct_3,
        )

        if (iso_out_03 != full_out_03).any():
            self.assertTrue(False)


class TestGetIsoLogJointProb(TestCase):
    """
    'get_iso_log_joint_prob' unit testing
    """

    def setUp(self: TestGetIsoLogJointProb) -> None:
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
        self.iso_fct_0 = tensor(
            [1.0, 1.0]
        )
        self.norm_logit_0 = tensor(
            [0.0, 0.0]
        )
        self.norm_logit_0 = log_softmax(self.norm_logit_0)

    def test_00(self: TestGetIsoLogJointProb) -> None:
        """
        """

        inpt_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )

        full_out_00 = get_log_joint_prob(
            inpt_0,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )

        iso_out_00 = get_iso_log_joint_prob(
            inpt_0,
            self.mean_0,
            self.iso_fct_0,
            self.norm_logit_0
        )

        if (iso_out_00 != full_out_00).any():
            self.assertTrue(False)

    def test_10(self: TestGetIsoLogJointProb) -> None:
        """
        """

        inpt_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_10 = get_log_joint_prob(
            inpt_1,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )

        iso_out_10 = get_iso_log_joint_prob(
            inpt_1,
            self.mean_0,
            self.iso_fct_0,
            self.norm_logit_0
        )

        if (iso_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_20(self: TestGetIsoLogJointProb) -> None:
        """
        """

        inpt_2 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_20 = get_log_joint_prob(
            inpt_2,
            self.mean_0,
            self.full_fct_0,
            self.norm_logit_0
        )

        iso_out_20 = get_iso_log_joint_prob(
            inpt_2,
            self.mean_0,
            self.iso_fct_0,
            self.norm_logit_0
        )

        if (iso_out_20 != full_out_20).any():
            self.assertTrue(False)


class TestGetIsoSample(TestCase):
    """
    'get_iso_sample' unit testing
    """

    def setUp(self: TestGetIsoSample) -> None:
        """
        """

        self.mean_0 = tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.iso_fct_0 = tensor(
            [1.0]
        )

        self.mean_1 = tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_fct_1 =  tensor(
            [1.0, 1.0]
        )

    def test_00(self: TestGetIsoSample) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = get_iso_sample(
            self.mean_0,
            self.iso_fct_0
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: TestGetIsoSample) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = get_iso_sample(
            self.mean_0,
            self.iso_fct_0,
            *true_out_batch_shape_1,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: TestGetIsoSample) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = get_iso_sample(
            self.mean_0,
            self.iso_fct_0,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: TestGetIsoSample) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = get_iso_sample(
            self.mean_1,
            self.iso_fct_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: TestGetIsoSample) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = get_iso_sample(
            self.mean_1,
            self.iso_fct_1,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: TestGetIsoSample) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = get_iso_sample(
            self.mean_1,
            self.iso_fct_1,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)
