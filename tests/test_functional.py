"""
Functional testing module

All function which do not have a an associated testing suite but are contained
on the 'functional' module are related to tested functions either by
composition or through trivial operations.
"""

from typing import Self

from unittest import TestCase


import torch


import gmml.functional as gF

from .utils import log_softmax


class Test_GetVBPairwiseDiffs(TestCase):
    """
    '_get_vb_pairwise_diffs' unit testing
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
                [1.0, 0.0],
            ]
        )
        self.inpt_2 = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
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

    def test_01(self: Self) -> None:
        """
        """

        true_out_01 = torch.tensor(
            [
                [
                    [- 1.0, 0.0],
                ],
            ]
        )

        out_01 = gF._get_vb_pairwise_diffs(self.inpt_0, self.inpt_1)

        if (out_01 != true_out_01).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        true_out_10 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                ],
            ]
        )

        out_10 = gF._get_vb_pairwise_diffs(self.inpt_1, self.inpt_0)

        if (out_10 != true_out_10).any():
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        """

        true_out_02 = torch.tensor(
            [
                [
                    [0.0, - 1.0],
                    [- 1.0, 0.0],
                ]
            ]
        )

        out_02 = gF._get_vb_pairwise_diffs(self.inpt_0, self.inpt_2)

        if (out_02 != true_out_02).any():
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        true_out_20 = torch.tensor(
            [
                [
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                ]
            ]
        )

        out_20 = gF._get_vb_pairwise_diffs(self.inpt_2, self.inpt_0)

        if (out_20 != true_out_20).any():
            self.assertTrue(False)

    def test_23(self: Self) -> None:
        """
        """

        true_out_23 = torch.tensor(
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

        out_23 = gF._get_vb_pairwise_diffs(self.inpt_2, self.inpt_3)

        if (out_23 != true_out_23).any():
            self.assertTrue(False)

    def test_32(self: Self) -> None:
        """
        """

        true_out_32 = torch.tensor(
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

        out_32 = gF._get_vb_pairwise_diffs(self.inpt_3, self.inpt_2)

        if (out_32 != true_out_32).any():
            self.assertTrue(False)

    def test_34(self: Self) -> None:
        """
        """

        true_out_34 = torch.tensor(
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

        out_34 = gF._get_vb_pairwise_diffs(self.inpt_3, self.inpt_4)

        if (out_34 != true_out_34).any():
            self.assertTrue(False)

    def test_43(self: Self) -> None:
        """
        """

        true_out_43 = torch.tensor(
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

        out_43 = gF._get_vb_pairwise_diffs(self.inpt_4, self.inpt_3)

        if (out_43 != true_out_43).any():
            self.assertTrue(False)


class Test_GetLogDets(TestCase):
    """
    '_get_log_dets' unit testing
    """

    def test_0(self: Self) -> None:
        """
        """

        full_factors_0 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        true_out_0 = torch.tensor(
            [0.0]
        )

        out_0 = gF._get_log_dets(full_factors_0)

        if (out_0 != true_out_0).any():
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        """

        full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            ]
        )

        true_out_1 = torch.tensor(
            [0.0]
        )

        out_1 = gF._get_log_dets(full_factors_1)

        if (out_1 != true_out_1).any():
            self.assertTrue(False)

    def test_2(self: Self) -> None:
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
            ]
        )

        true_out_2 = torch.tensor(
            [0.0, 0.0]
        )

        out_2 = gF._get_log_dets(full_factors_2)

        if (out_2 != true_out_2).any():
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        """

        full_factors_3 = torch.tensor(
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

        true_out_3 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        out_3 = gF._get_log_dets(full_factors_3)

        if (out_3 != true_out_3).any():
            self.assertTrue(False)


class Test_GetTriuMahalanobisDists(TestCase):
    """
    '_get_triu_mahalanobis_dists' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_factors_0 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        self.inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
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

    def test_000(self: Self) -> None:
        """
        """

        true_out_000 = torch.tensor(
            [
                [0.0],
            ]
        )

        out_000 = gF._get_triu_mahalanobis_dists(
            self.inpt_0,
            self.means_0,
            self.full_factors_0,
        )

        if (out_000 != true_out_000).any():
            self.assertTrue(False)

    def test_100(self: Self) -> None:
        """
        """

        true_out_100 = torch.tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        out_100 = gF._get_triu_mahalanobis_dists(
            self.inpt_1,
            self.means_0,
            self.full_factors_0,
        )

        if (out_100 != true_out_100).any():
            self.assertTrue(False)

    def test_010(self: Self) -> None:
        """
        """

        true_out_010 = torch.tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        out_010 = gF._get_triu_mahalanobis_dists(
            self.inpt_0,
            self.means_1,
            self.full_factors_0,
        )

        if (out_010 != true_out_010).any():
            self.assertTrue(False)

    def test_110(self: Self) -> None:
        """
        """

        true_out_110 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        out_110 = gF._get_triu_mahalanobis_dists(
            self.inpt_1,
            self.means_1,
            self.full_factors_0,
        )

        if (out_110 != true_out_110).any():
            self.assertTrue(False)

    def test_011(self: Self) -> None:
        """
        """

        true_out_011 = torch.tensor(
            [
                [0.0],
                [0.0],
            ]
        )

        out_011 = gF._get_triu_mahalanobis_dists(
            self.inpt_0,
            self.means_1,
            self.full_factors_1,
        )

        if (out_011 != true_out_011).any():
            self.assertTrue(False)

    def test_111(self: Self) -> None:
        """
        """

        true_out_111 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        out_111 = gF._get_triu_mahalanobis_dists(
            self.inpt_1,
            self.means_1,
            self.full_factors_1,
        )

        if (out_111 != true_out_111).any():
            self.assertTrue(False)


class Test_GetTrTriuTriu(TestCase):
    """
    '_get_tr_triu_triu' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.full_factors_0 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
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

    def test_00(self: Self) -> None:
        """
        """

        true_out_00 = torch.tensor(
            [
                [2.0],
            ]
        )

        out_00 = gF._get_tr_triu_triu(self.full_factors_0, self.full_factors_0)

        if (out_00 != true_out_00).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        true_out_10 = torch.tensor(
            [
                [2.0],
                [2.0],
            ]
        )

        out_10 = gF._get_tr_triu_triu(self.full_factors_1, self.full_factors_0)

        if (out_10 != true_out_10).any():
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        """

        true_out_01 = torch.tensor(
            [
                [2.0, 2.0],
            ]
        )

        out_01 = gF._get_tr_triu_triu(self.full_factors_0, self.full_factors_1)

        if (out_01 != true_out_01).any():
            self.assertTrue(False)

    def test_11(self: Self) -> None:
        """
        """

        true_out_11 = torch.tensor(
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        )

        out_11 = gF._get_tr_triu_triu(self.full_factors_1, self.full_factors_1)

        if (out_11 != true_out_11).any():
            self.assertTrue(False)


class Test_GetKLDivs(TestCase):
    """
    '_get_kl_divs' unit testing
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
        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )


    def test_01(self: Self) -> None:
        """
        """

        true_out_01_shape = [2, 1]

        out_01 = gF._get_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_1,
            self.full_factors_1
        )
        out_01_shape = list(out_01.shape)

        if out_01_shape != true_out_01_shape:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        true_out_10_shape = [1, 2]

        out_10 = gF._get_kl_divs(
            self.means_1,
            self.full_factors_1,
            self.means_0,
            self.full_factors_0
        )
        out_10_shape = list(out_10.shape)

        if out_10_shape != true_out_10_shape:
            self.assertTrue(False)


class TestGetLogJointProbs(TestCase):
    """
    'get_log_joint_probs' unit testing
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
        self.norm_logits_0 = torch.tensor(
            [0.0, 0.0]
        )
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        true_out_00_shape = [1, 2]

        out_00 = gF.get_log_joint_probs(
            inpt_0,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )
        out_00_shape = list(out_00.shape)

        if out_00_shape != true_out_00_shape:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        true_out_10_shape = [2, 2]

        out_10 = gF.get_log_joint_probs(
            inpt_1,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )
        out_10_shape = list(out_10.shape)

        if out_10_shape != true_out_10_shape:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        true_out_20_shape = [3, 2]

        out_20 = gF.get_log_joint_probs(
            inpt_2,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )
        out_20_shape = list(out_20.shape)

        if out_20_shape != true_out_20_shape:
            self.assertTrue(False)


class TestGetSamples(TestCase):
    """
    'get_samples' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_factors_0 = torch.tensor(
            [
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
        self.full_factors_1 =  torch.tensor(
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

    def test_00(self: Self) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_samples(
            self.means_0,
            self.full_factors_0
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_samples(
            self.means_0,
            self.full_factors_0,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_samples(
            self.means_0,
            self.full_factors_0,
            *true_out_batch_shape_2,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_samples(
            self.means_1,
            self.full_factors_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: Self) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_samples(
            self.means_1,
            self.full_factors_1,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: Self) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_samples(
            self.means_1,
            self.full_factors_1,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)


class Test_GetDiagKLDivs(TestCase):
    """
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
        self.diag_factors_0 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )
        self.diag_factors_1 = torch.tensor(
            [
                [1.0, 1.0],
            ]
        )

    def test_01(self: Self) -> None:
        """
        """

        full_out_01 = gF._get_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_1,
            self.full_factors_1,
        )

        diag_out_01 = gF.get_diag_kl_divs(
            self.means_0,
            self.diag_factors_0,
            self.means_1,
            self.diag_factors_1,
        )

        if (diag_out_01 != full_out_01).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        full_out_10 = gF._get_kl_divs(
            self.means_1,
            self.full_factors_1,
            self.means_0,
            self.full_factors_0,
        )

        diag_out_10 = gF.get_diag_kl_divs(
            self.means_1,
            self.diag_factors_1,
            self.means_0,
            self.diag_factors_0,
        )

        if (diag_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
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
            ]
        )
        diag_factors_2 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        full_out_02 = gF._get_kl_divs(
            self.means_0,
            self.full_factors_0,
            means_2,
            full_factors_2,
        )

        diag_out_02 = gF.get_diag_kl_divs(
            self.means_0,
            self.diag_factors_0,
            means_2,
            diag_factors_2,
        )

        if (diag_out_02 != full_out_02).any():
            self.assertTrue(False)

    def test_03(self: Self) -> None:
        """
        """

        means_3 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        full_factors_3 = torch.tensor(
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
        diag_factors_3 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        full_out_03 = gF._get_kl_divs(
            self.means_0,
            self.full_factors_0,
            means_3,
            full_factors_3,
        )

        diag_out_03 = gF.get_diag_kl_divs(
            self.means_0,
            self.diag_factors_0,
            means_3,
            diag_factors_3,
        )

        if (diag_out_03 != full_out_03).any():
            self.assertTrue(False)


class TestGetDiagLogJointProbs(TestCase):
    """
    'get_diag_log_joint_probs' unit testing
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

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        full_out_00 = gF.get_log_joint_probs(
            inpt_0,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )

        diag_out_00 = gF.get_diag_log_joint_probs(
            inpt_0,
            self.means_0,
            self.diag_factors_0,
            self.norm_logits_0
        )

        if (diag_out_00 != full_out_00).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_10 = gF.get_log_joint_probs(
            inpt_1,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )

        diag_out_10 = gF.get_diag_log_joint_probs(
            inpt_1,
            self.means_0,
            self.diag_factors_0,
            self.norm_logits_0
        )

        if (diag_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_20 = gF.get_log_joint_probs(
            inpt_2,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )

        diag_out_20 = gF.get_diag_log_joint_probs(
            inpt_2,
            self.means_0,
            self.diag_factors_0,
            self.norm_logits_0
        )

        if (diag_out_20 != full_out_20).any():
            self.assertTrue(False)


class TestGetDiagSamples(TestCase):
    """
    'get_diag_samples' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.diag_factors_0 = torch.tensor(
            [
                [1.0, 1.0],
            ]
        )

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.diag_factors_1 =  torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

    def test_00(self: Self) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_diag_samples(
            self.means_0,
            self.diag_factors_0
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_diag_samples(
            self.means_0,
            self.diag_factors_0,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_diag_samples(
            self.means_0,
            self.diag_factors_0,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_diag_samples(
            self.means_1,
            self.diag_factors_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: Self) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_diag_samples(
            self.means_1,
            self.diag_factors_1,
            *true_out_batch_shape_1,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: Self) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_diag_samples(
            self.means_1,
            self.diag_factors_1,
            *true_out_batch_shape_2,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)


class Test_GetIsoKLDivs(TestCase):
    """
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
        self.iso_factors_0 = torch.tensor(
            [1.0, 1.0]
        )

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )
        self.iso_factors_1 = torch.tensor(
            [1.0]
        )

    def test_01(self: Self) -> None:
        """
        """

        full_out_01 = gF._get_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_1,
            self.full_factors_1,
        )

        iso_out_01 = gF.get_iso_kl_divs(
            self.means_0,
            self.iso_factors_0,
            self.means_1,
            self.iso_factors_1,
        )

        if (iso_out_01 != full_out_01).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        full_out_10 = gF._get_kl_divs(
            self.means_1,
            self.full_factors_1,
            self.means_0,
            self.full_factors_0,
        )

        iso_out_10 = gF.get_iso_kl_divs(
            self.means_1,
            self.iso_factors_1,
            self.means_0,
            self.iso_factors_0,
        )

        if (iso_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        """

        means_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
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
            ]
        )
        iso_factors_2 = torch.tensor(
            [1.0, 1.0]
        )

        full_out_02 = gF._get_kl_divs(
            self.means_0,
            self.full_factors_0,
            means_2,
            full_factors_2,
        )

        iso_out_02 = gF.get_iso_kl_divs(
            self.means_0,
            self.iso_factors_0,
            means_2,
            iso_factors_2,
        )

        if (iso_out_02 != full_out_02).any():
            self.assertTrue(False)

    def test_03(self: Self) -> None:
        """
        """

        means_3 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        full_factors_3 = torch.tensor(
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
        iso_factors_3 = torch.tensor(
            [1.0, 1.0, 1.0]
        )

        full_out_03 = gF._get_kl_divs(
            self.means_0,
            self.full_factors_0,
            means_3,
            full_factors_3,
        )

        iso_out_03 = gF.get_iso_kl_divs(
            self.means_0,
            self.iso_factors_0,
            means_3,
            iso_factors_3,
        )

        if (iso_out_03 != full_out_03).any():
            self.assertTrue(False)


class TestGetIsoLogJointProbs(TestCase):
    """
    'get_iso_log_joint_probs' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
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
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )
        self.iso_factors_0 = torch.tensor(
            [1.0, 1.0, 1.0]
        )
        self.norm_logits_0 = torch.tensor(
            [0.0, 0.0, 0.0]
        )
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        full_out_00 = gF.get_log_joint_probs(
            inpt_0,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )

        iso_out_00 = gF.get_iso_log_joint_probs(
            inpt_0,
            self.means_0,
            self.iso_factors_0,
            self.norm_logits_0
        )

        if (iso_out_00 != full_out_00).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_10 = gF.get_log_joint_probs(
            inpt_1,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )

        iso_out_10 = gF.get_iso_log_joint_probs(
            inpt_1,
            self.means_0,
            self.iso_factors_0,
            self.norm_logits_0
        )

        if (iso_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_20 = gF.get_log_joint_probs(
            inpt_2,
            self.means_0,
            self.full_factors_0,
            self.norm_logits_0
        )

        iso_out_20 = gF.get_iso_log_joint_probs(
            inpt_2,
            self.means_0,
            self.iso_factors_0,
            self.norm_logits_0
        )

        if (iso_out_20 != full_out_20).any():
            self.assertTrue(False)


class TestGetIsoSamples(TestCase):
    """
    'get_iso_samples' unit testing
    """

    def setUp(self: Self) -> None:
        """
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.iso_factors_0 = torch.tensor(
            [1.0]
        )

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_factors_1 =  torch.tensor(
            [1.0, 1.0]
        )

    def test_00(self: Self) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_iso_samples(
            self.means_0,
            self.iso_factors_0
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_iso_samples(
            self.means_0,
            self.iso_factors_0,
            *true_out_batch_shape_1,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_iso_samples(
            self.means_0,
            self.iso_factors_0,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_iso_samples(
            self.means_1,
            self.iso_factors_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: Self) -> None:
        """
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_iso_samples(
            self.means_1,
            self.iso_factors_1,
            *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: Self) -> None:
        """
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_iso_samples(
            self.means_1,
            self.iso_factors_1,
            *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)
