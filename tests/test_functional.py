"""
Functional testing module

All function which do not have a an associated testing suite but are contained
on the 'functional' module are related to tested functions either by
composition or through trivial operations.
"""

import math
from typing import Self
from unittest import TestCase

import torch
from torch.distributions import MultivariateNormal

import gmml.functional as gF
import gmml.validation as val

from .utils import get_random_full_factors, get_ref_kl_divs, log_softmax


class Test_GetVBPairwiseDiffs(TestCase):
    """
    '_get_vb_pairwise_diffs' unit testing
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
        Checks the output of '_get_vb_pairwise_diffs' on case 01.
        """

        true_out_01 = torch.tensor(
            [
                [
                    [-1.0, 0.0],
                ],
            ]
        )

        out_01 = gF._get_vb_pairwise_diffs(self.inpt_0, self.inpt_1)

        if (out_01 != true_out_01).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks the output of '_get_vb_pairwise_diffs' on case 10.
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
        Checks the output of '_get_vb_pairwise_diffs' on case 02.
        """

        true_out_02 = torch.tensor(
            [
                [
                    [0.0, -1.0],
                    [-1.0, 0.0],
                ]
            ]
        )

        out_02 = gF._get_vb_pairwise_diffs(self.inpt_0, self.inpt_2)

        if (out_02 != true_out_02).any():
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks the output of '_get_vb_pairwise_diffs' on case 20.
        """

        true_out_20 = torch.tensor(
            [
                [
                    [0.0, 1.0],
                ],
                [
                    [1.0, 0.0],
                ],
            ]
        )

        out_20 = gF._get_vb_pairwise_diffs(self.inpt_2, self.inpt_0)

        if (out_20 != true_out_20).any():
            self.assertTrue(False)

    def test_23(self: Self) -> None:
        """
        Checks the output of '_get_vb_pairwise_diffs' on case 23.
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
                ],
            ]
        )

        out_23 = gF._get_vb_pairwise_diffs(self.inpt_2, self.inpt_3)

        if (out_23 != true_out_23).any():
            self.assertTrue(False)

    def test_32(self: Self) -> None:
        """
        Checks the output of '_get_vb_pairwise_diffs' on case 32.
        """

        true_out_32 = torch.tensor(
            [
                [
                    [0.0, -1.0],
                    [-1.0, 0.0],
                ],
                [
                    [0.0, -1.0],
                    [-1.0, 0.0],
                ],
            ]
        )

        out_32 = gF._get_vb_pairwise_diffs(self.inpt_3, self.inpt_2)

        if (out_32 != true_out_32).any():
            self.assertTrue(False)

    def test_34(self: Self) -> None:
        """
        Checks the output of '_get_vb_pairwise_diffs' on case 34.
        """

        true_out_34 = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    [
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ],
                    [
                        [0.0, -1.0],
                        [-1.0, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    [
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ],
                    [
                        [0.0, -1.0],
                        [-1.0, 0.0],
                    ],
                ],
            ]
        )

        out_34 = gF._get_vb_pairwise_diffs(self.inpt_3, self.inpt_4)

        if (out_34 != true_out_34).any():
            self.assertTrue(False)

    def test_43(self: Self) -> None:
        """
        Checks the output of '_get_vb_pairwise_diffs' on case 43.
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


class Test_GetFullLogDets(TestCase):
    """
    '_get_full_log_dets' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks the output of '_get_full_log_dets' on case 0.
        """

        full_factors_0 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        )

        true_out_0 = torch.tensor([0.0])

        out_0 = gF._get_full_log_dets(full_factors_0)

        if (out_0 != true_out_0).any():
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks the output of '_get_full_log_dets' on case 1.
        """

        full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            ]
        )

        true_out_1 = torch.tensor([0.0])

        out_1 = gF._get_full_log_dets(full_factors_1)

        if (out_1 != true_out_1).any():
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks the output of '_get_full_log_dets' on case 2.
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

        true_out_2 = torch.tensor([0.0, 0.0])

        out_2 = gF._get_full_log_dets(full_factors_2)

        if (out_2 != true_out_2).any():
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks the output of '_get_full_log_dets' on case 3.
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

        out_3 = gF._get_full_log_dets(full_factors_3)

        if (out_3 != true_out_3).any():
            self.assertTrue(False)


class Test_GetTriuMahalanobisDists(TestCase):
    """
    '_get_triu_mahalanobis_dists' unit testing
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
        Checks the output of '_get_triu_mahalanobis_dists' on case 000.
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
        Checks the output of '_get_triu_mahalanobis_dists' on case 100.
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
        Checks the output of '_get_triu_mahalanobis_dists' on case 010.
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
        Checks the output of '_get_triu_mahalanobis_dists' on case 110.
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
        Checks the output of '_get_triu_mahalanobis_dists' on case 011.
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
        Checks the output of '_get_triu_mahalanobis_dists' on case 111.
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
        Builds the shared test fixtures.
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
        Checks the output of '_get_tr_triu_triu' on case 00.
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
        Checks the output of '_get_tr_triu_triu' on case 10.
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
        Checks the output of '_get_tr_triu_triu' on case 01.
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
        Checks the output of '_get_tr_triu_triu' on case 11.
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


class Test_GetFullKLDivs(TestCase):
    """
    '_get_full_kl_divs' unit testing
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
        Checks the output shape of '_get_full_kl_divs' on case 01.
        """

        true_out_01_shape = [2, 1]

        out_01 = gF._get_full_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_1,
            self.full_factors_1,
        )
        out_01_shape = list(out_01.shape)

        if out_01_shape != true_out_01_shape:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks the output shape of '_get_full_kl_divs' on case 10.
        """

        true_out_10_shape = [1, 2]

        out_10 = gF._get_full_kl_divs(
            self.means_1,
            self.full_factors_1,
            self.means_0,
            self.full_factors_0,
        )
        out_10_shape = list(out_10.shape)

        if out_10_shape != true_out_10_shape:
            self.assertTrue(False)

    def test_23(self: Self) -> None:
        """
        Checks that '_get_full_kl_divs' matches torch.distributions for
        random, non-identity Cholesky factors.
        """

        torch.manual_seed(0)

        means_2 = torch.randn(2, 3)
        full_factors_2 = get_random_full_factors(2, 3)
        means_3 = torch.randn(4, 3)
        full_factors_3 = get_random_full_factors(4, 3)

        out_23 = gF._get_full_kl_divs(
            means_2, full_factors_2, means_3, full_factors_3
        )
        true_out_23 = get_ref_kl_divs(
            means_2, full_factors_2, means_3, full_factors_3
        )

        if not torch.allclose(out_23, true_out_23, atol=1e-4):
            self.assertTrue(False)


class TestGetFullLogJointProbs(TestCase):
    """
    'get_full_log_joint_probs' unit testing
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
        Checks the output shape of 'get_full_log_joint_probs' on case 00.
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        true_out_00_shape = [1, 2]

        out_00 = gF.get_full_log_joint_probs(
            inpt_0, self.means_0, self.full_factors_0, self.norm_logits_0
        )
        out_00_shape = list(out_00.shape)

        if out_00_shape != true_out_00_shape:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks the output shape of 'get_full_log_joint_probs' on case 10.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        true_out_10_shape = [2, 2]

        out_10 = gF.get_full_log_joint_probs(
            inpt_1, self.means_0, self.full_factors_0, self.norm_logits_0
        )
        out_10_shape = list(out_10.shape)

        if out_10_shape != true_out_10_shape:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks the output shape of 'get_full_log_joint_probs' on case 20.
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        true_out_20_shape = [3, 2]

        out_20 = gF.get_full_log_joint_probs(
            inpt_2, self.means_0, self.full_factors_0, self.norm_logits_0
        )
        out_20_shape = list(out_20.shape)

        if out_20_shape != true_out_20_shape:
            self.assertTrue(False)

    def test_30(self: Self) -> None:
        """
        Checks that 'get_full_log_joint_probs' follows broadcasting rules
        over a batch of mixtures.
        """

        torch.manual_seed(0)

        inpt_3 = torch.randn(5, 3)
        means_3 = torch.randn(2, 4, 3)
        full_factors_3 = get_random_full_factors(2, 4, 3)
        norm_logits_3 = log_softmax(torch.randn(2, 4))

        out_30 = gF.get_full_log_joint_probs(
            inpt_3, means_3, full_factors_3, norm_logits_3
        )

        if list(out_30.shape) != [5, 2, 4]:
            self.assertTrue(False)

        for i in range(2):
            true_out_30 = gF.get_full_log_joint_probs(
                inpt_3, means_3[i], full_factors_3[i], norm_logits_3[i]
            )

            if not torch.allclose(out_30[:, i], true_out_30, atol=1e-5):
                self.assertTrue(False)

    def test_31(self: Self) -> None:
        """
        Checks that 'get_full_log_joint_probs' and its private form match
        torch.distributions for random, non-identity Cholesky factors.
        """

        torch.manual_seed(0)

        inpt_3 = torch.randn(5, 3)
        means_3 = torch.randn(4, 3)
        full_factors_3 = get_random_full_factors(4, 3)
        norm_logits_3 = log_softmax(torch.randn(4))

        out_31 = gF.get_full_log_joint_probs(
            inpt_3, means_3, full_factors_3, norm_logits_3
        )
        private_out_31 = gF._get_full_log_joint_probs(
            inpt_3, means_3, full_factors_3, norm_logits_3
        )
        true_out_31 = norm_logits_3 + torch.stack(
            [
                MultivariateNormal(
                    mean, precision_matrix=factor.mT @ factor
                ).log_prob(inpt_3)
                for mean, factor in zip(means_3, full_factors_3, strict=True)
            ],
            dim=-1,
        )

        if not torch.equal(out_31, private_out_31):
            self.assertTrue(False)

        if not torch.allclose(out_31, true_out_31, atol=1e-4):
            self.assertTrue(False)

    def test_40(self: Self) -> None:
        """
        Checks that 'get_full_log_joint_probs' accepts 1 × 1 factors and
        agrees with the diag parametrization.
        """

        torch.manual_seed(0)

        inpt_0 = torch.randn(5, 1)
        means_0 = torch.randn(3, 1)
        diag_factors_0 = torch.rand(3, 1) + 0.5
        norm_logits_0 = log_softmax(torch.randn(3))

        out_0 = gF.get_full_log_joint_probs(
            inpt_0, means_0, diag_factors_0.diag_embed(), norm_logits_0
        )
        true_out_0 = gF.get_diag_log_joint_probs(
            inpt_0, means_0, diag_factors_0, norm_logits_0
        )

        if not torch.allclose(out_0, true_out_0, atol=1e-5):
            self.assertTrue(False)


class TestGetFullSamples(TestCase):
    """
    'get_full_samples' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
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
        Checks the output shape of 'get_full_samples' on case 00.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_full_samples(self.means_0, self.full_factors_0)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks the output shape of 'get_full_samples' on case 10.
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_full_samples(
            self.means_0, self.full_factors_0, *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks the output shape of 'get_full_samples' on case 20.
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_full_samples(
            self.means_0,
            self.full_factors_0,
            *true_out_batch_shape_2,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        Checks the output shape of 'get_full_samples' on case 01.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_full_samples(self.means_1, self.full_factors_1)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: Self) -> None:
        """
        Checks the output shape of 'get_full_samples' on case 11.
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_full_samples(
            self.means_1, self.full_factors_1, *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: Self) -> None:
        """
        Checks the output shape of 'get_full_samples' on case 21.
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_full_samples(
            self.means_1, self.full_factors_1, *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks that 'get_full_samples' draws independent noise for each
        component when the factors have a larger batch shape than the
        means.
        """

        torch.manual_seed(0)

        out_3 = gF.get_full_samples(
            torch.zeros(1, 2), torch.eye(2).expand(3, 2, 2), 4
        )

        if list(out_3.shape) != [4, 3, 2]:
            self.assertTrue(False)

        if torch.allclose(out_3[:, 0], out_3[:, 1]):
            self.assertTrue(False)

    def test_4(self: Self) -> None:
        """
        Checks that the samples of 'get_full_samples' have the covariance given
        by the inverse of the precision matrix.
        """

        torch.manual_seed(0)

        means_4 = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
        full_factors_4 = torch.tensor(
            [[[2.0, 0.5], [0.0, 1.0]]], dtype=torch.float64
        )

        out_4 = gF.get_full_samples(means_4, full_factors_4, 200_000).squeeze(
            1
        )

        true_cov_4 = torch.linalg.inv(full_factors_4.mT @ full_factors_4)[0]

        if not torch.allclose(out_4.mean(0), means_4[0], atol=1e-2):
            self.assertTrue(False)

        if not torch.allclose(out_4.mT.cov(), true_cov_4, atol=1e-2):
            self.assertTrue(False)


class TestGetDiagKLDivs(TestCase):
    """
    'get_diag_kl_divs' unit testing
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
        Checks that 'get_diag_kl_divs' agrees with the full parametrization on
        case 01.
        """

        full_out_01 = gF._get_full_kl_divs(
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
        Checks that 'get_diag_kl_divs' agrees with the full parametrization on
        case 10.
        """

        full_out_10 = gF._get_full_kl_divs(
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
        Checks that 'get_diag_kl_divs' agrees with the full parametrization on
        case 02.
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

        full_out_02 = gF._get_full_kl_divs(
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
        Checks that 'get_diag_kl_divs' agrees with the full parametrization on
        case 03.
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

        full_out_03 = gF._get_full_kl_divs(
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

    def test_45(self: Self) -> None:
        """
        Checks that 'get_diag_kl_divs' matches torch.distributions for
        random, non-identity diagonal factors.
        """

        torch.manual_seed(0)

        means_4 = torch.randn(2, 3)
        diag_factors_4 = torch.rand(2, 3) + 0.5
        means_5 = torch.randn(4, 3)
        diag_factors_5 = torch.rand(4, 3) + 0.5

        out_45 = gF.get_diag_kl_divs(
            means_4, diag_factors_4, means_5, diag_factors_5
        )
        true_out_45 = get_ref_kl_divs(
            means_4,
            diag_factors_4.diag_embed(),
            means_5,
            diag_factors_5.diag_embed(),
        )

        if not torch.allclose(out_45, true_out_45, atol=1e-4):
            self.assertTrue(False)


class TestGetDiagLogJointProbs(TestCase):
    """
    'get_diag_log_joint_probs' unit testing
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
        Checks that 'get_diag_log_joint_probs' agrees with the full
        parametrization on case 00.
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        full_out_00 = gF.get_full_log_joint_probs(
            inpt_0, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        diag_out_00 = gF.get_diag_log_joint_probs(
            inpt_0, self.means_0, self.diag_factors_0, self.norm_logits_0
        )

        if (diag_out_00 != full_out_00).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'get_diag_log_joint_probs' agrees with the full
        parametrization on case 10.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_10 = gF.get_full_log_joint_probs(
            inpt_1, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        diag_out_10 = gF.get_diag_log_joint_probs(
            inpt_1, self.means_0, self.diag_factors_0, self.norm_logits_0
        )

        if (diag_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks that 'get_diag_log_joint_probs' agrees with the full
        parametrization on case 20.
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_20 = gF.get_full_log_joint_probs(
            inpt_2, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        diag_out_20 = gF.get_diag_log_joint_probs(
            inpt_2, self.means_0, self.diag_factors_0, self.norm_logits_0
        )

        if (diag_out_20 != full_out_20).any():
            self.assertTrue(False)

    def test_30(self: Self) -> None:
        """
        Checks that 'get_diag_log_joint_probs' follows broadcasting rules
        over a batch of mixtures.
        """

        torch.manual_seed(0)

        inpt_3 = torch.randn(5, 3)
        means_3 = torch.randn(2, 4, 3)
        diag_factors_3 = torch.rand(2, 4, 3) + 0.5
        norm_logits_3 = log_softmax(torch.randn(2, 4))

        out_30 = gF.get_diag_log_joint_probs(
            inpt_3, means_3, diag_factors_3, norm_logits_3
        )
        full_out_30 = gF.get_full_log_joint_probs(
            inpt_3, means_3, diag_factors_3.diag_embed(), norm_logits_3
        )

        if list(out_30.shape) != [5, 2, 4]:
            self.assertTrue(False)

        if not torch.allclose(out_30, full_out_30, atol=1e-5):
            self.assertTrue(False)


class TestGetDiagSamples(TestCase):
    """
    'get_diag_samples' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
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
        self.diag_factors_1 = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

    def test_00(self: Self) -> None:
        """
        Checks the output shape of 'get_diag_samples' on case 00.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_diag_samples(self.means_0, self.diag_factors_0)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks the output shape of 'get_diag_samples' on case 10.
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_diag_samples(
            self.means_0, self.diag_factors_0, *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks the output shape of 'get_diag_samples' on case 20.
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_diag_samples(
            self.means_0, self.diag_factors_0, *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        Checks the output shape of 'get_diag_samples' on case 01.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_diag_samples(self.means_1, self.diag_factors_1)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: Self) -> None:
        """
        Checks the output shape of 'get_diag_samples' on case 11.
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
        Checks the output shape of 'get_diag_samples' on case 21.
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

    def test_3(self: Self) -> None:
        """
        Checks that 'get_diag_samples' draws independent noise for each
        component when the factors have a larger batch shape than the
        means.
        """

        torch.manual_seed(0)

        out_3 = gF.get_diag_samples(torch.zeros(1, 2), torch.ones(3, 2), 4)

        if list(out_3.shape) != [4, 3, 2]:
            self.assertTrue(False)

        if torch.allclose(out_3[:, 0], out_3[:, 1]):
            self.assertTrue(False)

    def test_4(self: Self) -> None:
        """
        Checks that the samples of 'get_diag_samples' have the covariance given
        by the inverse of the precision matrix.
        """

        torch.manual_seed(0)

        means_4 = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
        full_factors_4 = torch.tensor(
            [[2.0, 0.5]], dtype=torch.float64
        ).diag_embed()

        out_4 = gF.get_diag_samples(
            means_4, full_factors_4.diagonal(dim1=-2, dim2=-1), 200_000
        ).squeeze(1)

        true_cov_4 = torch.linalg.inv(full_factors_4.mT @ full_factors_4)[0]

        if not torch.allclose(out_4.mean(0), means_4[0], atol=1e-2):
            self.assertTrue(False)

        if not torch.allclose(out_4.mT.cov(), true_cov_4, atol=1e-2):
            self.assertTrue(False)


class TestGetIsoKLDivs(TestCase):
    """
    'get_iso_kl_divs' unit testing
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
        self.iso_factors_0 = torch.tensor([1.0, 1.0])
        self.shared_iso_factors_0 = torch.tensor([1.0])

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
        self.iso_factors_1 = torch.tensor([1.0])
        self.shared_iso_factors_1 = torch.tensor([1.0])

        self.means_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_factors_2 = torch.tensor(
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
        self.iso_factors_2 = torch.tensor([1.0, 1.0])
        self.shared_iso_factors_2 = torch.tensor([1.0])

        self.means_3 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.full_factors_3 = torch.tensor(
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
        self.iso_factors_3 = torch.tensor([1.0, 1.0, 1.0])
        self.shared_iso_factors_3 = torch.tensor([1.0])

    def test_01(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 01.
        """

        full_out_01 = gF._get_full_kl_divs(
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

    def test_01_shared(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 01_shared.
        """

        full_out_01 = gF._get_full_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_1,
            self.full_factors_1,
        )

        shared_iso_out_01 = gF.get_iso_kl_divs(
            self.means_0,
            self.shared_iso_factors_0,
            self.means_1,
            self.shared_iso_factors_1,
        )

        if (shared_iso_out_01 != full_out_01).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 10.
        """

        full_out_10 = gF._get_full_kl_divs(
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

    def test_10_shared(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 10_shared.
        """

        full_out_10 = gF._get_full_kl_divs(
            self.means_1,
            self.full_factors_1,
            self.means_0,
            self.full_factors_0,
        )

        shared_iso_out_10 = gF.get_iso_kl_divs(
            self.means_1,
            self.shared_iso_factors_1,
            self.means_0,
            self.shared_iso_factors_0,
        )

        if (shared_iso_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_02(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 02.
        """

        full_out_02 = gF._get_full_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_2,
            self.full_factors_2,
        )

        iso_out_02 = gF.get_iso_kl_divs(
            self.means_0,
            self.iso_factors_0,
            self.means_2,
            self.iso_factors_2,
        )

        if (iso_out_02 != full_out_02).any():
            self.assertTrue(False)

    def test_02_shared(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 02_shared.
        """

        full_out_02 = gF._get_full_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_2,
            self.full_factors_2,
        )

        shared_iso_out_02 = gF.get_iso_kl_divs(
            self.means_0,
            self.shared_iso_factors_0,
            self.means_2,
            self.shared_iso_factors_2,
        )

        if (shared_iso_out_02 != full_out_02).any():
            self.assertTrue(False)

    def test_03(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 03.
        """

        full_out_03 = gF._get_full_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_3,
            self.full_factors_3,
        )

        iso_out_03 = gF.get_iso_kl_divs(
            self.means_0,
            self.iso_factors_0,
            self.means_3,
            self.iso_factors_3,
        )

        if (iso_out_03 != full_out_03).any():
            self.assertTrue(False)

    def test_03_shared(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' agrees with the full parametrization on
        case 03_shared.
        """

        full_out_03 = gF._get_full_kl_divs(
            self.means_0,
            self.full_factors_0,
            self.means_3,
            self.full_factors_3,
        )

        shared_iso_out_03 = gF.get_iso_kl_divs(
            self.means_0,
            self.shared_iso_factors_0,
            self.means_3,
            self.shared_iso_factors_3,
        )

        if (shared_iso_out_03 != full_out_03).any():
            self.assertTrue(False)

    def test_45(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' matches torch.distributions for
        random, non-identity isometric factors.
        """

        torch.manual_seed(0)

        means_4 = torch.randn(2, 3)
        iso_factors_4 = torch.rand(2) + 0.5
        means_5 = torch.randn(4, 3)
        iso_factors_5 = torch.rand(4) + 0.5

        out_45 = gF.get_iso_kl_divs(
            means_4, iso_factors_4, means_5, iso_factors_5
        )
        true_out_45 = get_ref_kl_divs(
            means_4,
            iso_factors_4.unsqueeze(-1).expand(-1, 3).diag_embed(),
            means_5,
            iso_factors_5.unsqueeze(-1).expand(-1, 3).diag_embed(),
        )

        if not torch.allclose(out_45, true_out_45, atol=1e-4):
            self.assertTrue(False)

    def test_45_shared(self: Self) -> None:
        """
        Checks that 'get_iso_kl_divs' matches torch.distributions for
        random, non-identity shared isometric factors.
        """

        torch.manual_seed(0)

        means_4 = torch.randn(2, 3)
        shared_iso_factors_4 = torch.rand(1) + 0.5
        means_5 = torch.randn(4, 3)
        shared_iso_factors_5 = torch.rand(1) + 0.5

        out_45 = gF.get_iso_kl_divs(
            means_4, shared_iso_factors_4, means_5, shared_iso_factors_5
        )
        true_out_45 = get_ref_kl_divs(
            means_4,
            shared_iso_factors_4.expand(2, 3).diag_embed(),
            means_5,
            shared_iso_factors_5.expand(4, 3).diag_embed(),
        )

        if not torch.allclose(out_45, true_out_45, atol=1e-4):
            self.assertTrue(False)


class TestGetIsoLogJointProbs(TestCase):
    """
    'get_iso_log_joint_probs' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
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
        self.iso_factors_0 = torch.tensor([1.0, 1.0, 1.0])
        self.shared_iso_factors_0 = torch.tensor([1.0])
        self.norm_logits_0 = torch.tensor([0.0, 0.0, 0.0])
        self.norm_logits_0 = log_softmax(self.norm_logits_0)

    def test_00(self: Self) -> None:
        """
        Checks that 'get_iso_log_joint_probs' agrees with the full
        parametrization on case 00.
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        full_out_00 = gF.get_full_log_joint_probs(
            inpt_0, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        iso_out_00 = gF.get_iso_log_joint_probs(
            inpt_0, self.means_0, self.iso_factors_0, self.norm_logits_0
        )

        if (iso_out_00 != full_out_00).any():
            self.assertTrue(False)

    def test_00_shared(self: Self) -> None:
        """
        Checks that 'get_iso_log_joint_probs' agrees with the full
        parametrization on case 00_shared.
        """

        inpt_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )

        full_out_00 = gF.get_full_log_joint_probs(
            inpt_0, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        shared_iso_out_00 = gF.get_iso_log_joint_probs(
            inpt_0, self.means_0, self.shared_iso_factors_0, self.norm_logits_0
        )

        if (shared_iso_out_00 != full_out_00).any():
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks that 'get_iso_log_joint_probs' agrees with the full
        parametrization on case 10.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_10 = gF.get_full_log_joint_probs(
            inpt_1, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        iso_out_10 = gF.get_iso_log_joint_probs(
            inpt_1, self.means_0, self.iso_factors_0, self.norm_logits_0
        )

        if (iso_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_10_shared(self: Self) -> None:
        """
        Checks that 'get_iso_log_joint_probs' agrees with the full
        parametrization on case 10_shared.
        """

        inpt_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_10 = gF.get_full_log_joint_probs(
            inpt_1, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        shared_iso_out_10 = gF.get_iso_log_joint_probs(
            inpt_1, self.means_0, self.shared_iso_factors_0, self.norm_logits_0
        )

        if (shared_iso_out_10 != full_out_10).any():
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks that 'get_iso_log_joint_probs' agrees with the full
        parametrization on case 20.
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_20 = gF.get_full_log_joint_probs(
            inpt_2, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        iso_out_20 = gF.get_iso_log_joint_probs(
            inpt_2, self.means_0, self.iso_factors_0, self.norm_logits_0
        )

        if (iso_out_20 != full_out_20).any():
            self.assertTrue(False)

    def test_20_shared(self: Self) -> None:
        """
        Checks that 'get_iso_log_joint_probs' agrees with the full
        parametrization on case 20_shared.
        """

        inpt_2 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        full_out_20 = gF.get_full_log_joint_probs(
            inpt_2, self.means_0, self.full_factors_0, self.norm_logits_0
        )

        shared_iso_out_20 = gF.get_iso_log_joint_probs(
            inpt_2, self.means_0, self.shared_iso_factors_0, self.norm_logits_0
        )

        if (shared_iso_out_20 != full_out_20).any():
            self.assertTrue(False)

    def test_30(self: Self) -> None:
        """
        Checks that 'get_iso_log_joint_probs' follows broadcasting rules
        over a batch of mixtures.
        """

        torch.manual_seed(0)

        inpt_3 = torch.randn(5, 3)
        means_3 = torch.randn(2, 4, 3)
        iso_factors_3 = torch.rand(2, 4) + 0.5
        norm_logits_3 = log_softmax(torch.randn(2, 4))

        out_30 = gF.get_iso_log_joint_probs(
            inpt_3, means_3, iso_factors_3, norm_logits_3
        )
        full_out_30 = gF.get_full_log_joint_probs(
            inpt_3,
            means_3,
            iso_factors_3.unsqueeze(-1).expand(-1, -1, 3).diag_embed(),
            norm_logits_3,
        )

        if list(out_30.shape) != [5, 2, 4]:
            self.assertTrue(False)

        if not torch.allclose(out_30, full_out_30, atol=1e-5):
            self.assertTrue(False)


class TestGetIsoSamples(TestCase):
    """
    'get_iso_samples' unit testing
    """

    def setUp(self: Self) -> None:
        """
        Builds the shared test fixtures.
        """

        self.means_0 = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        self.iso_factors_0 = torch.tensor([1.0])
        self.shared_iso_factors_0 = torch.tensor([1.0])

        self.means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.iso_factors_1 = torch.tensor([1.0, 1.0])
        self.shared_iso_factors_1 = torch.tensor([1.0])

    def test_00(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 00.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_iso_samples(self.means_0, self.iso_factors_0)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_00_shared(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 00_shared.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_iso_samples(self.means_0, self.shared_iso_factors_0)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_10(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 10.
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

    def test_10_shared(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 10_shared.
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_iso_samples(
            self.means_0,
            self.shared_iso_factors_0,
            *true_out_batch_shape_1,
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_20(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 20.
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_iso_samples(
            self.means_0, self.iso_factors_0, *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_20_shared(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 20_shared.
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_iso_samples(
            self.means_0, self.shared_iso_factors_0, *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_01(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 01.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_iso_samples(self.means_1, self.iso_factors_1)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_01_shared(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 01_shared.
        """

        true_out_batch_shape_0: list[int] = []

        out = gF.get_iso_samples(self.means_1, self.shared_iso_factors_1)
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_0:
            self.assertTrue(False)

    def test_11(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 11.
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_iso_samples(
            self.means_1, self.iso_factors_1, *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_11_shared(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 11_shared.
        """

        true_out_batch_shape_1 = [1]

        out = gF.get_iso_samples(
            self.means_1, self.shared_iso_factors_1, *true_out_batch_shape_1
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_1:
            self.assertTrue(False)

    def test_21(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 21.
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_iso_samples(
            self.means_1, self.iso_factors_1, *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_21_shared(self: Self) -> None:
        """
        Checks the output shape of 'get_iso_samples' on case 21_shared.
        """

        true_out_batch_shape_2 = [1, 1]

        out = gF.get_iso_samples(
            self.means_1, self.shared_iso_factors_1, *true_out_batch_shape_2
        )
        *out_batch_shape, _, _ = list(out.shape)

        if out_batch_shape != true_out_batch_shape_2:
            self.assertTrue(False)

    def test_3(self: Self) -> None:
        """
        Checks that 'get_iso_samples' draws independent noise for each
        component when the factors have a larger batch shape than the
        means.
        """

        torch.manual_seed(0)

        out_3 = gF.get_iso_samples(torch.zeros(1, 2), torch.ones(3), 4)

        if list(out_3.shape) != [4, 3, 2]:
            self.assertTrue(False)

        if torch.allclose(out_3[:, 0], out_3[:, 1]):
            self.assertTrue(False)

    def test_4(self: Self) -> None:
        """
        Checks that the samples of 'get_iso_samples' have the covariance given
        by the inverse of the precision matrix.
        """

        torch.manual_seed(0)

        means_4 = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
        full_factors_4 = torch.tensor(
            [[2.0, 2.0]], dtype=torch.float64
        ).diag_embed()

        out_4 = gF.get_iso_samples(
            means_4, full_factors_4[..., 0, 0], 200_000
        ).squeeze(1)

        true_cov_4 = torch.linalg.inv(full_factors_4.mT @ full_factors_4)[0]

        if not torch.allclose(out_4.mean(0), means_4[0], atol=1e-2):
            self.assertTrue(False)

        if not torch.allclose(out_4.mT.cov(), true_cov_4, atol=1e-2):
            self.assertTrue(False)


class TestGetTrTriuTriu(TestCase):
    """
    'get_tr_triu_triu' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_tr_triu_triu' raises IsNotUpperTriangular when the
        Cholesky factors are not upper-triangular.
        """

        full_factors_0 = torch.eye(2).unsqueeze(0)
        full_factors_1 = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
            ]
        )

        try:
            gF.get_tr_triu_triu(full_factors_0, full_factors_1)
        except val.IsNotUpperTriangular:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestGetFullJointEntropyUpperBound(TestCase):
    """
    'get_full_joint_entropy_upper_bound' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_full_joint_entropy_upper_bound' equals the entropy
        of a single Gaussian component.
        """

        torch.manual_seed(0)

        means_0 = torch.randn(1, 3)
        full_factors_0 = get_random_full_factors(1, 3)
        norm_logits_0 = log_softmax(torch.zeros(1))

        out_0 = gF.get_full_joint_entropy_upper_bound(
            means_0, full_factors_0, norm_logits_0
        )
        true_out_0 = MultivariateNormal(
            means_0[0],
            precision_matrix=full_factors_0[0].mT @ full_factors_0[0],
        ).entropy()

        if not torch.allclose(out_0, true_out_0, atol=1e-5):
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'get_full_joint_entropy_upper_bound' adds the entropy of
        the weights for two far-apart, equally weighted components.
        """

        means_1 = torch.tensor(
            [
                [0.0, 0.0],
                [1e3, 0.0],
            ]
        )
        full_factors_1 = torch.eye(2).expand(2, 2, 2)
        norm_logits_1 = log_softmax(torch.zeros(2))

        out_1 = gF.get_full_joint_entropy_upper_bound(
            means_1, full_factors_1, norm_logits_1
        )
        true_out_1 = MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        ).entropy() + math.log(2.0)

        if not torch.allclose(out_1, true_out_1, atol=1e-5):
            self.assertTrue(False)

    def test_2(self: Self) -> None:
        """
        Checks that 'get_full_joint_entropy_upper_bound' raises IsNotVector
        for a batch of mixtures.
        """

        means_2 = torch.zeros(2, 2, 2)
        full_factors_2 = torch.eye(2).expand(2, 2, 2, 2)
        norm_logits_2 = log_softmax(torch.zeros(2, 2))

        try:
            gF.get_full_joint_entropy_upper_bound(
                means_2, full_factors_2, norm_logits_2
            )
        except val.IsNotVector:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestGetDiagJointEntropyUpperBound(TestCase):
    """
    'get_diag_joint_entropy_upper_bound' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_diag_joint_entropy_upper_bound' agrees with the full
        parametrization for random, non-identity diagonal factors.
        """

        torch.manual_seed(0)

        means_0 = torch.randn(4, 3)
        diag_factors_0 = torch.rand(4, 3) + 0.5
        norm_logits_0 = log_softmax(torch.randn(4))

        out_0 = gF.get_diag_joint_entropy_upper_bound(
            means_0, diag_factors_0, norm_logits_0
        )
        full_out_0 = gF.get_full_joint_entropy_upper_bound(
            means_0, diag_factors_0.diag_embed(), norm_logits_0
        )

        if not torch.allclose(out_0, full_out_0, atol=1e-5):
            self.assertTrue(False)


class TestGetIsoJointEntropyUpperBound(TestCase):
    """
    'get_iso_joint_entropy_upper_bound' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_iso_joint_entropy_upper_bound' agrees with the full
        parametrization for random, non-identity isometric factors.
        """

        torch.manual_seed(0)

        means_0 = torch.randn(4, 3)
        iso_factors_0 = torch.rand(4) + 0.5
        norm_logits_0 = log_softmax(torch.randn(4))

        out_0 = gF.get_iso_joint_entropy_upper_bound(
            means_0, iso_factors_0, norm_logits_0
        )
        full_out_0 = gF.get_full_joint_entropy_upper_bound(
            means_0,
            iso_factors_0.unsqueeze(-1).expand(-1, 3).diag_embed(),
            norm_logits_0,
        )

        if not torch.allclose(out_0, full_out_0, atol=1e-5):
            self.assertTrue(False)

    def test_0_shared(self: Self) -> None:
        """
        Checks that 'get_iso_joint_entropy_upper_bound' agrees with the full
        parametrization for a random, non-identity shared isometric factor.
        """

        torch.manual_seed(0)

        means_0 = torch.randn(4, 3)
        shared_iso_factors_0 = torch.rand(1) + 0.5
        norm_logits_0 = log_softmax(torch.randn(4))

        out_0 = gF.get_iso_joint_entropy_upper_bound(
            means_0, shared_iso_factors_0, norm_logits_0
        )
        full_out_0 = gF.get_full_joint_entropy_upper_bound(
            means_0,
            shared_iso_factors_0.expand(4, 3).diag_embed(),
            norm_logits_0,
        )

        if not torch.allclose(out_0, full_out_0, atol=1e-5):
            self.assertTrue(False)


class TestGetFullModalEntropies(TestCase):
    """
    'get_full_modal_entropies' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_full_modal_entropies' agrees with torch.distributions.
        """

        torch.manual_seed(0)

        full_factors_0 = get_random_full_factors(4, 3)

        out_0 = gF.get_full_modal_entropies(full_factors_0)
        true_out_0 = MultivariateNormal(
            torch.zeros(4, 3),
            precision_matrix=full_factors_0.mT @ full_factors_0,
        ).entropy()

        if not torch.allclose(out_0, true_out_0, atol=1e-5):
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'get_full_modal_entropies' accepts 1 × 1 factors.
        """

        out_1 = gF.get_full_modal_entropies(torch.ones(2, 1, 1))
        true_out_1 = torch.full((2,), (1 + math.log(2 * math.pi)) / 2)

        if not torch.allclose(out_1, true_out_1):
            self.assertTrue(False)


class TestGetDiagLogDets(TestCase):
    """
    'get_diag_log_dets' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_diag_log_dets' agrees with the full parametrization.
        """

        diag_factors_0 = torch.rand(4, 3) + 0.5

        out_0 = gF.get_diag_log_dets(diag_factors_0)
        true_out_0 = gF.get_full_log_dets(diag_factors_0.diag_embed())

        if not torch.allclose(out_0, true_out_0):
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'get_diag_log_dets' raises HasNonPositiveElement.
        """

        try:
            gF.get_diag_log_dets(torch.zeros(4, 3))
        except val.HasNonPositiveElement:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestGetDiagModalEntropies(TestCase):
    """
    'get_diag_modal_entropies' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_diag_modal_entropies' agrees with the full
        parametrization.
        """

        diag_factors_0 = torch.rand(4, 3) + 0.5

        out_0 = gF.get_diag_modal_entropies(diag_factors_0)
        true_out_0 = gF.get_full_modal_entropies(diag_factors_0.diag_embed())

        if not torch.allclose(out_0, true_out_0):
            self.assertTrue(False)


class TestGetIsoLogDets(TestCase):
    """
    'get_iso_log_dets' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_iso_log_dets' agrees with the diag parametrization.
        """

        iso_factors_0 = torch.rand(4) + 0.5

        out_0 = gF.get_iso_log_dets(iso_factors_0, 3)
        true_out_0 = gF.get_diag_log_dets(
            iso_factors_0.unsqueeze(-1).expand(-1, 3)
        )

        if not torch.allclose(out_0, true_out_0):
            self.assertTrue(False)

    def test_1(self: Self) -> None:
        """
        Checks that 'get_iso_log_dets' raises IsNotPositiveInteger on a
        non-positive dimensionality.
        """

        try:
            gF.get_iso_log_dets(torch.ones(4), 0)
        except val.IsNotPositiveInteger:
            pass
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(False)


class TestGetIsoModalEntropies(TestCase):
    """
    'get_iso_modal_entropies' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_iso_modal_entropies' agrees with the diag
        parametrization.
        """

        iso_factors_0 = torch.rand(4) + 0.5

        out_0 = gF.get_iso_modal_entropies(iso_factors_0, 3)
        true_out_0 = gF.get_diag_modal_entropies(
            iso_factors_0.unsqueeze(-1).expand(-1, 3)
        )

        if not torch.allclose(out_0, true_out_0):
            self.assertTrue(False)


class TestGetJointEntropyUpperBound(TestCase):
    """
    'get_joint_entropy_upper_bound' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Checks that 'get_joint_entropy_upper_bound' agrees with
        'get_full_joint_entropy_upper_bound'.
        """

        torch.manual_seed(0)

        means_0 = torch.randn(4, 3)
        full_factors_0 = get_random_full_factors(4, 3)
        norm_logits_0 = log_softmax(torch.randn(4))

        out_0 = gF.get_joint_entropy_upper_bound(
            gF.get_full_modal_entropies(full_factors_0),
            gF.get_full_kl_divs(
                means_0, full_factors_0, means_0, full_factors_0
            ),
            norm_logits_0,
        )
        true_out_0 = gF.get_full_joint_entropy_upper_bound(
            means_0, full_factors_0, norm_logits_0
        )

        if not torch.allclose(out_0, true_out_0):
            self.assertTrue(False)
