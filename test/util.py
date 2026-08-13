"""
Testing utilities module
"""

from __future__ import annotations
from unittest import TestCase

from torch import Tensor, tensor
from torch.nn.functional import log_softmax as _log_softmax
from torch.testing import assert_close


def log_softmax(inpt: Tensor) -> Tensor:
    """
    Applies log_softmax over batch of scalars.

    Args:
        inpt: Input tensor of scalar values.

    Returns:
        Tensor: Log-softmax values reshaped to the input shape.
    """

    out = inpt.flatten()
    out = _log_softmax(out, dim=-1)
    out = out.unflatten(-1, inpt.shape)

    return out


class TestLogSoftmax(TestCase):
    """
    'log_softmax' unit testing
    """

    def test_0(self: TestLogSoftmax) -> None:
        """
        """

        inpt_0 =  tensor(
            [0.0, 0.0]
        )

        out_0 = log_softmax(inpt_0)

        try:
            assert_close(
                out_0.exp().sum(),
                tensor(1.0),
                check_dtype=False,
                check_device=False
            )
        except:
            self.assertTrue(False)
