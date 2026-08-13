"""
Neural network layers for Gaussian mixture models.
"""

from __future__ import annotations
from math import log
from typing import Any

from torch import Tensor, zeros
from torch.nn import Parameter
from torch.nn.functional import softplus

from gmml.codebook import _WeightedCodeBook, _GaussianCodeBook

from gmml.functional import get_log_joint_prob, get_sample
from gmml.functional import get_diag_log_joint_prob, get_diag_sample
from gmml.functional import get_iso_log_joint_prob, get_iso_sample

from gmml.util import set_softplus_diag


class GMMLayer(_WeightedCodeBook, _GaussianCodeBook):
    """
    Full-covariance Gaussian mixture layer.
    """

    def _init_unconstrained_full_fct(self: GMMLayer) -> None:
        """
        Initializes the unconstrained full_fct parameter.

        Args:
            None.

        Returns:
            None: This function returns only after successful initialization.
        """

        unconstrained_full_fct = zeros(self.size, self.dim, self.dim)

        self.unconstrained_full_fct = Parameter(
            unconstrained_full_fct,
            requires_grad=True
        )

    def __init__(
        self: GMMLayer,
        size: int,
        dim: int,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base module.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(size, dim, *args, **kwargs)

        self._init_unconstrained_full_fct()

    @property
    def full_fct(self: GMMLayer) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Upper-triangular Cholesky factors of the mixture
            components.
        """

        return set_softplus_diag(self.unconstrained_full_fct).triu()

    def get_log_joint_prob(self: GMMLayer, inpt: Tensor) -> Tensor:
        """
        Returns the logarithm of the joint probability.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return get_log_joint_prob(
            inpt,
            self.mean,
            self.full_fct,
            self.norm_logit
        )

    def sample(self: GMMLayer, *sample_batch_shape: int) -> Tensor:
        """
        Samples each component.

        Args:
            *sample_batch_shape: Optional leading batch shape for the
                generated samples.

        Returns:
            Tensor: Samples drawn from each mixture component.
        """

        return get_sample(self.mean, self.full_fct, *sample_batch_shape)


class DiagGMMLayer(_WeightedCodeBook, _GaussianCodeBook):
    """
    Diagonal-constrained Gaussian mixture layer.
    """

    def _init_unconstrained_diag_fct(self: DiagGMMLayer) -> None:
        """
        Initializes the unconstrained diag_fct parameter.

        Args:
            None.

        Returns:
            None: This function returns only after successful initialization.
        """

        unconstrained_diag_fct = zeros(self.size, self.dim)

        self.unconstrained_diag_fct = Parameter(
            unconstrained_diag_fct,
            requires_grad=True
        )

    def __init__(
        self: DiagGMMLayer,
        size: int,
        dim: int,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base module.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(size, dim, *args, **kwargs)

        self._init_unconstrained_diag_fct()

    @property
    def diag_fct(self: DiagGMMLayer) -> Tensor:
        """
        Returns the diagonally-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Positive diagonal factors of the mixture components.
        """

        return softplus(self.unconstrained_diag_fct) / log(2.0)

    @property
    def full_fct(self: DiagGMMLayer) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Diagonal Cholesky factors of the mixture components.
        """

        return self.diag_fct.diag_embed()

    def get_log_joint_prob(self: DiagGMMLayer, inpt: Tensor) -> Tensor:
        """
        Returns the logarithm of the joint probability distributions.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return get_diag_log_joint_prob(
            inpt,
            self.mean,
            self.diag_fct,
            self.norm_logit
        )

    def sample(self: DiagGMMLayer, *sample_batch_shape: int) -> Tensor:
        """
        Samples each component.

        Args:
            *sample_batch_shape: Optional leading batch shape for the
                generated samples.

        Returns:
            Tensor: Samples drawn from each mixture component.
        """


        return get_diag_sample(self.mean, self.diag_fct, *sample_batch_shape)


class IsoGMMLayer(_WeightedCodeBook, _GaussianCodeBook):
    """
    Isoemtric-covariance Gaussian mixture layer.
    """

    def _init_unconstrained_iso_fct(self: IsoGMMLayer) -> None:
        """
        Initializes the unconstrained iso_fct parameter.

        Args:
            None.

        Returns:
            None: This function returns only after successful initialization.
        """

        unconstrained_iso_fct = zeros(self.size)

        self.unconstrained_iso_fct = Parameter(
            unconstrained_iso_fct,
            requires_grad=True
        )

    def __init__(
        self: IsoGMMLayer,
        size: int,
        dim: int,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base module.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(size, dim, *args, **kwargs)

        self._init_unconstrained_iso_fct()

    @property
    def iso_fct(self: IsoGMMLayer) -> Tensor:
        """
        Returns the isometrically-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Positive isometric factors of the mixture components.
        """

        return softplus(self.unconstrained_iso_fct) / log(2.0)

    @property
    def diag_fct(self: IsoGMMLayer) -> Tensor:
        """
        Returns the diagonally-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Diagonal factors derived from the isoemtric parameters.
        """

        return self.iso_fct.unsqueeze(-1).expand(-1, self.dim)

    @property
    def full_fct(self: IsoGMMLayer) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Diagonal Cholesky factors of the mixture components.
        """

        return self.diag_fct.diag_embed()

    def get_log_joint_prob(self: IsoGMMLayer, inpt: Tensor) -> Tensor:
        """
        Returns the logarithm of the joint probability distributions under an
        isometric constraint.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return get_iso_log_joint_prob(
            inpt,
            self.codevector,
            self.iso_fct,
            self.norm_logit
        )

    def sample(self: IsoGMMLayer, *sample_batch_shape: int) -> Tensor:
        """
        Samples each component.

        Args:
            *sample_batch_shape: Optional leading batch shape for the
                generated samples.

        Returns:
            Tensor: Samples drawn from each mixture component.
        """

        return get_iso_sample(self.mean, self.iso_fct, *sample_batch_shape)
