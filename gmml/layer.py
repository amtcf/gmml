"""
Neural network layers for Gaussian mixture models.
"""

from typing import Self, Any, Optional

from math import log


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


import gmml.codebook as cb
import gmml.functional as gF
import gmml.utils as utils


class GMMLayer(cb.WeightedCodeBook, cb.GaussianCodeBook):
    """
    Full-covariance Gaussian mixture layer.

    Calling the layer (forward/__call__) returns component-conditional
    probabilities; get_log_joint_probs() and sample() are plain methods
    that bypass forward hooks.
    """

    def _init_full_factors(
        self: Self,
        size: int,
        dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Parameter:
        """
        Initializes the unconstrained full_factors parameter.

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized unconstrained full_factors
            parameter.
        """

        return nn.Parameter(
            torch.zeros(size, dim, dim, device=device, dtype=dtype),
            requires_grad=True
        )

    def __init__(
        self: Self,
        size: int,
        dim: int,
        *args: Any,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes a full-covariance Gaussian mixture layer.

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base module.
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(
            size, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self._full_factors = self._init_full_factors(
            size, dim, device=device, dtype=dtype
        )

    def reset_parameters(self: Self) -> None:
        """
        Reinitializes all learnable parameters of the layer in place.

        Args:
            None.

        Returns:
            None: This method returns after reinitializing the parameters.
        """

        super().reset_parameters()
        self._full_factors.data.copy_(
            self._init_full_factors(
                self.size,
                self.dim,
                device=self._full_factors.device,
                dtype=self._full_factors.dtype
            )
        )

    @property
    def full_factors(self: Self) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Upper-triangular Cholesky factors of the mixture
            components.
        """

        return utils.set_softplus_diag(self._full_factors).triu()

    def get_log_joint_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the logarithm of the joint probability distributions.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return gF.get_log_joint_probs(
            inpt, self.means, self.full_factors, self.norm_logits
        )

    def sample(self: Self, *sample_batch_shape: int) -> Tensor:
        """
        Samples each component.

        Args:
            *sample_batch_shape: Optional leading batch shape for the
                generated samples.

        Returns:
            Tensor: Samples drawn from each mixture component.
        """

        return gF.get_samples(
            self.means, self.full_factors, *sample_batch_shape
        )


class DiagGMMLayer(cb.WeightedCodeBook, cb.GaussianCodeBook):
    """
    Diagonal-constrained Gaussian mixture layer.

    Calling the layer (forward/__call__) returns component-conditional
    probabilities; get_log_joint_probs() and sample() are plain methods
    that bypass forward hooks.
    """

    def _init_diag_factors(
        self: Self,
        size: int,
        dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Parameter:
        """
        Initializes the unconstrained _diag_factors parameter.

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized unconstrained _diag_factors
            parameter.
        """

        return nn.Parameter(
            torch.zeros(size, dim, device=device, dtype=dtype),
            requires_grad=True
        )

    def __init__(
        self: Self,
        size: int,
        dim: int,
        *args: Any,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes a diagonal-covariance Gaussian mixture layer.

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base module.
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(
            size, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self._diag_factors = self._init_diag_factors(
            size, dim, device=device, dtype=dtype
        )

    def reset_parameters(self: Self) -> None:
        """
        Reinitializes all learnable parameters of the layer in place.

        Args:
            None.

        Returns:
            None: This method returns after reinitializing the parameters.
        """

        super().reset_parameters()
        self._diag_factors.data.copy_(
            self._init_diag_factors(
                self.size,
                self.dim,
                device=self._diag_factors.device,
                dtype=self._diag_factors.dtype
            )
        )

    @property
    def diag_factors(self: Self) -> Tensor:
        """
        Returns the diagonally-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Positive diagonal factors of the mixture components.
        """

        return F.softplus(self._diag_factors) / log(2.0)

    @property
    def full_factors(self: Self) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Diagonal Cholesky factors of the mixture components.
        """

        return self.diag_factors.diag_embed()

    def get_log_joint_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the logarithm of the joint probability distributions.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return gF.get_diag_log_joint_probs(
            inpt, self.means, self.diag_factors, self.norm_logits
        )

    def sample(self: Self, *sample_batch_shape: int) -> Tensor:
        """
        Samples each component.

        Args:
            *sample_batch_shape: Optional leading batch shape for the
                generated samples.

        Returns:
            Tensor: Samples drawn from each mixture component.
        """

        return gF.get_diag_samples(
            self.means, self.diag_factors, *sample_batch_shape
        )


class IsoGMMLayer(cb.WeightedCodeBook, cb.GaussianCodeBook):
    """
    Isometric-covariance Gaussian mixture layer.

    Calling the layer (forward/__call__) returns component-conditional
    probabilities; get_log_joint_probs() and sample() are plain methods
    that bypass forward hooks.
    """

    def _init_iso_factors(
        self: Self,
        size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Parameter:
        """
        Initializes the unconstrained _iso_factors parameter.

        Args:
            size: Number of mixture components.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized unconstrained _iso_factors
            parameter.
        """

        return nn.Parameter(
            torch.zeros(size, device=device, dtype=dtype),
            requires_grad=True
        )

    def __init__(
        self: Self,
        size: int,
        dim: int,
        *args: Any,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes an isometric-covariance Gaussian mixture layer.

        Args:
            size: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base module.
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(
            size, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self._iso_factors = self._init_iso_factors(
            size, device=device, dtype=dtype
        )

    def reset_parameters(self: Self) -> None:
        """
        Reinitializes all learnable parameters of the layer in place.

        Args:
            None.

        Returns:
            None: This method returns after reinitializing the parameters.
        """

        super().reset_parameters()
        self._iso_factors.data.copy_(
            self._init_iso_factors(
                self.size,
                device=self._iso_factors.device,
                dtype=self._iso_factors.dtype
            )
        )

    @property
    def iso_factors(self: Self) -> Tensor:
        """
        Returns the isometrically-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Positive isometric factors of the mixture components.
        """

        return F.softplus(self._iso_factors) / log(2.0)

    @property
    def diag_factors(self: Self) -> Tensor:
        """
        Returns the diagonally-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Diagonal factors derived from the isometric parameters.
        """

        return self.iso_factors.unsqueeze(-1).expand(-1, self.dim)

    @property
    def full_factors(self: Self) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Diagonal Cholesky factors of the mixture components.
        """

        return self.diag_factors.diag_embed()

    def get_log_joint_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the logarithm of the joint probability distributions under an
        isometric constraint.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return gF.get_iso_log_joint_probs(
            inpt, self.means, self.iso_factors, self.norm_logits
        )

    def sample(self: Self, *sample_batch_shape: int) -> Tensor:
        """
        Samples each component.

        Args:
            *sample_batch_shape: Optional leading batch shape for the
                generated samples.

        Returns:
            Tensor: Samples drawn from each mixture component.
        """

        return gF.get_iso_samples(
            self.means, self.iso_factors, *sample_batch_shape
        )
