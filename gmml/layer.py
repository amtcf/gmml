"""
Neural network layers for Gaussian mixture models.

Each layer holds an unconstrained Cholesky factors parameter, accessed
through get__factors, set__factors, reset__factors and the factors
property, whose shape depends on the parametrization:

- FullGMMLayer: _full_factors, of shape (num_components, dim, dim).
- DiagGMMLayer: _diag_factors, of shape (num_components, dim).
- IsoGMMLayer: _iso_factors, of shape (num_components,).
- SharedIsoGMMLayer: _shared_iso_factors, of shape (1,).

Note that "factors" is the raw parameter, while the unprefixed properties
(full_factors, diag_factors, iso_factors, shared_iso_factors) are the
constrained, positive Cholesky factors derived from it.
"""

from math import log
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import gmml.codebook as cb
import gmml.functional as gF
import gmml.utils as utils
import gmml.validation as val


class FullGMMLayer(cb.CholeskyGMMCodeBook):
    """
    Full-covariance Gaussian mixture layer.

    Calling the layer (forward/__call__) returns component-conditional
    probabilities; get_log_joint_probs() and sample() are plain methods
    that bypass forward hooks.
    """

    def _init__full_factors(
        self: Self,
        num_components: int,
        dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """
        Initializes the unconstrained _full_factors parameter to zero, which
        corresponds to identity Cholesky factors.

        Args:
            num_components: Number of mixture components.
            dim: Dimensionality of each component.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized unconstrained _full_factors
            parameter.
        """

        return nn.Parameter(
            torch.zeros(num_components, dim, dim, device=device, dtype=dtype),
            requires_grad=True,
        )

    def __init__(
        self: Self,
        num_components: int,
        dim: int,
        *args: Any,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a full-covariance Gaussian mixture layer.

        Args:
            num_components: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base codebook,
                i.e. equi_weighted (see **kwargs).
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base codebook,
                i.e. equi_weighted (bool, default False): whether to keep
                all mixture weights equal, fixing the logits to zero.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(
            num_components, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self._full_factors = self._init__full_factors(
            num_components, dim, device=device, dtype=dtype
        )

    def get__factors(self: Self) -> nn.Parameter:
        """
        Returns the unconstrained _full_factors parameter.

        Args:
            None.

        Returns:
            nn.Parameter: The unconstrained _full_factors parameter, of
            shape (num_components, dim, dim).
        """

        return self._full_factors

    def set__factors(self: Self, _full_factors: Tensor) -> None:
        """
        Sets the unconstrained _full_factors parameter.

        Args:
            _full_factors: Tensor containing the new unconstrained
                _full_factors, of shape (num_components, dim, dim).

        Returns:
            None: This method updates the parameter in place.
        """

        val.validate_shape(
            _full_factors, (self.num_components, self.dim, self.dim)
        )

        self._full_factors.data.copy_(_full_factors)

    def reset__factors(self: Self) -> None:
        """
        Resets the unconstrained _full_factors parameter to its initial
        state.

        Args:
            None.

        Returns:
            None: This method updates the parameter in place.
        """

        self._full_factors.data.copy_(
            self._init__full_factors(
                self.num_components,
                self.dim,
                device=self._full_factors.device,
                dtype=self._full_factors.dtype,
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
            components, with positive diagonals.
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

        return gF._get_full_log_joint_probs(
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

        return gF._get_full_samples(
            self.means, self.full_factors, *sample_batch_shape
        )


class DiagGMMLayer(cb.CholeskyGMMCodeBook):
    """
    Diagonal-covariance Gaussian mixture layer.

    Calling the layer (forward/__call__) returns component-conditional
    probabilities; get_log_joint_probs() and sample() are plain methods
    that bypass forward hooks.
    """

    def _init__diag_factors(
        self: Self,
        num_components: int,
        dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """
        Initializes the unconstrained _diag_factors parameter to zero, which
        corresponds to identity Cholesky factors.

        Args:
            num_components: Number of mixture components.
            dim: Dimensionality of each component.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized unconstrained _diag_factors
            parameter.
        """

        return nn.Parameter(
            torch.zeros(num_components, dim, device=device, dtype=dtype),
            requires_grad=True,
        )

    def __init__(
        self: Self,
        num_components: int,
        dim: int,
        *args: Any,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a diagonal-covariance Gaussian mixture layer.

        Args:
            num_components: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base codebook,
                i.e. equi_weighted (see **kwargs).
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base codebook,
                i.e. equi_weighted (bool, default False): whether to keep
                all mixture weights equal, fixing the logits to zero.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(
            num_components, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self._diag_factors = self._init__diag_factors(
            num_components, dim, device=device, dtype=dtype
        )

    def get__factors(self: Self) -> nn.Parameter:
        """
        Returns the unconstrained _diag_factors parameter.

        Args:
            None.

        Returns:
            nn.Parameter: The unconstrained _diag_factors parameter, of
            shape (num_components, dim).
        """

        return self._diag_factors

    def set__factors(self: Self, _diag_factors: Tensor) -> None:
        """
        Sets the unconstrained _diag_factors parameter.

        Args:
            _diag_factors: Tensor containing the new unconstrained
                _diag_factors, of shape (num_components, dim).

        Returns:
            None: This method updates the parameter in place.
        """

        val.validate_shape(_diag_factors, (self.num_components, self.dim))

        self._diag_factors.data.copy_(_diag_factors)

    def reset__factors(self: Self) -> None:
        """
        Resets the unconstrained _diag_factors parameter to its initial
        state.

        Args:
            None.

        Returns:
            None: This method updates the parameter in place.
        """

        self._diag_factors.data.copy_(
            self._init__diag_factors(
                self.num_components,
                self.dim,
                device=self._diag_factors.device,
                dtype=self._diag_factors.dtype,
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
        Returns the logarithm of the joint probability distributions under a
        diagonal constraint.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return gF._get_diag_log_joint_probs(
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

        return gF._get_diag_samples(
            self.means, self.diag_factors, *sample_batch_shape
        )


class IsoGMMLayer(cb.CholeskyGMMCodeBook):
    """
    Isometric-covariance Gaussian mixture layer.

    Calling the layer (forward/__call__) returns component-conditional
    probabilities; get_log_joint_probs() and sample() are plain methods
    that bypass forward hooks.
    """

    def _init__iso_factors(
        self: Self,
        num_components: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """
        Initializes the unconstrained _iso_factors parameter to zero, which
        corresponds to identity Cholesky factors.

        Args:
            num_components: Number of mixture components.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized unconstrained _iso_factors
            parameter.
        """

        return nn.Parameter(
            torch.zeros(num_components, device=device, dtype=dtype),
            requires_grad=True,
        )

    def __init__(
        self: Self,
        num_components: int,
        dim: int,
        *args: Any,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an isometric-covariance Gaussian mixture layer.

        Args:
            num_components: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base codebook,
                i.e. equi_weighted (see **kwargs).
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base codebook,
                i.e. equi_weighted (bool, default False): whether to keep
                all mixture weights equal, fixing the logits to zero.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(
            num_components, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self._iso_factors = self._init__iso_factors(
            num_components, device=device, dtype=dtype
        )

    def get__factors(self: Self) -> nn.Parameter:
        """
        Returns the unconstrained _iso_factors parameter.

        Args:
            None.

        Returns:
            nn.Parameter: The unconstrained _iso_factors parameter, of
            shape (num_components,).
        """

        return self._iso_factors

    def set__factors(self: Self, _iso_factors: Tensor) -> None:
        """
        Sets the unconstrained _iso_factors parameter.

        Args:
            _iso_factors: Tensor containing the new unconstrained
                _iso_factors, of shape (num_components,).

        Returns:
            None: This method updates the parameter in place.
        """

        val.validate_shape(_iso_factors, (self.num_components,))

        self._iso_factors.data.copy_(_iso_factors)

    def reset__factors(self: Self) -> None:
        """
        Resets the unconstrained _iso_factors parameter to its initial state.

        Args:
            None.

        Returns:
            None: This method updates the parameter in place.
        """

        self._iso_factors.data.copy_(
            self._init__iso_factors(
                self.num_components,
                device=self._iso_factors.device,
                dtype=self._iso_factors.dtype,
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
            Tensor: Diagonal factors derived from the isometric factors.
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

        return gF._get_iso_log_joint_probs(
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

        return gF._get_iso_samples(
            self.means, self.iso_factors, *sample_batch_shape
        )


class SharedIsoGMMLayer(cb.CholeskyGMMCodeBook):
    """
    Isometric-covariance Gaussian mixture layer, with a single isometric
    factor shared by all components.

    Calling the layer (forward/__call__) returns component-conditional
    probabilities; get_log_joint_probs() and sample() are plain methods
    that bypass forward hooks.
    """

    def _init__shared_iso_factors(
        self: Self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """
        Initializes the unconstrained _shared_iso_factors parameter to zero,
        which corresponds to identity Cholesky factors.

        Args:
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized unconstrained _shared_iso_factors
            parameter.
        """

        return nn.Parameter(
            torch.zeros((1,), device=device, dtype=dtype), requires_grad=True
        )

    def __init__(
        self: Self,
        num_components: int,
        dim: int,
        *args: Any,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a shared isometric-covariance Gaussian mixture layer.

        Args:
            num_components: Number of mixture components.
            dim: Dimensionality of each component.
            *args: Positional arguments forwarded to the base codebook,
                i.e. equi_weighted (see **kwargs).
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base codebook,
                i.e. equi_weighted (bool, default False): whether to keep
                all mixture weights equal, fixing the logits to zero.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(
            num_components, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self._shared_iso_factors = self._init__shared_iso_factors(
            device=device, dtype=dtype
        )

    def get__factors(self: Self) -> nn.Parameter:
        """
        Returns the unconstrained _shared_iso_factors parameter.

        Args:
            None.

        Returns:
            nn.Parameter: The unconstrained _shared_iso_factors parameter, of
            shape (1,).
        """

        return self._shared_iso_factors

    def set__factors(self: Self, _shared_iso_factors: Tensor) -> None:
        """
        Sets the unconstrained _shared_iso_factors parameter.

        Args:
            _shared_iso_factors: Tensor containing the new unconstrained
                _shared_iso_factors, of shape (1,).

        Returns:
            None: This method updates the parameter in place.
        """

        val.validate_shape(_shared_iso_factors, (1,))

        self._shared_iso_factors.data.copy_(_shared_iso_factors)

    def reset__factors(self: Self) -> None:
        """
        Resets the unconstrained _shared_iso_factors parameter to its initial
        state.

        Args:
            None.

        Returns:
            None: This method updates the parameter in place.
        """

        self._shared_iso_factors.data.copy_(
            self._init__shared_iso_factors(
                device=self._shared_iso_factors.device,
                dtype=self._shared_iso_factors.dtype,
            )
        )

    @property
    def shared_iso_factors(self: Self) -> Tensor:
        """
        Returns the shared isometrically-constrained Cholesky factor.

        Args:
            None.

        Returns:
            Tensor: Positive isometric factor shared by all mixture
            components, of shape (1,).
        """

        return F.softplus(self._shared_iso_factors) / log(2.0)

    @property
    def iso_factors(self: Self) -> Tensor:
        """
        Returns the isometrically-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Positive isometric factors of the mixture components,
            expanded from the shared factor.
        """

        return self.shared_iso_factors.expand(self.num_components)

    @property
    def diag_factors(self: Self) -> Tensor:
        """
        Returns the diagonally-constrained Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Diagonal factors derived from the isometric factors.
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
        Returns the logarithm of the joint probability distributions under a
        shared isometric constraint.

        Args:
            inpt: Input tensor passed to the mixture model.

        Returns:
            Tensor: Log joint probabilities for each component.
        """

        return gF._get_iso_log_joint_probs(
            inpt, self.means, self.shared_iso_factors, self.norm_logits
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

        return gF._get_iso_samples(
            self.means, self.shared_iso_factors, *sample_batch_shape
        )
