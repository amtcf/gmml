"""
Codebook abstractions for Gaussian mixture models.
"""

from typing import Any, Optional, Self

import abc


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


import gmml.utils as utils


class CodeBook(nn.Module, abc.ABC):
    """
    Base class for codebooks with learned component vectors.
    Codevectors are represented by the 'means' parameter.
    """

    def _init_means(
        self: Self,
        size: int,
        dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Parameter:
        """
        Initializes the means parameter according to a standard multivariate
        normal distribution.

        Args:
            size: Number of codebook entries.
            dim: Dimensionality of each code vector.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            Parameter: The initialized means parameter.
        """

        return nn.Parameter(
            utils.sample_standard_multivariate_normal(
                size, dim, device=device, dtype=dtype
            ),
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
        Initializes the codebook means.

        Args:
            size: Number of codebook entries.
            dim: Dimensionality of each code vector.
            *args: Positional arguments forwarded to the base module.
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(*args, **kwargs)

        self.size = size
        self.dim = dim

        self.means = self._init_means(
            size, dim, device=device, dtype=dtype
        )

    def reset_parameters(self: Self) -> None:
        """
        Reinitializes the means parameter in place.

        Args:
            None.

        Returns:
            None: This method returns after reinitializing the parameter.
        """

        self.means.data.copy_(
            self._init_means(
                self.size,
                self.dim,
                device=self.means.device,
                dtype=self.means.dtype
            )
        )

    def extra_repr(self: Self) -> str:
        """
        Returns the extra representation string for this codebook.

        Args:
            None.

        Returns:
            str: String describing the codebook's size and dimensionality.
        """

        return f"size={self.size}, dim={self.dim}"

    @abc.abstractmethod
    def forward(self: Self, inpt: torch.Tensor) -> torch.Tensor:
        """
        Abstract forward method.

        Args:
            inpt: Input tensor passed to the codebook.

        Returns:
            Tensor: Output tensor produced by the codebook.
        """

        raise NotImplementedError()


class WeightedCodeBook(CodeBook, abc.ABC):
    """
    Codebook base class with learnable component weights.
    """

    def _init_logits(
        self: Self,
        size: int,
        equi_weighted: bool,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Parameter:
        """
        Initializes the logits parameter, fixed to zero when equi_weighted.

        Args:
            size: Number of codebook entries.
            equi_weighted: Whether to keep all mixture weights equal.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            Parameter: The initialized logits parameter.
        """

        return nn.Parameter(
            torch.zeros(size, device=device, dtype=dtype),
            requires_grad=not equi_weighted
        )

    def __init__(
        self: Self,
        size: int,
        dim: int,
        equi_weighted: bool = False,
        *args: Any,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes the codebook means and mixture logits.

        Args:
            size: Number of codebook entries.
            dim: Dimensionality of each code vector.
            equi_weighted: Whether to keep all mixture weights equal.
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

        self.equi_weighted = equi_weighted
        self.logits = self._init_logits(
            size, equi_weighted, device=device, dtype=dtype
        )

    def reset_parameters(self: Self) -> None:
        """
        Reinitializes the means and logits parameters in place.

        Args:
            None.

        Returns:
            None: This method returns after reinitializing the parameters.
        """

        super().reset_parameters()
        self.logits.data.copy_(
            self._init_logits(
                self.size,
                self.equi_weighted,
                device=self.logits.device,
                dtype=self.logits.dtype
            )
        )

    @property
    def norm_logits(self: Self) -> torch.Tensor:
        """
        Returns the normalized logit vector.

        Args:
            None.

        Returns:
            Tensor: Log-softmax-normalized logits.
        """

        return F.log_softmax(self.logits, dim=-1)

    @property
    def weights(self: Self) -> torch.Tensor:
        """
        Returns the weights vector.

        Args:
            None.

        Returns:
            Tensor: Softmax-normalized weights.
        """

        return F.softmax(self.logits, dim=-1)

    def get_balance(self: Self) -> torch.Tensor:
        """
        Returns the balance between the entries of the codebook.
        Higher values indicate more balanced components.

        Args:
            None.

        Returns:
            Tensor: Balance between the codebook entries.
        """

        return - (self.norm_logits.exp() * self.norm_logits).sum(-1)

    def extra_repr(self: Self) -> str:
        """
        Returns the extra representation string for this codebook.

        Args:
            None.

        Returns:
            str: String describing the codebook's size, dimensionality, and
            weighting mode.
        """

        return f"{super().extra_repr()}, equi_weighted={self.equi_weighted}"


class ProbCodeBook(CodeBook, abc.ABC):
    """
    Codebook base class that exposes conditional probabilities.
    """

    def forward(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the conditional probability distribution over components.

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Conditional probability distribution over components.
        """

        return self.get_cond_probs(inpt)

    def get_cond_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the conditional categorical probability distributions.

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Conditional categorical probability distributions.
        """

        return F.softmax(self.get_log_cond_probs(inpt), dim=-1)

    def get_cond_entropies(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the entropy associated with the conditional categorical
        probability distributions.

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Conditional entropy value for each input.
        """

        log_cond_probs = self.get_log_cond_probs(inpt)

        return - (log_cond_probs.exp() * log_cond_probs).sum(-1)

    @abc.abstractmethod
    def get_log_cond_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Abstract method for returning the logarithm of the conditional
        categorical probability distributions over components.

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Log-conditional probability distributions.
        """

        raise NotImplementedError()


class GaussianCodeBook(ProbCodeBook, abc.ABC):
    """
    Codebook base class specialized for Gaussian components.
    """

    def get_log_cond_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the logarithm of the conditional categorical probability
        distributions over components.

        Args:
            inpt: Input tensor passed to the Gaussian codebook.

        Returns:
            Tensor: Log-conditional probability distributions.
        """

        return F.log_softmax(self.get_log_joint_probs(inpt), dim=-1)

    @property
    def precisions(self: Self) -> Tensor:
        """
        Returns the precision matrices.

        Args:
            None.

        Returns:
            Tensor: Precision matrices of the Gaussian components.
        """

        return self.full_factors.mT @ self.full_factors

    @property
    @abc.abstractmethod
    def full_factors(self: Self) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Cholesky factors of the Gaussian components.
        """

        raise NotImplementedError()

    def get_log_likelihoods(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the log-likelihoods.

        Args:
            inpt: Input tensor passed to the Gaussian codebook.

        Returns:
            Tensor: Log-likelihood values.
        """

        return torch.logsumexp(self.get_log_joint_probs(inpt), dim=-1)

    @abc.abstractmethod
    def get_log_joint_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Abstract method for returning the logarithm of the joint probability
        distributions.

        Args:
            inpt: Input tensor passed to the Gaussian codebook.

        Returns:
            Tensor: Log joint probability distributions.
        """

        raise NotImplementedError()
