"""
Codebook abstractions for Gaussian mixture models.
"""

import abc
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import gmml.utils as utils
import gmml.validation as val


class CodeBook(nn.Module, abc.ABC):
    """
    Base class for codebooks with learned component vectors.
    Codevectors are represented by the 'means' parameter.
    """

    def _init__means(
        self: Self,
        num_components: int,
        dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """
        Initializes the means parameter according to a standard multivariate
        normal distribution.

        Args:
            num_components: Number of codebook entries.
            dim: Dimensionality of each code vector.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized means parameter.
        """

        return nn.Parameter(
            utils.sample_standard_multivariate_normal(
                num_components, dim, device=device, dtype=dtype
            ),
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
        Initializes the codebook means.

        Args:
            num_components: Number of codebook entries.
            dim: Dimensionality of each code vector.
            *args: Positional arguments forwarded to the base module.
            device: Optional device on which to allocate parameters.
            dtype: Optional dtype used to allocate parameters.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(*args, **kwargs)

        self.num_components = num_components
        self.dim = dim

        self._means = self._init__means(
            num_components, dim, device=device, dtype=dtype
        )

    def get__means(self: Self) -> nn.Parameter:
        """
        Returns the means parameter.

        Args:
            None.

        Returns:
            nn.Parameter: The means of the codebook.
        """

        return self._means

    def set__means(self: Self, _means: Tensor) -> None:
        """
        Sets the means parameter to a new value.

        Args:
            _means: Tensor containing the new means for the codebook.

        Returns:
            None: This method updates the means parameter in place.
        """

        val.validate_shape(_means, (self.num_components, self.dim))

        self._means.data.copy_(_means)

    def reset__means(self: Self) -> None:
        """
        Randomly reinitializes the means parameter in place.

        Args:
            None.

        Returns:
            None: This method returns after reinitializing the parameter.
        """

        self._means.data.copy_(
            self._init__means(
                self.num_components,
                self.dim,
                device=self._means.device,
                dtype=self._means.dtype,
            )
        )

    @property
    def means(self: Self) -> Tensor:
        """
        Property to access the means parameter.

        Args:
            None.

        Returns:
            Tensor: The means of the codebook.
        """

        return self._means

    def reset_parameters(self: Self) -> None:
        """
        Randomly resets all learnable parameters of the codebook.

        Args:
            None.

        Returns:
            None: This method updates the parameters in place.
        """

        self.reset__means()

    def extra_repr(self: Self) -> str:
        """
        Returns the extra representation string for this codebook.

        Args:
            None.

        Returns:
            str: String describing the codebook's size and dimensionality.
        """

        return f"num_components={self.num_components}, dim={self.dim}"

    @abc.abstractmethod
    def forward(self: Self, inpt: Tensor) -> Tensor:
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

    When equi_weighted, the logits are fixed to zero and created with
    requires_grad=False, and set__logits refuses to change them. Note that
    module-wide calls such as requires_grad_(True) re-enable their gradient.
    """

    def _init__logits(
        self: Self,
        num_components: int,
        equi_weighted: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """
        Initializes the logits parameter, fixed to zero when equi_weighted.

        Args:
            num_components: Number of codebook entries.
            equi_weighted: Whether to keep all mixture weights equal.
            device: Optional device on which to allocate the parameter.
            dtype: Optional dtype used to allocate the parameter.

        Returns:
            nn.Parameter: The initialized logits parameter.
        """

        return nn.Parameter(
            torch.zeros(num_components, device=device, dtype=dtype),
            requires_grad=not equi_weighted,
        )

    def __init__(
        self: Self,
        num_components: int,
        dim: int,
        equi_weighted: bool = False,
        *args: Any,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the codebook means and mixture logits.

        Args:
            num_components: Number of codebook entries.
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
            num_components, dim, *args, device=device, dtype=dtype, **kwargs
        )

        self.equi_weighted = equi_weighted
        self._logits = self._init__logits(
            num_components, equi_weighted, device=device, dtype=dtype
        )

    def get__logits(self: Self) -> nn.Parameter:
        """
        Returns the current logits parameter.

        Args:
            None.

        Returns:
            nn.Parameter: The current logits of the codebook.
        """

        return self._logits

    def set__logits(self: Self, _logits: Tensor) -> None:
        """
        Sets the logits parameter to the provided tensor.

        Args:
            _logits: Tensor containing the new logits values.

        Returns:
            None: This method updates the logits parameter in place.

        Raises:
            IsEquiWeighted: If the codebook is equi-weighted, in which case
                the logits are fixed to zero.
        """

        if self.equi_weighted:
            raise val.IsEquiWeighted(
                "Cannot set the logits of an equi-weighted codebook"
            )

        val.validate_shape(_logits, (self.num_components,))

        self._logits.data.copy_(_logits)

    def reset__logits(self: Self) -> None:
        """
        Resets the logits parameter to its initial state.

        Args:
            None.

        Returns:
            None: This method updates the logits parameter in place.
        """

        self._logits.data.copy_(
            self._init__logits(
                self.num_components,
                self.equi_weighted,
                device=self._logits.device,
                dtype=self._logits.dtype,
            )
        )

    @property
    def logits(self: Self) -> Tensor:
        """
        Returns the current logits parameter as a property.

        Args:
            None.

        Returns:
            Tensor: The current logits of the codebook.
        """

        return self._logits

    def reset_parameters(self: Self) -> None:
        """
        Reinitializes the means and logits parameters in place.

        Args:
            None.

        Returns:
            None: This method returns after reinitializing the parameters.
        """

        super().reset_parameters()
        self.reset__logits()

    @property
    def norm_logits(self: Self) -> Tensor:
        """
        Returns the normalized logit vector.

        Args:
            None.

        Returns:
            Tensor: Log-softmax-normalized logits.
        """

        return F.log_softmax(self.logits, dim=-1)

    @property
    def weights(self: Self) -> Tensor:
        """
        Returns the weights vector.

        Args:
            None.

        Returns:
            Tensor: Softmax-normalized weights.
        """

        return F.softmax(self.logits, dim=-1)

    def get_balance(self: Self) -> Tensor:
        """
        Returns the balance between the entries of the codebook.
        Higher values indicate more balanced components.

        Args:
            None.

        Returns:
            Tensor: Balance between the codebook entries.
        """

        norm_logits = self.norm_logits

        return -(norm_logits.exp() * norm_logits).sum(-1)

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

        return -(log_cond_probs.exp() * log_cond_probs).sum(-1)

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


class CholeskyGMMCodeBook(WeightedCodeBook, ProbCodeBook, abc.ABC):
    """
    Codebook base class for Gaussian mixtures, parametrized by means,
    mixture logits and Cholesky factors of the component precisions.

    Concrete layers subclass this class only, and hold an unconstrained
    Cholesky factors parameter, whose shape depends on the parametrization,
    accessed through get__factors, set__factors, reset__factors and the
    factors property. The constrained Cholesky factors are derived tensors
    (e.g. full_factors).

    Method resolution order: CholeskyGMMCodeBook, WeightedCodeBook,
    ProbCodeBook, CodeBook. Constructors cooperate through
    super().__init__(num_components, dim, *args, **kwargs), and
    reset_parameters resets the means, the logits and the factors.
    """

    @abc.abstractmethod
    def get__factors(self: Self) -> nn.Parameter:
        """
        Abstract method for returning the unconstrained Cholesky factors
        parameter of the Gaussian components.

        Args:
            None.

        Returns:
            nn.Parameter: Unconstrained Cholesky factors parameter.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def set__factors(self: Self, _factors: Tensor) -> None:
        """
        Abstract method for setting the unconstrained Cholesky factors
        parameter of the Gaussian components.

        Args:
            _factors: Tensor containing the new unconstrained Cholesky
                factors.

        Returns:
            None: This method updates the parameter in place.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def reset__factors(self: Self) -> None:
        """
        Abstract method for resetting the unconstrained Cholesky factors
        parameter of the Gaussian components.

        Args:
            None.

        Returns:
            None: This method updates the parameter in place.
        """

        raise NotImplementedError()

    @property
    def factors(self: Self) -> Tensor:
        """
        Property to access the unconstrained Cholesky factors parameter.
        Its shape depends on the parametrization of the concrete layer.

        Args:
            None.

        Returns:
            Tensor: Unconstrained Cholesky factors of the Gaussian components.
            Not to be confused with the constrained Cholesky factors (e.g.
            full_factors or diag_factors).
        """

        return self.get__factors()

    def reset_parameters(self: Self) -> None:
        """
        Resets all learnable parameters of the Gaussian codebook, including
        the Cholesky factors.

        Args:
            None.

        Returns:
            None: This method updates the parameters in place.
        """

        super().reset_parameters()
        self.reset__factors()

    def get_cond_probs(self: Self, inpt: Tensor) -> Tensor:
        """
        Returns the conditional categorical probability distributions,
        normalizing the log joint probabilities once, since
        softmax(log_softmax(z)) = softmax(z).

        Args:
            inpt: Input tensor passed to the Gaussian codebook.

        Returns:
            Tensor: Conditional categorical probability distributions.
        """

        return F.softmax(self.get_log_joint_probs(inpt), dim=-1)

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

    @abc.abstractmethod
    def sample(self: Self, *sample_batch_shape: int) -> Tensor:
        """
        Abstract method for sampling each Gaussian component.

        Args:
            *sample_batch_shape: Optional leading batch shape for the
                generated samples.

        Returns:
            Tensor: Samples drawn from each mixture component.
        """

        raise NotImplementedError()
