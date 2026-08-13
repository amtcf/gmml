"""
Codebook abstractions for Gaussian mixture models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, zeros
from torch.nn import Module, Parameter
from torch.nn.functional import softmax, log_softmax

from gmml.util import sample_standard_multivariate_normal


class _CodeBook(Module, ABC):
    """
    Base class for codebooks with learned component vectors.
    """

    def _init_codevector(self: _CodeBook) -> None:
        """
        Initializes the codevectors parameter according to a standard
        multivariate normal distribution.

        Args:
            None.

        Returns:
            None: This function returns only after successful initialization.
        """

        codevector = sample_standard_multivariate_normal(self.size, self.dim)

        self.codevector = Parameter(
            codevector,
            requires_grad=True
        )

    def __init__(
        self: _CodeBook,
        size: int,
        dim: int,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """

        Args:
            size: Number of codebook entries.
            dim: Dimensionality of each code vector.
            *args: Positional arguments forwarded to the base module.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(*args, **kwargs)

        self.size = size
        self.dim = dim

        self._init_codevector()

    @abstractmethod
    def forward(self: _CodeBook, inpt: Tensor) -> Tensor:
        """
        Abstract forward method.

        Args:
            inpt: Input tensor passed to the codebook.

        Returns:
            Tensor: Output tensor produced by the codebook.
        """

        raise NotImplementedError()


class _WeightedCodeBook(_CodeBook, ABC):
    """
    Codebook base class with learnable component weights.
    """

    def _init_logit(self: _WeightedCodeBook) -> None:
        """
        Initializes an equi-weighted logit vector parameter.

        Args:
            None.

        Returns:
            None: This function returns only after successful initialization.
        """

        requires_grad = not self.equi_weighted

        logit = zeros(self.size)

        self.logit = Parameter(logit, requires_grad=requires_grad)

    def __init__(
        self: _WeightedCodeBook,
        size: int,
        dim: int,
        equi_weighted: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """

        Args:
            size: Number of codebook entries.
            dim: Dimensionality of each code vector.
            equi_weighted: Whether to keep all mixture weights equal.
            *args: Positional arguments forwarded to the base module.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            None: This constructor returns after initializing the module.
        """

        super().__init__(size, dim, *args, **kwargs)

        self.equi_weighted = equi_weighted

        self._init_logit()

    @property
    def norm_logit(self: _WeightedCodeBook) -> Tensor:
        """
        Returns the normalized logit vector.

        Args:
            None.

        Returns:
            Tensor: Log-softmax-normalized logits.
        """

        return log_softmax(self.logit, dim=-1)

    @property
    def weight(self: _WeightedCodeBook) -> Tensor:
        """
        Returns weight vector.

        Args:
            None.

        Returns:
            Tensor: Softmax-normalized weights.
        """

        return softmax(self.logit, dim=-1)


class _ProbCodeBook(_CodeBook, ABC):
    """
    Codebook base class that exposes conditional probabilities.
    """

    def forward(self: _ProbCodeBook, inpt: Tensor) -> Tensor:
        """

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Conditional probability distribution over components.
        """

        return self.get_cond_prob(inpt)

    def get_cond_prob(self: _ProbCodeBook, inpt: Tensor) -> Tensor:
        """
        Returns the conditional categorical probability distributions.

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Conditional categorical probability distributions.
        """

        log_cond_prob = self.get_log_cond_prob(inpt)

        return softmax(log_cond_prob, dim=-1)

    def get_cond_entropy(self: _ProbCodeBook, inpt: Tensor) -> Tensor:
        """
        Returns the entropy associated with the conditional categorical
        probability distributions.

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Conditional entropy.
        """

        log_cond_prob = self.get_log_cond_prob(inpt)

        return - (log_cond_prob.exp() * log_cond_prob).sum(-1)

    @abstractmethod
    def get_log_cond_prob(self: _ProbCodeBook, inpt: Tensor) -> Tensor:
        """
        Abstract method for returning logarithm of the conditional categorical
        probability distributions.

        Args:
            inpt: Input tensor passed to the probabilistic codebook.

        Returns:
            Tensor: Log-conditional probability distributions.
        """

        raise NotImplementedError()


class _GaussianCodeBook(_ProbCodeBook, ABC):
    """
    Codebook base class specialized for Gaussian components.
    """

    def get_log_cond_prob(self: _GaussianCodeBook, inpt: Tensor) -> Tensor:
        """
        Returns logarithm of the conditional categorical probability
        distributions.

        Args:
            inpt: Input tensor passed to the Gaussian codebook.

        Returns:
            Tensor: Log-conditional probability distributions.
        """

        log_joint_prob = self.get_log_joint_prob(inpt)

        return log_softmax(log_joint_prob, dim=-1)

    @property
    def mean(self: _GaussianCodeBook) -> Tensor:
        """
        Returns the mean-vectors.

        Args:
            None.

        Returns:
            Tensor: Mean vectors of the Gaussian components.
        """

        return self.codevector

    @property
    def precision(self: _GaussianCodeBook) -> Tensor:
        """
        Returns the precision matrices.

        Args:
            None.

        Returns:
            Tensor: Precision matrices of the Gaussian components.
        """

        return self.full_fct.mT @ self.full_fct

    @property
    @abstractmethod
    def full_fct(self: _GaussianCodeBook) -> Tensor:
        """
        Returns the Cholesky factors.

        Args:
            None.

        Returns:
            Tensor: Cholesky factors of the Gaussian components.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_log_joint_prob(self: _GaussianCodeBook, inpt: Tensor) -> Tensor:
        """
        Abstract method for returning the logarithm of the joint probability
        distributions.

        Args:
            inpt: Input tensor passed to the Gaussian codebook.

        Returns:
            Tensor: Log joint probability distributions.
        """

        raise NotImplementedError()
