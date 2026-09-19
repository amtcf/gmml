# Gaussian Mixture Modeling Layer

This package contains a Pytorch implementation of a
**Gaussian mixture modeling layer**.

A **Gaussian mixture modeling layer** maps $d$-dimensional input vectors
onto categorical probability distributions. Such distributions measure the
likelihood of the input vectors belonging to each component of a Gaussian
mixture model. The underlying Gaussian mixture distribution is parametrized
by a collection of: mean-vectors (`means`), Cholesky factors (`*_factors`),
and a logits vector (`norm_logits`). The logits vector defines the relative
importance of each Gaussian component within the mixture, and the collection
of Cholesky factors (`*_factors`) - each resulting from the Cholesky
decomposition of the precision matrices of the Gaussian components - defines
the shape of the Gaussians. Under such parametrization, the computational
complexity of the functional form of the forward and backward calls to this
layer is reduced, leading to more efficient evaluations. The forward and
backward calls of Gaussian mixture modeling layer parametrized by a general
Cholesky factor (`full_factors`) have a computational complexity of
$O(n \times d ^ 2)$, where $n$ is the number of components of the underlying
Gaussian mixture distribution. Moreover, its functional form is differentiable
in all of its inputs and parameters, hence it is autograd compatible.

A **Gaussian mixture modeling layer** can be instantiated in the following way:

```python
from torch import rand

from gmml import GMMLayer

dim = 3
n_components = 7
batch_size = 4

gmml = GMMLayer(n_components, dim)

inpt = rand(batch_size, dim)
probs = gmml(inpt)
```

The **Gaussian mixture modeling layer** fits the input data by maximizing the
log-likelihood, inducing a gradient on both it's inputs and parameters.

```python
from torch import rand

from gmml import GMMLayer, get_log_likelihood

dim = 3
n_components = 7
batch_size = 4

gmml = GMMLayer(n_components, dim)

inpt = rand(batch_size, dim)
ll = get_log_likelihood(inpt, gmml)
```

The `gmml.functional` namespace contains implementations of differentiable
procedures relevant to general Gaussian mixture modeling, including: joint
probability and pair-wise Kullback-Liebler divergence determinations,
differentiable sampling, and joint entropy upper-bounding.

To install this package:

```bash
python -m venv .test
.test/bin/pip install -r requirements.txt
.test/bin/pip install .
```

To run the test suite:

```bash
python -m venv .test
.test/bin/pip install -r requirements.txt
.test/bin/pip install .
.test/bin/python -Bm unittest ./test/*
```
