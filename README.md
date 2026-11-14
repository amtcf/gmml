# Gaussian Mixture Modeling Layer

This package contains a PyTorch implementation of a
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

The package provides four layers, which differ in how the Cholesky factors
of the precision matrices are constrained:

- `FullGMMLayer`: general upper-triangular factors (`full_factors`),
- `DiagGMMLayer`: diagonal factors (`diag_factors`),
- `IsoGMMLayer`: isometric factors, one scale per component
  (`iso_factors`),
- `SharedIsoGMMLayer`: a single isometric factor shared by all components.

Every layer takes the number of components and the dimensionality, plus
the optional `equi_weighted` (default `False`), which keeps all mixture
weights equal by fixing the logits to zero, and the usual `device` and
`dtype` arguments.

A **Gaussian mixture modeling layer** can be instantiated in the following way:

```python
from torch import rand

from gmml import FullGMMLayer

dim = 3
num_components = 7
batch_size = 4

gmm_layer = FullGMMLayer(num_components, dim)

inpt = rand(batch_size, dim)
probs = gmm_layer(inpt)
```

The **Gaussian mixture modeling layer** fits the input data by maximizing the
log-likelihood, inducing a gradient on both its inputs and parameters.

```python
from torch import rand

from gmml import FullGMMLayer

dim = 3
num_components = 7
batch_size = 4

gmm_layer = FullGMMLayer(num_components, dim)

inpt = rand(batch_size, dim)
loss = - gmm_layer.get_log_likelihoods(inpt).mean()
loss.backward()
```

The `gmml.functional` namespace contains implementations of differentiable
procedures relevant to general Gaussian mixture modeling, including: joint
probability and pair-wise Kullback-Leibler divergence determinations,
differentiable sampling, and joint entropy upper-bounding.

## Numerical accuracy of the diagonal and isometric layers

To save memory and time, `DiagGMMLayer`, `IsoGMMLayer` and
`SharedIsoGMMLayer` (and `get_diag_log_joint_probs`,
`get_iso_log_joint_probs`) evaluate the squared Mahalanobis distance between
an input $x$ and a component mean $\mu$ by expanding it,

$$
\sum_{k = 1}^{d} \gamma_k ^ 2 (x_k - \mu_k) ^ 2 =
\sum_{k = 1}^{d} \gamma_k ^ 2 x_k ^ 2 -
2 \sum_{k = 1}^{d} \gamma_k ^ 2 x_k \mu_k +
\sum_{k = 1}^{d} \gamma_k ^ 2 \mu_k ^ 2 ,
$$

where $\gamma$ is the diagonal (or isometric) Cholesky factor. The right-hand
side is computed with matrix products, so memory drops from
$O(b \times n \times d)$ to $O(b \times n)$ for a batch of $b$ inputs and $n$
components. In exchange, the three terms on the right can be much larger than
the distance itself, and subtracting them cancels most of their significant
digits. The rounding error in the distance is of the order of

$$
\varepsilon \sum_{k = 1}^{d} \gamma_k ^ 2 \left(x_k ^ 2 + \mu_k ^ 2\right) ,
$$

where $\varepsilon \approx 6 \times 10 ^ {-8}$ in `float32` and
$\varepsilon \approx 10 ^ {-16}$ in `float64`. The error depends on the size of
$x$ and $\mu$, not on the size of the distance itself. Inputs and means are
centered on the average of the means before expanding, which helps when the
components are clustered away from the origin, but not when the components
are spread far apart.

The error is negligible for standardized inputs and moderate factors. It
becomes significant when an input lies close to a component mean that is
far from the other components, and that component is tight (large
$\gamma$), which is the typical state of a well-fitted mixture on unscaled
data. For example, in `float32` with $d = 32$, means of norm about $600$ and
$\gamma \approx 30$, the log-probability of the component containing the
input was measured to be off by about $10 ^ 2$, against $10 ^ {-5}$ when
differences $x - \mu$ are formed explicitly. When moderate scales are not
guaranteed:

- standardize (or normalize) the inputs before the layer,
- use `float64` (`dtype=torch.float64`), or
- use `FullGMMLayer`, which forms the differences $x - \mu$ explicitly and is
  not affected (it uses $O(b \times n \times d)$ memory).

The KL divergences, entropy bounds and sampling functions are not affected.

## Installation

To install this package:

```bash
pip install gmml
```

To install it from a clone of the repository:

```bash
pip install .
```

## Testing

The test suite runs against the dependency versions pinned in
`requirements.txt`, which also pins the linter (`ruff`):

```bash
python -m venv .test
.test/bin/pip install -r requirements.txt
.test/bin/pip install .
.test/bin/python -Bm unittest
```
