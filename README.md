## Hyperbolic Manifolds
Hi there! this is my mini-repo on hyperbolic manifolds. Manifolds realized here are:
    - Poincare Ball
    ...


At the moment the following modules are realized aat the basis of `torch`:
    - Linear
    - Embedding
    - Hyperplane (module that outputs unnormalized logits)
    - Wrapper for custom callable function to map them onto the manifold (called `HyperbolicWrapper`)
    ...

## Example
```python
import torch
from torch import nn
from poincare_ball import PoincareBall
from modules import HyperbolicLinear, Hyperplane, HyperbolicParameter
from rsgd import RSGD


data = torch.rand(16, 25)

manifold = PoincareBall()
model = nn.Sequential(
    nn.Linear(25, 10),
    HyperbolicLinear(10, 5, manifold),
    Hyperplane(5, 1, manifold)
)

euclidean_optimizer = torch.optim.SGD(
        [p for p in model.parameters() if not isinstance(p, HyperbolicParameter)], lr=0.001
)
hyperbolic_optimizer = RSGD(
    [p for p in model.parameters() if isinstance(p, HyperbolicParameter)], manifold, lr=0.001
)
euclidean_optimizer.zero_grad()
hyperbolic_optimizer.zero_grad()

logits = model(data)
loss = logits.mean()

loss.backward()
euclidean_optimizer.step()
hyperbolic_optimizer.step()
```