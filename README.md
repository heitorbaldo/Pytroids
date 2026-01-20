# Pytroids

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

------
A small Python package to perform computations on graphic matroids. It is compatible with NumPy (NumPy adjacency matrices as inputs) and NetworkX (NetworkX graphs as inputs).


* Free software: MIT license
* Documentation: TODO


Examples
--------

```python
import numpy as np
import networkx as nx
from pytroids.graphic_matroids import *
from pytroids.dual_matroids import *

A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
])

#Input: NumPy adjacency matrix:
M = GraphicMatroid(A)

print("Edges:", M.edges)
print("Rank:", M.rank(M.edges))
print("Bases:", M.bases())
print("Circuits:", M.circuits())

#Input: NetworkX graph:
G = nx.from_numpy_array(A)
M = GraphicMatroidNX(G)

print("Edges:", M.edges)
print("Rank:", M.rank())
print("Bases:", M.bases())
print("Circuits:", M.circuits())

#Dual matroid:
M_star = DualMatroid(M)

print("Edges:", M.edges)
print("Primal bases:", M.bases())
print("Dual bases:", M_star.bases())
print("Dual rank:", M_star.rank())
```

Warnings
--------
This package does not implement any optimized algorithm for computing graphical matroids. Therefore, it is only efficient for minimal graphs with a few nodes.

Dependencies
--------

* [NumPy](https://github.com/numpy/numpy)
* [NetworkX](https://github.com/networkx/networkx)


