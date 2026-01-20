'''
This code computes the dual matroid of a given graphic matroid.
NetworkX-compatible.
'''

import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import networkx as nx


__all__ = [
    "DualMatroid",
]

class DualMatroid:
    def __init__(self, matroid):
        """
        matroid: matroid = GraphicMatroid(A) or GraphicMatroid(G)
        A: adjacency matrix.
        G: networkx.Graph or networkx.MultiGraph.
        """
        self.M = matroid
        self.E = set(matroid.edges)

    def bases(self):
        """Dual bases = complements of primal bases.
        """
        return [self.E - B for B in self.M.bases()]

    def is_independent(self, edge_set):
        """Subset of some dual basis.
        """
        S = set(edge_set)
        return any(S.issubset(Bstar) for Bstar in self.bases())

    def rank(self, edge_set=None):
        """
        Rank of a dual matroid:
        r*(S) = |S| - r(E) + r(E \\ S)
        """
        if edge_set is None:
            edge_set = self.E

        S = set(edge_set)
        r_star = len(S) - self.M.rank(self.E) + self.M.rank(self.E - S)
        return r_star


