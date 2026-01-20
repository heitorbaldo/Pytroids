'''
Graphic Matroids.
NetworkX-compatible.
'''

import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import networkx as nx


__all__ = [
    "UnionFind",
    "GraphicMatroid",
    "UnionFindNX",
    "GraphicMatroidNX",
]

#-------------- Input: NumPy Adjacency Matrix --------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

class GraphicMatroid:
    def __init__(self, A: np.ndarray):
        if not isinstance(A, np.ndarray):
            raise TypeError("Adjacency matrix must be a NumPy array")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix must be square")

        self.A = A
        self.n = A.shape[0]
        self.edges = self._extract_edges()

    def _extract_edges(self):
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.A[i, j] != 0:
                    edges.append((i, j))
        return edges

    def is_independent(self, edge_set):
        """Acyclicity test (forest).
        """
        uf = UnionFind(self.n)
        for u, v in edge_set:
            if not uf.union(u, v):
                return False
        return True

    def rank(self, edge_set=None):
        """Matroid rank.
        """
        if edge_set is None:
            edge_set = self.edges

        max_rank = 0
        for r in range(len(edge_set) + 1):
            for subset in itertools.combinations(edge_set, r):
                if self.is_independent(subset):
                    max_rank = max(max_rank, r)
        return max_rank

    def bases(self):
        """All spanning forests
        """
        r = self.rank()
        return [
            set(B) for B in itertools.combinations(self.edges, r)
            if self.is_independent(B)
        ]

    def circuits(self):
        """Minimal cycles
        """
        circuits = []
        for r in range(1, len(self.edges) + 1):
            for subset in itertools.combinations(self.edges, r):
                if not self.is_independent(subset):
                    if all(
                        self.is_independent(set(subset) - {e})
                        for e in subset
                    ):
                        circuits.append(set(subset))
        return circuits


#-------------- Input: NetworkX Graph --------------
#NetworkX-compatible version


class UnionFindNX:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # cycle detected
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True


class GraphicMatroidNX:
    def __init__(self, G):
        """
        G: networkx.Graph or networkx.MultiGraph.
        """
        if not isinstance(G, (nx.Graph, nx.MultiGraph)):
            raise TypeError("Input must be a NetworkX Graph or MultiGraph")

        self.G = G
        self.nodes = list(G.nodes)

        # Preserve edge identity (important for MultiGraph)
        if isinstance(G, nx.MultiGraph):
            self.edges = list(G.edges(keys=True))
        else:
            self.edges = list(G.edges())

    def _subgraph_from_edges(self, edge_set):
        """Build a NetworkX graph from a given edge set.
        """
        if isinstance(self.G, nx.MultiGraph):
            H = nx.MultiGraph()
            H.add_nodes_from(self.nodes)
            for u, v, k in edge_set:
                H.add_edge(u, v, key=k)
        else:
            H = nx.Graph()
            H.add_nodes_from(self.nodes)
            H.add_edges_from(edge_set)
        return H

    def is_independent(self, edge_set):
        """Independence oracle using nx.is_forest.
        """
        H = self._subgraph_from_edges(edge_set)
        return nx.is_forest(H)

    def independent_sets(self):
        """All independent sets (forests).
        """
        indep = []
        for r in range(len(self.edges) + 1):
            for subset in itertools.combinations(self.edges, r):
                if self.is_independent(subset):
                    indep.append(set(subset))
        return indep

    def rank(self, edge_set=None):
        """
        Rank = size of largest independent subset.
        Defaults to full ground set.
        """
        if edge_set is None:
            edge_set = self.edges

        max_rank = 0
        for r in range(len(edge_set) + 1):
            for subset in itertools.combinations(edge_set, r):
                if self.is_independent(subset):
                    max_rank = max(max_rank, r)
        return max_rank

    def bases(self):
        """All bases (spanning forests/spanning trees if connected).
        """
        r = self.rank()
        return [
            set(S) for S in itertools.combinations(self.edges, r)
            if self.is_independent(S)
        ]

    def circuits(self):
        """All circuits (minimal dependent sets = simple cycles).
        """
        circuits = []
        for r in range(1, len(self.edges) + 1):
            for subset in itertools.combinations(self.edges, r):
                if not self.is_independent(subset):
                    # minimal dependent?
                    minimal = True
                    for e in subset:
                        if not self.is_independent(set(subset) - {e}):
                            minimal = False
                            break
                    if minimal:
                        circuits.append(set(subset))
        return circuits


