'''
unittest / numpy.testing
'''

import unittest
from unittest import TestCase
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pytroids.graphic_matroids import *
from pytroids.dual_matroids import *


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

A = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 0]
])

M = GraphicMatroid(A)
M_star = DualMatroid(M)

B_star = [{(1, 2), (1, 3)}, {(1, 2), (0, 3)}, {(0, 2), (1, 3)}, {(0, 2), (0, 3)}, {(0, 1), (1, 3)}, {(0, 1), (1, 2)}, {(0, 1), (0, 3)}, {(0, 1), (0, 2)}]




class DualMatroid(TestCase):
    
    def test_DualMatroid(self):
        self.assertEqual(M_star.bases(), B_star)


if __name__ == '__main__':
    unittest.main()

    
