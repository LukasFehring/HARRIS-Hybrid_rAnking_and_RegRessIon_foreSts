import unittest

import numpy as np

from approaches.ranking_and_regression_forest.consensus import average_runtimes


class TestConsenusMechanisms(unittest.TestCase):
    def test_average_runtimes(self):
        data = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])]

        result = average_runtimes(data)

        self.assertTrue(np.array_equal(result, [5, 6, 7, 8]))
