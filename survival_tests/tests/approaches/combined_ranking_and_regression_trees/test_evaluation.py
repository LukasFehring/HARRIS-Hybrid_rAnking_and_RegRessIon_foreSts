import unittest

from approaches.combined_ranking_regression_trees.evaulation_metrices import NDCG


class ModifiedPositionTest(unittest.TestCase):
    def setUp(self):
        self.predicted_scores = [1, 2, 3, 4]
        self.gt_scores = [0, 2, 3, 4]

    def test_ndcg(self):
        ndcg = NDCG().evaluate(self.gt_scores, self.predicted_scores, None, None)
        print(ndcg)
