import unittest

from approaches.combined_ranking_regression_trees.stopping_criteria import max_depth, same_ranking, same_ranking_percentage


class TestStoppingCriteria(unittest.TestCase):
    def setUp(self):
        self.test_performances = [([]), ([(1, 2, 3, 4)]), [(1, 2, 3, 4), (4, 2, 3, 1)]]

    def test_same_ranking(self):
        assert same_ranking(self.test_performances[0], min_sample_split=0) == True
        assert same_ranking(self.test_performances[1], min_sample_split=0) == True
        assert same_ranking(self.test_performances[2], min_sample_split=0) == False

    def test_ranking_percentage(self):
        assert same_ranking_percentage(self.test_performances[0], min_sample_split=0, percentage=0.5) == True
        assert same_ranking_percentage(self.test_performances[1], min_sample_split=0, percentage=0.5) == True
        assert same_ranking_percentage(self.test_performances[2], min_sample_split=0, percentage=0.5) == True
        assert same_ranking_percentage(self.test_performances[2], min_sample_split=0, percentage=0.56) == False

    def test_max_depth(self):
        assert max_depth(self.test_performances[0], min_sample_split=0) == True
        assert max_depth(self.test_performances[1], min_sample_split=0) == True
        assert max_depth(self.test_performances[2], min_sample_split=0, max_depth=5, depth=5) == False
        assert max_depth(self.test_performances[2], min_sample_split=0, max_depth=6, depth=5) == True
