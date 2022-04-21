import os
import unittest
from ast import Assert

from approaches.combined_ranking_regression_trees.borda_score import *
from approaches.combined_ranking_regression_trees.evaulation_metrices import NDCG
from approaches.combined_ranking_regression_trees.ranking_loss import modified_position_error, spearman_footrule, spearman_rank_correlation, squared_hinge_loss, number_of_discordant_pairs
from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from aslib_scenario import ASlibScenario


class ModifiedPositionTest(unittest.TestCase):
    def setUp(self):
        test_scenario_name = "MAXSAT15-PMS-INDU"
        scenario = ASlibScenario()
        scenario.read_scenario(os.path.join("data", "aslib_data-master", test_scenario_name))
        self.performance_data = scenario.performance_data.iloc[0:10].values
        self.ranking_data = calculate_ranking_from_performance_data(self.performance_data)

    def test_modified_position_error(self):
        error = modified_position_error(self.performance_data, borda_score_mean_performance, self.ranking_data)
        self.assertEqual(error, 13.983400000000003)

    def test_spearman_rank_correlation(self):
        error = spearman_rank_correlation(self.performance_data, borda_score_mean_performance, self.ranking_data)
        self.assertEqual(error, 6009.8)

    def test_spearman_footrule(self):
        error = spearman_footrule(self.performance_data, borda_score_mean_performance, self.ranking_data)
        self.assertEqual(error, 330.8)

    def test_squared_hinge_loss(self):
        error = squared_hinge_loss(self.performance_data, borda_score_mean_performance, self.ranking_data)
        self.assertEqual(error, 11286.520025522042)
        
    def test_number_of_discordant_pairs(self):
        error = number_of_discordant_pairs(self.performance_data, borda_score_mean_performance, self.ranking_data)
        print()