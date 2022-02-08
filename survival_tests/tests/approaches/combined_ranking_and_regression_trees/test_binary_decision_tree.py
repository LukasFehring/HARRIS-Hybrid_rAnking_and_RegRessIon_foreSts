import cProfile
import os
import time
import unittest

from sqlalchemy import false

from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from approaches.combined_ranking_regression_trees.borda_score import borda_score_mean
from approaches.combined_ranking_regression_trees.ranking_loss import *
from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from approaches.combined_ranking_regression_trees.regression_error_loss import regression_error_loss
from approaches.combined_ranking_regression_trees.stopping_criteria import loss_under_threshold
from aslib_scenario import ASlibScenario


class BinaryDecisionTreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = ASlibScenario()
        self.fold = 1
        self.amount_of_training_instances = 1
        test_scenario_name = "QBF-2016"
        self.scenario.read_scenario(os.path.join("data", "aslib_data-master", test_scenario_name))

        self.scenario.performance_data = self.scenario.performance_data
        self.scenario.feature_data = self.scenario.feature_data

    def test_basic(self):
        start_time = time.time()
        # todo handle value penalisation
        # todo split dataset
        basinc_ranking_loss = spearman_rank_correlation
        basic_regression_loss = regression_error_loss
        basic_borda_score = borda_score_mean
        basic_impact_factor = 0.9
        basic_stopping_criterion = loss_under_threshold
        binary_decision_tree = BinaryDecisionTree(basinc_ranking_loss, basic_regression_loss, basic_borda_score, basic_impact_factor, basic_stopping_criterion)
        binary_decision_tree.fit(self.scenario, self.fold, self.amount_of_training_instances)
        end_time = time.time()

        print("execution_time", end_time - start_time)
        self.assertIsInstance(binary_decision_tree, BinaryDecisionTree)


if __name__ == "__main__":
    cProfile.run("unittest.main()", "profiler/profiler.gstat")
