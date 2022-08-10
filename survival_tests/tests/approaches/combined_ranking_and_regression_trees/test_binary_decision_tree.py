import cProfile
import os
import time
import unittest

from parameterized import parameterized
from sqlalchemy import false

from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from approaches.combined_ranking_regression_trees.borda_score import borda_score_mean_ranking
from approaches.combined_ranking_regression_trees.ranking_loss import *
from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from approaches.combined_ranking_regression_trees.regression_error_loss import mean_square_error
from approaches.combined_ranking_regression_trees.stopping_criteria import loss_under_threshold
from aslib_scenario import ASlibScenario


class BinaryDecisionTreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = ASlibScenario()
        self.fold = 1
        self.amount_of_training_instances = 1
        test_scenario_name = "QBF-2016"
        self.scenario.read_scenario(os.path.join("data", "aslib_data-master", test_scenario_name))

        self.scenario.performance_data = self.scenario.performance_data.iloc[:200]
        self.scenario.feature_data = self.scenario.feature_data.iloc[:200]

        basinc_ranking_loss = squared_hinge_loss
        basic_regression_loss = mean_square_error
        basic_borda_score = borda_score_mean_ranking
        basic_impact_factor = 0.9
        basic_stopping_criterion = loss_under_threshold
        self.binary_decision_tree: BinaryDecisionTree = BinaryDecisionTree(basinc_ranking_loss, basic_regression_loss, basic_borda_score, basic_impact_factor, basic_stopping_criterion)

    def test_basic(self):
        start_time = time.time()
        # todo handle value penalisation
        # todo split dataset

        self.binary_decision_tree.fit(self.scenario, self.fold, self.amount_of_training_instances)
        end_time = time.time()

        print("execution_time", end_time - start_time)
        self.assertIsInstance(self.binary_decision_tree, BinaryDecisionTree)

        prediction_instance = self.scenario.feature_data.iloc[3].values
        predicted_label = self.binary_decision_tree.predict(prediction_instance, None)

    @parameterized.expand([[[0, 0, 1, 1, 2, 4, 5, 6], [2]], [[0, 0, 1, 1, 2], []], [[1, 1, 1, 1, 1, 1, 5, 6], []]])
    def test_get_candidate_splitting_points(self, feature, result):
        self.assertTrue(np.array_equal(self.binary_decision_tree.get_candidate_splitting_points(feature), np.array(result))),


if __name__ == "__main__":
    cProfile.run("unittest.main()", "profiler/profiler.gstat")
