import os
import unittest

import numpy as np
import pandas as pd

from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from approaches.combined_ranking_regression_trees.regression_error_loss import mean_square_error
from aslib_scenario import ASlibScenario


class CalculateRegressionErrorLoss(unittest.TestCase):
    def setUp(self) -> None:
        scenario = ASlibScenario()
        test_scenario_name = "ASP-POTASSCO"

        scenario.read_scenario(os.path.join("data", "aslib_data-master", test_scenario_name))
        self.performance_data = scenario.performance_data.values

    def test_calculate_regression_error_loss_trivial(self):
        trivial_example = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        error_loss = mean_square_error(trivial_example)
        self.assertEqual(6.25, error_loss)

    def test_non_trivial_example(self):
        error_loss = mean_square_error(self.performance_data)
        self.assertEqual(error_loss, 5645586.041590792)

    def test_one_instance_case(self):
        example = np.array([[1, 2, 3, 4, 5]])
        error_loss = mean_square_error(example)
        self.assertEqual(0, error_loss)
