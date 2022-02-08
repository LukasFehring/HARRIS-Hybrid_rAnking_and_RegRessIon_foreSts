import os
import unittest

import numpy as np
import pandas as pd

from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from aslib_scenario import ASlibScenario


class TestRankingTransformer(unittest.TestCase):
    def setUp(self) -> None:
        test_scenario_name = "MAXSAT15-PMS-INDU"
        scenario = ASlibScenario()
        scenario.read_scenario(os.path.join("data", "aslib_data-master", test_scenario_name))
        self.performance_data = scenario.performance_data.iloc[0:5]
        self.performance_data.loc["pms_industrial/aes/mul_8_3.wcnf", "LMHS-C"] = 53.406
        self.performance_data.loc["pms_industrial/aes/mul_8_9.wcnf", "CCLS2akms"] = 1478.927
        self.performance_data.loc["pms_industrial/aes/mul_8_9.wcnf", "LMHS-C"] = 1478.927
        self.performance_data.loc["pms_industrial/aes/mul_8_9.wcnf", "LMHS-I"] = 1478.927

    def test_transform_with_ties(self):
        ranked_data = calculate_ranking_from_performance_data(self.performance_data.values)
        self.assertTrue(
            np.array_equal(
                ranked_data,
                (
                    [
                        [28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0],
                        [28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0],
                        [28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0],
                        [28.0, 28.0, 3.0, 0.5, 0.5, 2.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0],
                        [28.0, 1.0, 28.0, 1.0, 1.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0],
                    ]
                ),
            )
        )
