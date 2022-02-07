import os
import unittest

from approaches.combined_ranking_regression_trees.borda_score import borda_score_mean
from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from aslib_scenario import ASlibScenario


class TestRankingTransformer(unittest.TestCase):
    def setUp(self) -> None:
        test_scenario_name = "MAXSAT15-PMS-INDU"
        scenario = ASlibScenario()
        scenario.read_scenario(os.path.join("data", "aslib_data-master", test_scenario_name))
        performance_data = scenario.performance_data.iloc[0:20]
        self.ranking_data = calculate_ranking_from_performance_data(performance_data.values)

    def test_borda_score_mean(self):
        borda_ranking = borda_score_mean(self.ranking_data)
        self.assertEqual(
            borda_ranking,
            {
                19: 0,
                18: 1,
                4: 2,
                5: 3,
                3: 4,
                6: 5,
                8: 6,
                13: 7,
                22: 8,
                23: 9,
                10: 10,
                24: 11,
                7: 12,
                21: 13,
                9: 14,
                25: 15,
                20: 16,
                2: 17,
                15: 18,
                14: 19,
                17: 20,
                16: 21,
                0: 22,
                1: 23,
                11: 20,
                12: 20,
                26: 20,
                27: 20,
                28: 20,
            },
        )
