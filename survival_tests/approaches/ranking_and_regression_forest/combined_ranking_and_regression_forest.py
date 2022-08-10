import copy
import logging

import numpy as np
import pandas as pd

from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from aslib_scenario import ASlibScenario


class CombinedRankingAndRegressionForest:
    def __init__(self, amount_of_trees, tree: BinaryDecisionTree, consensus) -> None:
        self.trees = [copy.deepcopy(tree) for _ in range(amount_of_trees)]
        self.consensus = consensus

    def get_name(self):
        return f"CombinedForest with {len(self.trees)} trees, {self.consensus.__name__} as consensus"

    def fit(self, train_scenario: ASlibScenario, fold, amount_of_training_instances, depth=0, do_preprocessing=True):
        for treenumber, tree in enumerate(self.trees):
            logging.info(f"Fitting tree {treenumber}")

            selected_instances = copy.deepcopy(train_scenario)

            feature_data = []
            performance_data = []

            for _ in range(len(train_scenario.performance_data)):
                number_of_chosen_instance = np.random.random_integers(0, len(train_scenario.performance_data) - 1)

                feature_data.append(train_scenario.feature_data.iloc[number_of_chosen_instance, :])
                performance_data.append(train_scenario.performance_data.iloc[number_of_chosen_instance, :])

            # todo bagging for features
                
            selected_instances.feature_data = pd.DataFrame(columns=train_scenario.feature_data.columns, data=feature_data)
            selected_instances.performance_data = pd.DataFrame(columns=train_scenario.performance_data.columns, data=performance_data)
            tree.fit(selected_instances, fold, amount_of_training_instances, depth, do_preprocessing)

        return self

    def predict(self, features: np.array, scenario):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(features, scenario))
        return self.consensus(predictions)
