import copy
import logging
import math
import random

import numpy as np
import pandas as pd
from pyexpat import features

from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from aslib_scenario import ASlibScenario

logger = logging.getLogger("run")


class CombinedRankingAndRegressionForest:
    def __init__(self, amount_of_trees, tree: BinaryDecisionTree, consensus, feature_percentage=1) -> None:
        self.trees = [copy.deepcopy(tree) for _ in range(amount_of_trees)]
        self.consensus = consensus
        self.feature_percentage = feature_percentage
        self.list_of_bagged_features = []
        self.amount_of_features = 0

    def get_name(self):
        return f"CombinedForest with {len(self.trees)} trees, {self.consensus.__name__} as consensus, {self.feature_percentage} of features"

    def fit(self, train_scenario: ASlibScenario, fold, amount_of_training_instances, depth=0, do_preprocessing=True):
        self.amount_of_features = math.ceil(len(train_scenario.features) * self.feature_percentage)
        if self.amount_of_features == 0:
            logger.error(f"No features selected for scenario {train_scenario.name}, and fold {fold}, consensus {self.consensus.__name__}, amount of trees {len(self.trees)}")
            raise Exception("No features selected")
        for tree in self.trees:
            selected_instances = copy.deepcopy(train_scenario)

            feature_data = []
            performance_data = []

            #select instances
            for _ in range(len(train_scenario.performance_data)):
                number_of_chosen_instance = np.random.random_integers(0, len(train_scenario.performance_data) - 1)

                feature_data.append(train_scenario.feature_data.iloc[number_of_chosen_instance, :])
                performance_data.append(train_scenario.performance_data.iloc[number_of_chosen_instance, :])

            selected_instances.feature_data = pd.DataFrame(columns=train_scenario.feature_data.columns, data=feature_data)
            selected_instances.performance_data = pd.DataFrame(columns=train_scenario.performance_data.columns, data=performance_data)

            #select features is now done in the binary decision tree
            #selected_features = self.selet_features(train_scenario)
            #selected_features_positions = [train_scenario.features.index(feature) for feature in selected_features]
            #self.list_of_bagged_features.append(selected_features_positions)

            #selected_instances.features = selected_features
            #selected_instances.feature_data = selected_instances.feature_data[selected_features]

            tree.fit(selected_instances, fold, amount_of_training_instances, depth, do_preprocessing, self.amount_of_features)

        return self

    def predict(self, features: np.array, scenario):
        predictions = []
        for tree, selected_features in zip(self.trees, self.list_of_bagged_features):
            tree_features = features[selected_features]
            predictions.append(tree.predict(tree_features, scenario))
        return self.consensus(predictions)


