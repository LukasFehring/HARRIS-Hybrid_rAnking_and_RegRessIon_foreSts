import math
import sys
from cmath import atanh
from copy import deepcopy
from xml.sax.handler import feature_external_ges

import numpy as np
import pandas as pd
from pyexpat import features
from sqlalchemy import false

from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from approaches.combined_ranking_regression_trees.regression_error_loss import regression_error_loss
from aslib_scenario import ASlibScenario


class BinaryDecisionTree:
    def __init__(self, ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion):
        # values to be filled in training
        self.is_split = false
        self.left = None
        self.right = None
        self.splitting_feature = None
        self.splitting_value = None
        self.train_scenario = None
        self.fold = None
        self.amount_of_Training_instances = None

        # functions used in training
        self.ranking_loss = ranking_loss
        self.regression_loss = regression_loss
        self.borda_score = borda_score
        self.impact_factor = impact_factor
        self.stopping_criterion = stopping_criterion

    def fit(self, train_scenario: ASlibScenario, fold, amount_of_training_instances, depth=0, min_samples_leave=3):
        def split_by_feature_value(feature_data, splitting_point):
            smaller_feature_instances = feature_data < splitting_point
            bigger_feature_instances = feature_data >= splitting_point

            return smaller_feature_instances, bigger_feature_instances
            # first calculate rows which are above below split point
            # then split the performance_data dataframe

        def calculate_loss(performance_data, smaller_instances, bigger_instances, rankings):
            def regression_loss(performance_data, smaller_instances, bigger_instances):
                if self.impact_factor == 1:
                    return 0
                else:
                    smaller_feature_instances = performance_data[smaller_instances]
                    bigger_feature_instances = performance_data[bigger_instances]
                    smaller_regression_error_loss = regression_error_loss(smaller_feature_instances) * len(smaller_feature_instances) / len(performance_data)
                    bigger_regression_error_loss = regression_error_loss(bigger_feature_instances) * len(bigger_feature_instances) / len(performance_data)

                    return smaller_regression_error_loss + bigger_regression_error_loss

            def ranking_loss(performance_data, smaller_instances, bigger_instances, rankings):
                if self.impact_factor == 0:
                    return 0
                else:
                    smaller_feature_instances = performance_data[smaller_instances]
                    bigger_feature_instances = performance_data[bigger_instances]
                    smaller_ranking_loss = self.ranking_loss(smaller_feature_instances, self.borda_score, rankings) * len(smaller_feature_instances) / len(performance_data)
                    bigger_ranking_loss = self.ranking_loss(bigger_feature_instances, self.borda_score, rankings) * len(bigger_feature_instances) / len(performance_data)

                    return smaller_ranking_loss + bigger_ranking_loss

            regression_error = regression_loss(performance_data, smaller_instances, bigger_instances)
            ranking_error = ranking_loss(performance_data, smaller_instances, bigger_instances, rankings)
            return self.impact_factor * ranking_error + (1 - self.impact_factor) * regression_error

        def evaluate_splitting_point(performance_data, feature_data, splitting_point, rankings):
            smaller_instances, bigger_instances = split_by_feature_value(feature_data, splitting_point)

            loss = calculate_loss(performance_data, smaller_instances, bigger_instances, rankings)
            return loss

        def get_candidate_splitting_points(feature_data: np.array, min_sample_split):
            def filter_feature_data(feature_data):
                if len(np.unique(feature_data)) <= 1:
                    return []
                split_value = feature_data[min_sample_split - 2]
                lower_bound = min_sample_split
                if feature_data[min_sample_split - 1] == split_value:
                    for val_number, value in enumerate(feature_data):
                        if value > split_value:
                            lower_bound = val_number
                            break
                split_value = feature_data[-min_sample_split]
                higher_bound = -min_sample_split
                if feature_data[-min_sample_split + 1] == split_value:
                    for val_number in range(len(feature_data) - 1, 0, -1):
                        value = feature_data[val_number]
                        if value < split_value:
                            higher_bound = val_number
                            break
                return feature_data[lower_bound:higher_bound]

            nan_array: np.array = np.isnan(feature_data)
            feature_data = feature_data[~nan_array]
            feature_data.sort()
            feature_data = filter_feature_data(feature_data)
            return feature_data

        self.fold = fold  # todo what is this paramerer
        # todo what is this parameter
        self.amount_of_Training_instances = amount_of_training_instances

        self.instance_number_map = {i: name for i, name in enumerate(train_scenario.instances)}
        self.feature_number_map = {i: name for i, name in enumerate(train_scenario.features)}

        performance_data = train_scenario.performance_data.values
        feature_data = train_scenario.feature_data.values
        rankings = calculate_ranking_from_performance_data(performance_data)

        if not self.stopping_criterion(performance_data, min_samples_leave):
            best_known_split_loss = math.inf
            counter = 0

            for feature in range(len(train_scenario.features)):
                candidate_splitting_points = get_candidate_splitting_points(feature_data[:, feature], min_samples_leave)
                for splitting_point in candidate_splitting_points:
                    split_loss = evaluate_splitting_point(performance_data, feature_data[:, feature], splitting_point, rankings)
                    counter += 1
                    print(f"split number {counter} of {len(candidate_splitting_points) * len(train_scenario.features )}")

                    if split_loss < best_known_split_loss:
                        best_known_split_loss = split_loss
                        best_known_split_feature = feature
                        best_known_split_point = splitting_point

            smaller_instances, bigger_instances = split_by_feature_value(feature_data[:, best_known_split_feature], best_known_split_point)

            smaller_scenario = deepcopy(train_scenario)
            smaller_scenario.feature_data = train_scenario.feature_data[smaller_instances]
            smaller_scenario.performance_data = train_scenario.performance_data[smaller_instances]

            bigger_scenario = deepcopy(train_scenario)
            bigger_scenario.feature_data = train_scenario.feature_data[bigger_instances]
            bigger_scenario.performance_data = train_scenario.performance_data[bigger_instances]

            self.left = BinaryDecisionTree(self.ranking_loss, self.regression_loss, self.borda_score, self.impact_factor, self.stopping_criterion)
            self.right = BinaryDecisionTree(self.ranking_loss, self.regression_loss, self.borda_score, self.impact_factor, self.stopping_criterion)

            if len(smaller_scenario.performance_data) == 0:
                get_candidate_splitting_points(feature_data[:, best_known_split_feature], min_samples_leave)

            self.left.fit(smaller_scenario, self.fold, self.amount_of_Training_instances, depth=depth + 1)

            self.right.fit(bigger_scenario, self.fold, self.amount_of_Training_instances, depth=depth + 1)

        return self

    def predict(self, scenario):
        pass
