import math
import sys
from cmath import atanh
from copy import deepcopy
from xml.sax.handler import feature_external_ges

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from approaches.combined_ranking_regression_trees.regression_error_loss import regression_error_loss
from aslib_scenario import ASlibScenario


class BinaryDecisionTree:
    def __init__(self, ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, calculate_label=None, min_sample_leave=3, min_sample_split=7):
        self.left = None
        self.right = None
        self.splitting_feature = None
        self.splitting_value = None
        self.train_scenario = None
        self.fold = None
        self.label = None
        self.min_sample_leave = min_sample_leave
        self.min_sample_split = min_sample_split

        # functions used in training
        self.ranking_loss = ranking_loss
        self.regression_loss = regression_loss
        self.borda_score = borda_score
        self.impact_factor = impact_factor
        self.stopping_criterion = stopping_criterion
        self.calculate_label = calculate_label

    def fit(self, train_scenario: ASlibScenario, fold, amount_of_training_instances, depth=0):
        def scenario_preporcessing():
            self.imputer = SimpleImputer()
            transformed_features = self.imputer.fit_transform(train_scenario.feature_data)

            # standardize feature values
            self.scaler = preprocessing.StandardScaler()
            transformed_features = self.scaler.fit_transform(transformed_features)
            return transformed_features

        def split_by_feature_value(feature_data, splitting_point):
            smaller_feature_instances = feature_data < splitting_point
            bigger_feature_instances = feature_data >= splitting_point

            return smaller_feature_instances, bigger_feature_instances
            # first calculate rows which are above below split point
            # then split the performance_data dataframe

        def calculate_loss(performance_data, rankings, smaller_feature_instances, bigger_feature_instances, smaller_ranking_instances, bigger_ranking_instances):
            def regression_loss():
                if self.impact_factor == 1:
                    return 0
                else:
                    smaller_regression_error_loss = regression_error_loss(smaller_feature_instances) * len(smaller_feature_instances) / len(performance_data)
                    bigger_regression_error_loss = regression_error_loss(bigger_feature_instances) * len(bigger_feature_instances) / len(performance_data)

                    return smaller_regression_error_loss + bigger_regression_error_loss

            def ranking_loss():
                if self.impact_factor == 0:
                    return 0
                else:
                    smaller_ranking_loss = self.ranking_loss(smaller_feature_instances, self.borda_score, smaller_ranking_instances) * len(smaller_feature_instances) / len(performance_data)
                    bigger_ranking_loss = self.ranking_loss(bigger_feature_instances, self.borda_score, bigger_ranking_instances) * len(bigger_feature_instances) / len(performance_data)

                    return smaller_ranking_loss + bigger_ranking_loss

            regression_error = regression_loss()
            ranking_error = ranking_loss()
            return self.impact_factor * ranking_error + (1 - self.impact_factor) * regression_error

        def evaluate_splitting_point(performance_data, feature_data, splitting_point, rankings):
            smaller_instances, bigger_instances = split_by_feature_value(feature_data, splitting_point)

            smaller_feature_instances = performance_data[smaller_instances]
            bigger_feature_instances = performance_data[bigger_instances]
            smaller_ranking_instances = rankings[smaller_instances]
            bigger_ranking_instances = rankings[bigger_instances]

            loss = calculate_loss(performance_data, rankings, smaller_feature_instances, bigger_feature_instances, smaller_ranking_instances, bigger_ranking_instances)
            return loss

        def get_candidate_splitting_points(feature_data: np.array, min_sample_leave):
            def filter_feature_data(feature_data):
                if len(np.unique(feature_data)) <= 1:
                    return []
                split_value = feature_data[min_sample_leave - 2]
                lower_bound = min_sample_leave
                if feature_data[min_sample_leave - 1] == split_value:
                    for val_number, value in enumerate(feature_data):
                        if value > split_value:
                            lower_bound = val_number
                            break
                split_value = feature_data[-min_sample_leave]
                higher_bound = -min_sample_leave
                if feature_data[-min_sample_leave + 1] == split_value:
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

        if depth == 0:
            feature_data = scenario_preporcessing()
        else:
            feature_data = train_scenario.feature_data.values

        performance_data = train_scenario.performance_data.values

        self.fold = fold
        self.instance_number_map = {i: name for i, name in enumerate(train_scenario.instances)}
        self.feature_number_map = {i: name for i, name in enumerate(train_scenario.features)}

        rankings = calculate_ranking_from_performance_data(performance_data)
        if self.stopping_criterion(performance_data, self.min_sample_split):
            self.label = np.average(performance_data, axis=0)
        else:
            best_known_split_loss = math.inf
            counter = 0

            for feature in range(len(train_scenario.features)):
                candidate_splitting_points = get_candidate_splitting_points(feature_data[:, feature], self.min_sample_leave)
                for splitting_point in candidate_splitting_points:
                    split_loss = evaluate_splitting_point(performance_data, feature_data[:, feature], splitting_point, rankings)
                    counter += 1
                    print(f"split number {counter} of {len(candidate_splitting_points) * len(train_scenario.features )}")

                    if split_loss < best_known_split_loss:
                        best_known_split_loss = split_loss
                        best_known_split_feature = feature
                        best_known_split_point = splitting_point

            self.splitting_feature = best_known_split_feature
            self.splitting_value = best_known_split_point
            smaller_instances, bigger_instances = split_by_feature_value(feature_data[:, best_known_split_feature], best_known_split_point)

            smaller_scenario = deepcopy(train_scenario)
            smaller_scenario.feature_data = train_scenario.feature_data[smaller_instances]
            smaller_scenario.performance_data = train_scenario.performance_data[smaller_instances]

            bigger_scenario = deepcopy(train_scenario)
            bigger_scenario.feature_data = train_scenario.feature_data[bigger_instances]
            bigger_scenario.performance_data = train_scenario.performance_data[bigger_instances]

            self.left = BinaryDecisionTree(self.ranking_loss, self.regression_loss, self.borda_score, self.impact_factor, self.stopping_criterion)
            self.right = BinaryDecisionTree(self.ranking_loss, self.regression_loss, self.borda_score, self.impact_factor, self.stopping_criterion)

            self.left.fit(smaller_scenario, self.fold, amount_of_training_instances, depth=depth + 1)

            self.right.fit(bigger_scenario, self.fold, amount_of_training_instances, depth=depth + 1)

        return self

    def predict(self, features: np.array, scenario):
        assert features.ndim == 1, "Must be 1-dimensional"
        if self.label is not None:
            return self.label

        else:
            if features[self.splitting_feature] < self.splitting_value:
                return self.left.predict(features, scenario)
            else:
                return self.left.predict(features, scenario)
