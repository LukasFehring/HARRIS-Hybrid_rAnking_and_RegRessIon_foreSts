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
    def __init__(self, ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, min_sample_leave=3, min_sample_split=7):
        self.left = None
        self.right = None
        self.splitting_feature = None
        self.splitting_value = None
        self.train_scenario = None
        self.fold = None
        self.label = None

        # parameters
        self.min_sample_leave = min_sample_leave
        self.min_sample_split = min_sample_split
        self.impact_factor = impact_factor

        # functions used in training
        self.ranking_loss = ranking_loss
        self.regression_loss = regression_loss
        self.borda_score = borda_score
        self.stopping_criterion = stopping_criterion

    def get_candidate_splitting_points(self, feature_data: np.array):  # todo write a testcase for this. parametrized with a bunch of different values
        def filter_feature_data(feature_data):
            lower_split = self.min_sample_leave

            finished = len(np.unique(feature_data[:lower_split])) == self.min_sample_leave

            if not finished:
                lower_split = lower_split - 1
                last_val = feature_data[lower_split]
                while not finished:
                    old_val = last_val
                    lower_split += 1
                    try:
                        last_val = feature_data[lower_split]
                        if last_val > old_val:
                            finished = True
                    except LookupError as e:
                        return np.array([], np.float64)

            higher_split = -self.min_sample_leave
            finished = len(np.unique(feature_data[higher_split:])) == self.min_sample_leave

            if not finished:
                last_val = feature_data[-higher_split]
                while not finished:
                    old_val = last_val
                    higher_split -= 1
                    try:
                        last_val = feature_data[higher_split]
                        if last_val < old_val:
                            finished = True
                    except LookupError as e:
                        return np.array([], np.float64)

            if feature_data[lower_split] < feature_data[higher_split]:
                return feature_data[lower_split:higher_split]

            else:
                return np.array([], np.float64)

        feature_data.sort()
        feature_data = filter_feature_data(feature_data)
        return feature_data

    def get_name(self):
        return f"BinaryDecisionTree with rankings loss {self.ranking_loss}, impact factor {self.impact_factor}, and borda score {self.borda_score}"

    def fit(self, train_scenario: ASlibScenario, fold, amount_of_training_instances, depth=0):
        def scenario_preporcessing():
            self.imputer = SimpleImputer()
            transformed_features = self.imputer.fit_transform(train_scenario.feature_data.values)

            # self.scaler = preprocessing.StandardScaler()
            # transformed_features = self.scaler.fit_transform(transformed_features) #todo einkommentieren wenn ich das ganze auswerte
            return transformed_features

        def split_by_feature_value(feature_data, splitting_point):
            smaller_feature_instances = feature_data < splitting_point
            bigger_feature_instances = feature_data >= splitting_point

            return smaller_feature_instances, bigger_feature_instances

        def calculate_loss(performance_data, smaller_performance_instances, bigger_performance_instances, smaller_ranking_instances, bigger_ranking_instances):
            def regression_loss():
                if self.impact_factor == 1:
                    return 0
                else:
                    smaller_regression_error_loss = regression_error_loss(smaller_performance_instances) * len(smaller_performance_instances) / len(performance_data)
                    bigger_regression_error_loss = regression_error_loss(bigger_performance_instances) * len(bigger_performance_instances) / len(performance_data)

                    return smaller_regression_error_loss + bigger_regression_error_loss

            def ranking_loss():
                if self.impact_factor == 0:
                    return 0
                else:
                    smaller_ranking_loss = self.ranking_loss(smaller_performance_instances, self.borda_score, smaller_ranking_instances) * len(smaller_performance_instances) / len(performance_data)
                    bigger_ranking_loss = self.ranking_loss(bigger_performance_instances, self.borda_score, bigger_ranking_instances) * len(bigger_performance_instances) / len(performance_data)

                    return smaller_ranking_loss + bigger_ranking_loss

            regression_error = regression_loss()
            ranking_error = ranking_loss()
            return self.impact_factor * ranking_error + (1 - self.impact_factor) * regression_error

        def evaluate_splitting_point(performance_data, feature_data, splitting_point, rankings):
            smaller_instances, bigger_instances = split_by_feature_value(feature_data, splitting_point)

            smaller_performance_instances = performance_data[smaller_instances]
            bigger_performance_instances = performance_data[bigger_instances]
            smaller_ranking_instances = rankings[smaller_instances]
            bigger_ranking_instances = rankings[bigger_instances]

            loss = calculate_loss(performance_data, smaller_performance_instances, bigger_performance_instances, smaller_ranking_instances, bigger_ranking_instances)
            return loss

        if depth == 0:
            feature_data = scenario_preporcessing()
            train_scenario.feature_data = pd.DataFrame(feature_data)
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
                candidate_splitting_points = self.get_candidate_splitting_points(feature_data[:, feature])

                for splitting_point in candidate_splitting_points:
                    split_loss = evaluate_splitting_point(performance_data, feature_data[:, feature], splitting_point, rankings)
                    counter += 1

                    if split_loss < best_known_split_loss:
                        best_known_split_loss = split_loss
                        best_known_split_feature = feature
                        best_known_split_point = splitting_point
            try:
                self.splitting_feature = best_known_split_feature
            except UnboundLocalError:
                self.get_candidate_splitting_points(feature_data[:, feature])
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

        "ALGO DER AM BESTEN IST SOLLTE DEN NIEDRIGSTEM SCORE HABEN  - ich kann hier aber auch andere sachen machen, die dann"
        assert features.ndim == 1, "Must be 1-dimensional"
        if self.label is not None:
            return self.label

        else:
            if features[self.splitting_feature] < self.splitting_value:
                return self.left.predict(features, scenario)
            else:
                return self.right.predict(features, scenario)
