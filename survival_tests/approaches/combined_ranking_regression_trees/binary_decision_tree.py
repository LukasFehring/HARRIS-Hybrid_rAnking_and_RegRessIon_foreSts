import math
import os
import random
from copy import deepcopy
from xml.sax.handler import feature_external_ges

from sklearn import preprocessing

#
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
#
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from approaches.combined_ranking_regression_trees.regression_error_loss import mean_square_error
from aslib_scenario import ASlibScenario


class BinaryDecisionTree:
    def __init__(
        self,
        ranking_loss,
        regression_loss,
        borda_score,
        impact_factor,
        stopping_criterion,
        min_sample_leave=3,
        min_sample_split=7,
        stopping_threshold=None,
        old_threshold=None,
        loss_overview=list(),
        mu=1,
        do_preprocessing=True,
    ):
        self.left = None
        self.right = None
        self.splitting_feature = None
        self.splitting_value = None
        self.train_scenario = None
        self.fold = None
        self.label = None
        self.loss_overview: List = loss_overview

        # parameters
        self.min_sample_leave = min_sample_leave
        self.min_sample_split = min_sample_split
        self.impact_factor = impact_factor
        self.stopping_threshold = stopping_threshold
        self.old_threshold = old_threshold

        # functions used in training
        self.ranking_loss = ranking_loss
        self.regression_loss = regression_loss
        self.borda_score = borda_score
        self.stopping_criterion = stopping_criterion
        self.mu = mu

        self.preprocessing = do_preprocessing

    def get_candidate_splitting_points(self, feature_data: np.array):
        feature_data.sort()
        lower_split_value = feature_data[self.min_sample_leave]
        higher_split_value = feature_data[-self.min_sample_leave]
        return feature_data[(lower_split_value < feature_data) & (feature_data < higher_split_value)]

    def get_name(self):
        return "BinaryDecisionTree_" + str(self.mu)

    def fit(self, train_scenario: ASlibScenario, fold, amount_of_training_instances, depth=0, do_preprocessing=True, amount_of_features_to_use=1):
        def scenario_preporcessing():
            self.imputer = SimpleImputer()
            transformed_features = self.imputer.fit_transform(train_scenario.feature_data.values)
            threshold = train_scenario.algorithm_cutoff_time
            train_scenario.performance_data = train_scenario.performance_data.replace(10 * threshold, self.mu * threshold)
            # self.scaler = preprocessing.MinMaxScaler()
            # transformed_features = self.scaler.fit_transform(transformed_features)
            if self.preprocessing:
                performance_data = preprocessing.MinMaxScaler().fit_transform(train_scenario.performance_data.values)
            else:
                performance_data = train_scenario.performance_data.values
            return pd.DataFrame(transformed_features, columns=train_scenario.feature_data.columns,), pd.DataFrame(
                performance_data,
                columns=train_scenario.performance_data.columns,
            )

        def split_by_feature_value(feature_data, splitting_point):
            smaller_feature_instances = feature_data < splitting_point
            bigger_feature_instances = feature_data >= splitting_point

            return smaller_feature_instances, bigger_feature_instances

        def calculate_loss(performance_data, smaller_performance_instances, bigger_performance_instances, smaller_ranking_instances, bigger_ranking_instances):
            def regression_loss():
                if self.impact_factor == 1:
                    return 0
                else:
                    smaller_regression_error_loss = mean_square_error(smaller_performance_instances) * len(smaller_performance_instances) / len(performance_data)
                    bigger_regression_error_loss = mean_square_error(bigger_performance_instances) * len(bigger_performance_instances) / len(performance_data)

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
            smaller_ranking_instances = rankings[smaller_instances]  # todo funktioniert dieses select richtig?
            bigger_ranking_instances = rankings[bigger_instances]

            loss = calculate_loss(performance_data, smaller_performance_instances, bigger_performance_instances, smaller_ranking_instances, bigger_ranking_instances)
            return loss

        if depth == 0:
            train_scenario.feature_data, train_scenario.performance_data = scenario_preporcessing()

        # select set of features randomly at each node
        local_scenario = deepcopy(train_scenario)
        local_scenario.features = self.select_features(local_scenario, amount_of_features_to_use)
        local_scenario.feature_data = local_scenario.feature_data[local_scenario.features]

        feature_data = local_scenario.feature_data.values
        performance_data = local_scenario.performance_data.values

        self.fold = fold

        rankings = calculate_ranking_from_performance_data(performance_data)
        # print(type(rankings))
        # print(f"amount of rankings is {len(rankings) / len(performance_data)} of the length of instances")
        stop, self.old_threshold = self.stopping_criterion(
            performance_data,
            self.min_sample_split,
            self.impact_factor,
            depth=depth,
            borda_score=self.borda_score,
            ranking_loss=self.ranking_loss,
            threshold=self.stopping_threshold,
            old_threshold=self.old_threshold,
        )
        if stop:
            self.label = np.average(performance_data, axis=0)
        else:
            best_known_split_loss = math.inf
            best_known_split_point = best_known_split_feature_number = None
            for feature in range(len(local_scenario.features)):
                candidate_splitting_points = self.get_candidate_splitting_points(feature_data[:, feature])

                for splitting_point in candidate_splitting_points:
                    split_loss = evaluate_splitting_point(performance_data, feature_data[:, feature], splitting_point, rankings)

                    if split_loss < best_known_split_loss:
                        best_known_split_loss = split_loss
                        best_known_split_feature_number = feature
                        best_known_split_point = splitting_point
            if best_known_split_feature_number is None:
                self.label = np.average(performance_data, axis=0)
                return self
            self.splitting_value = best_known_split_point
            splitting_feature_name = local_scenario.feature_data.columns[best_known_split_feature_number]
            self.splitting_feature = train_scenario.feature_data.columns.get_loc(splitting_feature_name)

            smaller_instances, bigger_instances = split_by_feature_value(train_scenario.feature_data.values[:, self.splitting_feature], self.splitting_value)

            smaller_scenario = deepcopy(train_scenario)
            smaller_scenario.feature_data = train_scenario.feature_data[smaller_instances]
            smaller_scenario.performance_data = train_scenario.performance_data[smaller_instances]

            smaller_performance_instances = performance_data[smaller_instances]
            smaller_ranking_instances = rankings[smaller_instances]

            # print(f"amount of smaller rankings is {len(smaller_ranking_instances)/len(performance_data)} of the length of instances")

            # bigger_performance_instances = performance_data[bigger_instances]
            # bigger_ranking_instances = rankings[bigger_instances]

            ## print(f"amount of bigger rankings is {len(bigger_ranking_instances)/len(performance_data)} of the length of instances")

            ## calculate smalelr and bigger loss
            # regression_loss = mean_square_error(smaller_performance_instances) * len(smaller_performance_instances) / len(performance_data) + mean_square_error(bigger_performance_instances) * len(
            #    bigger_performance_instances
            # ) / len(performance_data)
            # ranking_loss = self.ranking_loss(smaller_performance_instances, self.borda_score, smaller_ranking_instances) * len(smaller_performance_instances) / len(
            #    performance_data
            # ) + self.ranking_loss(bigger_performance_instances, self.borda_score, bigger_ranking_instances) * len(bigger_performance_instances) / len(performance_data)

            # self.loss_overview.append(((1 - self.impact_factor) * regression_loss / best_known_split_loss, self.impact_factor * ranking_loss / best_known_split_loss))

            bigger_scenario = deepcopy(train_scenario)
            bigger_scenario.feature_data = train_scenario.feature_data[bigger_instances]
            bigger_scenario.performance_data = train_scenario.performance_data[bigger_instances]
            self.left = BinaryDecisionTree(
                self.ranking_loss,
                self.regression_loss,
                self.borda_score,
                self.impact_factor,
                self.stopping_criterion,
                stopping_threshold=self.stopping_threshold,
                old_threshold=self.old_threshold,
                loss_overview=self.loss_overview,
            )
            self.right = BinaryDecisionTree(
                self.ranking_loss,
                self.regression_loss,
                self.borda_score,
                self.impact_factor,
                self.stopping_criterion,
                stopping_threshold=self.stopping_threshold,
                old_threshold=self.old_threshold,
                loss_overview=self.loss_overview,
            )

            self.left.fit(smaller_scenario, self.fold, amount_of_training_instances, depth=depth + 1)

            self.right.fit(bigger_scenario, self.fold, amount_of_training_instances, depth=depth + 1)

        return self

    def select_features(self, scenario: ASlibScenario, amount_of_features):
        return random.sample(scenario.features, amount_of_features)

    def predict(self, features: np.array, scenario):
        assert features.ndim == 1, "Must be 1-dimensional"
        if self.label is not None:
            return self.label

        else:
            if features[self.splitting_feature] < self.splitting_value:
                return self.left.predict(features, scenario)
            else:
                return self.right.predict(features, scenario)
