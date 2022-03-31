import math
from copy import deepcopy

import numpy as np
from scipy.stats import kendalltau, rankdata
from sklearn.preprocessing import minmax_scale


class KendallsTau_b:
    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        """
        gr_runtiems = ground truth runtimes, daher die aus dem test
        predicted scores ist meine ausgabe des algos. also beispielsweise laufzueit oder score
        feature cost ist die summe der zeit der feature berechnung - daher wird sie im part10 dazu addiert - in anderen muss man es sich im detail anschauen"""
        x = rankdata(gt_runtimes)
        y = rankdata(predicted_scores)

        return kendalltau(x, y).correlation

    def get_name(self):
        return "KendallsTau_b"


class NDCG:
    def evaluate(self, gt_scores: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        predicted_scores = np.array(deepcopy(predicted_scores))
        gt_scores = np.array(minmax_scale(gt_scores, feature_range=(0, 1)))

        predicted_ranking = rankdata(predicted_scores)
        gt_ranking = rankdata(gt_scores)

        predicted_scores_sum = np.sum((gt_scores - 1) / np.log2(1 + predicted_ranking))
        return predicted_scores_sum / np.sum((gt_scores - 1) / np.log2(1 + gt_ranking))

    def get_name(self):
        return "NDCG"


class Performance_Regret:
    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        min_predicted_alg = np.argmin(predicted_scores)
        min_predicted_alg_performance = gt_runtimes[min_predicted_alg]
        return min_predicted_alg_performance - np.min(gt_runtimes)

    def get_name(self):
        return "performanceRegret"
