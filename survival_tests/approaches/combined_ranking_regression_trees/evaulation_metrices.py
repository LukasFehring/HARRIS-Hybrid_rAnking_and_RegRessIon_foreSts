import math
from copy import deepcopy
from dis import disco

import numpy as np
from scipy.stats import kendalltau, rankdata
from sklearn.preprocessing import minmax_scale


class KendallsTau_b:
    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        x = rankdata(gt_runtimes, method="min")
        y = rankdata(predicted_scores, method="min")

        relevant_pairs = {(i, j) for i in range(len(x)) for j in range(len(y)) if i < j}

        # caclulate concordant instances
        concordant_instances = 0
        discordant_instances = 0
        tied_predicted_instances = 0
        tied_gt_instances = 0
        for pair in relevant_pairs:
            if x[pair[0]] < x[pair[1]] and y[pair[0]] < y[pair[1]]:
                concordant_instances += 1
            elif x[pair[0]] > x[pair[1]] and y[pair[0]] > y[pair[1]]:
                concordant_instances += 1
            elif x[pair[0]] < x[pair[1]] and y[pair[0]] > y[pair[1]]:
                discordant_instances += 1
            elif x[pair[0]] > x[pair[1]] and y[pair[0]] < y[pair[1]]:
                discordant_instances += 1
            if x[pair[0]] == x[pair[1]]:
                tied_gt_instances += 1
            if y[pair[0]] == y[pair[1]]:
                tied_predicted_instances += 1
        try:
            return (concordant_instances - discordant_instances) / math.sqrt((len(relevant_pairs) - tied_gt_instances) * (relevant_pairs - tied_predicted_instances))
        except ZeroDivisionError:
            return 10000000000000000000000000  # arbitrary high so mistake in data is recognized

    def get_name(self):
        return "KendallsTau_b"


class NDCG:
    def evaluate(self, gt_scores: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        predicted_scores = np.array(deepcopy(predicted_scores))
        gt_scores = np.array(minmax_scale(gt_scores, feature_range=(0, 1)))

        predicted_ranking = rankdata(predicted_scores)
        gt_ranking = rankdata(gt_scores)

        predicted_scores_sum = np.sum((-gt_scores + 1 - 1) / np.log2(1 + predicted_ranking))
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
