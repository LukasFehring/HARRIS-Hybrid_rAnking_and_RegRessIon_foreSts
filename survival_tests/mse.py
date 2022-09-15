import numpy as np


class MSE:
    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        difference = (gt_runtimes - predicted_scores) ** 2
        return np.mean(difference)

    def get_name(self):
        return "mse"
