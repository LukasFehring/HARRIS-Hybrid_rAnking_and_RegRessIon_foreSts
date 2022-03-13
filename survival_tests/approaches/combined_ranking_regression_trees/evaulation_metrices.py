import math

import numpy as np
from scipy.stats import kendalltau, rankdata

# from sklearn.metrics import ndcg_score todo kann ich das irgendwie doch nehmen obwohl ne andere python libary version unterliegt?


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
    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        def ndcg(y_true, y_pred):
            n_instances, n_objects = y_true.shape
            relevance = np.power(2.0, ((n_objects - y_true) * 60) / n_objects) - 1.0
            relevance_pred = np.power(2.0, ((n_objects - y_pred) * 60) / n_objects) - 1.0

            log_term = np.log(np.arange(dtype="float32") + 2.0) / np.log(2.0)

            # Calculate ideal dcg:
            top_t = np.argsort(relevance, axis=1)[:, ::-1][:, :]
            toprel = relevance[np.arange(n_instances)[:, None], top_t]
            idcg = np.sum(toprel / log_term, axis=-1, keepdims=True)

            # Calculate actual dcg:
            top_p = np.argsort(relevance_pred, axis=1)[:, ::-1][:, :]
            pred_rel = relevance[np.arange(n_instances)[:, None], top_p]
            pred_rel = np.sum(pred_rel / log_term, axis=-1, keepdims=True)
            gain = pred_rel / idcg
            return gain

        x = rankdata(gt_runtimes)
        y = rankdata(predicted_scores)

        return ndcg(x, y)

    def get_name(self):
        return "NDCG"


class Performance_Regret:
    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        return np.min(predicted_scores) - np.min(gt_runtimes)

    def get_name(self):
        return "performanceRegret"
