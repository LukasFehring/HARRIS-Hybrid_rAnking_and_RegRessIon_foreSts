import math
from tkinter.messagebox import NO

import numpy as np
from scipy.stats import rankdata

from approaches.combined_ranking_regression_trees.borda_score import borda_score_mean_performance
from approaches.combined_ranking_regression_trees.ranking_loss import spearman_rank_correlation
from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from approaches.combined_ranking_regression_trees.regression_error_loss import regression_error_loss


def loss_under_threshold(
    performance_data: np.array, min_sample_split, impact_factor, depth=None, borda_score=borda_score_mean_performance, ranking_loss=spearman_rank_correlation, threshold=100000, old_threshold=None
):
    def _calc_loss():
        ranking_instances = calculate_ranking_from_performance_data(performance_data)
        ranking_error = ranking_loss(performance_data, borda_score, ranking_instances)
        regression_error = regression_error_loss(performance_data)
        return impact_factor * ranking_error + (1 - impact_factor) * regression_error

    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True, _calc_loss()

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True, _calc_loss()

    if old_threshold is not None and _calc_loss() >= old_threshold:
        return False, _calc_loss()

    return _calc_loss() <= threshold, _calc_loss()


def same_ranking(performance_data: np.array, min_sample_split, impact_factor, depth=None, threshold=None, old_threshold=None, borda_score=None, ranking_loss=None):
    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True, None

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True, None

    rankings: np.array = calculate_ranking_from_performance_data(performance_data)

    for column in np.transpose(rankings):
        if not np.all(column == column[0]):
            return False, None
    return True, None


def same_ranking_percentage(performance_data: np.array, min_sample_split, impact_factor, depth=None, percentage=0.75, threshold=None, old_threshold=None, borda_score=None, ranking_loss=None):
    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True, None

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True, None

    rankings: np.array = calculate_ranking_from_performance_data(performance_data)

    for i in range(math.ceil((1 - percentage) * len(performance_data))):
        same = list()
        for column in np.transpose(rankings):
            if np.all(column == column[0]):
                same.append(1)
            else:
                same.append(0)
        if np.mean(same) >= percentage:
            return True, None


def max_depth(performance_data: np.array, min_sample_split, impact_factor, depth=None, threshold=65, old_threshold=None, borda_score=None, ranking_loss=None):
    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True, None

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True, None

    else:
        return depth >= threshold, None
