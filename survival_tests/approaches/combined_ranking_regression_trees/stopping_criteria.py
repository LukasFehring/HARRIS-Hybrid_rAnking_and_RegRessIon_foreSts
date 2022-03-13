import numpy as np
from scipy.stats import rankdata
from torch import threshold

from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data
from approaches.combined_ranking_regression_trees.regression_error_loss import regression_error_loss
from aslib_scenario import ASlibScenario


def loss_under_threshold(performance_data: np.array, min_sample_split, threshold=10000, max_depth=None, depth=None, percentage=None):
    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True

    return regression_error_loss(performance_data) <= threshold


def same_ranking(performance_data: np.array, min_sample_split, threshold=None, max_depth=None, depth=None, percentage=None):
    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True

    rankings: np.array = calculate_ranking_from_performance_data(performance_data)

    for column in np.transpose(rankings):
        if not np.all(column == column[0]):  # todo testcase
            return False
    return True


def same_ranking_percentage(performance_data: np.array, min_sample_split, threshold=None, max_depth=None, depth=None, percentage=0.75):
    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True

    rankings: np.array = calculate_ranking_from_performance_data(performance_data)

    same = list()
    for column in np.transpose(rankings):
        if not np.all(column == column[0]):  # todo testcase
            same.append(1)
        else:
            same.append(0)
    return np.mean(same) >= percentage


def max_depth(performance_data: np.array, min_sample_split, threshold=None, max_depth=None, depth=None, percentage=None):
    if min_sample_split is not None and np.ma.size(performance_data, axis=0) < min_sample_split:
        return True

    elif np.ma.size(performance_data, axis=0) <= 1: #todo testcase
        return True

    else:
        return depth == None
