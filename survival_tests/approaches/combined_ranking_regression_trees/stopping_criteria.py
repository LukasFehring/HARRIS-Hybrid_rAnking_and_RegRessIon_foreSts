import numpy as np
from torch import threshold

from approaches.combined_ranking_regression_trees.regression_error_loss import regression_error_loss
from aslib_scenario import ASlibScenario


def loss_under_threshold(performance_data: np.array, min_samples_leave, threshold=10000):
    if np.ma.size(performance_data, axis=0) < 2 * min_samples_leave + 1:
        return True

    elif np.ma.size(performance_data, axis=0) <= 1:
        return True

    return regression_error_loss(performance_data) <= threshold
