import numpy as np
from scipy.stats import rankdata

from approaches.combined_ranking_regression_trees.util import calculate_ranking_with_ties


def borda_score_mean(ranking_data: np.array):
    ranking_sums = ranking_data.sum(axis=0)
    return calculate_ranking_with_ties(ranking_sums)
