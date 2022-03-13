import numpy as np
from scipy.stats import gmean, rankdata

from approaches.combined_ranking_regression_trees.util import calculate_ranking_with_ties


def borda_score_mean_ranking(ranking_data: np.array, performance_data: np.array):
    ranking_sums = ranking_data.sum(axis=0)
    return calculate_ranking_with_ties(ranking_sums)


def borda_score_median_ranking(ranking_data: np.array, performance_data: np.array):
    ranking_medians = np.median(ranking_data, axis=0)
    return calculate_ranking_with_ties(ranking_medians)


def borda_score_mean_performance(ranking_data: np.array, performance_data: np.array):
    performance_sums = performance_data.sum(axis=0)
    return calculate_ranking_with_ties(performance_sums)


def geometric_mean_performance(ranking_data: np.array, performance_data: np.array):
    ranking_sums = gmean(ranking_data, axis=0)
    return calculate_ranking_with_ties(ranking_sums)
