import numpy as np

from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data


def modified_position_error(performance_data, borda_score, rankings):
    consensus_ranking = borda_score(rankings)
    avg_performances = np.mean(performance_data, axis=0)
    error = 0
    for ranking in rankings:  # todo this is not efficient
        val = avg_performances[int(consensus_ranking[0])] - avg_performances[int(ranking[0])]
        error += val
    return error / len(rankings)


def spearman_rank_correlation(performance_data, borda_score, rankings):
    consensus_ranking = borda_score(rankings)
    return np.sum((rankings - consensus_ranking) ** 2) / len(rankings)
