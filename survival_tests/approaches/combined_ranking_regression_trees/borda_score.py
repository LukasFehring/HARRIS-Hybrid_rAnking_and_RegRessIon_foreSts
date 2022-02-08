import numpy as np
from scipy.stats import rankdata


def borda_score_mean(ranking_data: np.array):
    ranking_sums = ranking_data.sum(axis=0)
    consensus_ranking = rankdata(ranking_sums, "average")
    return consensus_ranking
