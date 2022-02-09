from scipy.stats import rankdata


def calculate_ranking_with_ties(data):
    ranking = rankdata(data, "average") - 1
    ranking_max = rankdata(data, "max") - 1
    ranking_max = ranking_max == len(data) - 1
    ranking[ranking_max] = len(data) - 1
    return ranking
