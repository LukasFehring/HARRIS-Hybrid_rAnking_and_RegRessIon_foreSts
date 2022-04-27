import numpy as np

from approaches.combined_ranking_regression_trees.ranking_transformer import calculate_ranking_from_performance_data


def modified_position_error(performance_data, borda_score, rankings):
    consensus_ranking = borda_score(rankings, performance_data)
    error = 0
    for ranking, instance_data in zip(rankings, performance_data):
        val = instance_data[np.argmin(consensus_ranking)] - instance_data[np.argmin(ranking)]
        if val < 0:
            raise RuntimeError("val should be > 0 but is not")
        error += val
    return error / len(rankings)


def spearman_rank_correlation(performance_data, borda_score, rankings):
    consensus_ranking = borda_score(rankings, performance_data)
    return np.sum((rankings - consensus_ranking) ** 2) / len(rankings) / (len(rankings) - 1) ** 2


def spearman_footrule(performance_data, borda_scroe, rankings):
    consensuns_ranking = borda_scroe(rankings, performance_data)
    return np.sum(abs(rankings - consensuns_ranking)) / len(rankings) / len(rankings) - 1


def number_of_discordant_pairs(performance_data, borda_score, rankings):
    consensus_ranking = borda_score(rankings, performance_data)
    discordant_pairs = 0
    maximal_discordant_pairs = 0 
    for ranking in rankings: 
        for first_pair in range(len(consensus_ranking)):
            for second_pair in range(first_pair + 1, len(consensus_ranking)):
                if (
                    ranking[first_pair] < ranking[second_pair]
                    and consensus_ranking[first_pair] > consensus_ranking[second_pair]
                    or ranking[first_pair] > ranking[second_pair]
                    and consensus_ranking[first_pair] < consensus_ranking[second_pair]
                ):
                    discordant_pairs += 1
                maximal_discordant_pairs += 1
    return discordant_pairs / maximal_discordant_pairs / len(rankings)


def squared_hinge_loss(performance_data, borda_score, rankings):
    def relevant_algo_pairs(ranking):
        ranking_pairs = list()
        for lower_index, rank in enumerate(ranking):
            for bigger_index, bigger_rank in enumerate(ranking):
                if bigger_rank > rank:
                    ranking_pairs.append((lower_index, bigger_index))

        return ranking_pairs

    performance_array = -1 * performance_data
    diffs = list()
    for instance, ranking in zip(performance_array, rankings):
        algo_pairs = relevant_algo_pairs(ranking)
        for algo_pair in algo_pairs:
            diffs.append(instance[algo_pair[0]] - instance[algo_pair[1]])

    return np.mean(diffs)
