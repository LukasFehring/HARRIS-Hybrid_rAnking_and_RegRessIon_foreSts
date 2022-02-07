from pyrsistent import l


def borda_score_mean(performance_data):
    algorithm_score = {key: 0 for key in performance_data[0].keys()}

    for instance in performance_data:
        for algorithm, ranking in instance.items():
            algorithm_score[algorithm] += ranking

    max_val = max(algorithm_score.values())
    borda_ranking = dict()
    for position, key_value in enumerate(sorted(algorithm_score.items(),key=  lambda x: x[1])): #todo take care of duplicates
        algorithm, ranking = key_value
        if ranking == max_val:
            borda_ranking[algorithm] = len(performance_data)
        else:
            borda_ranking[algorithm]= position

    return borda_ranking
