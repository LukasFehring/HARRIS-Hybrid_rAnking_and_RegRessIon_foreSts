import numpy as np
import pandas as pd
from scipy.stats import rankdata


def calculate_ranking_from_performance_data(performance_data: np.array):  # todo modify this to be a np array
    def calculate_ranking_from_instance(instance):
        ranking = rankdata(instance, "average") - 1
        ranking_max = rankdata(instance, "max") - 1
        ranking_max = ranking_max == len(instance) - 1
        ranking[ranking_max] = len(instance) - 1
        return ranking

    ranked_instances = list()
    for instance in performance_data:
        ranked_instances.append(calculate_ranking_from_instance(instance))
    return np.array(ranked_instances)
