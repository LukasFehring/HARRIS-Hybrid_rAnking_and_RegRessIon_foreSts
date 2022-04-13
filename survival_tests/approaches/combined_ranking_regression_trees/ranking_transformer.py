import numpy as np
import pandas as pd
from scipy.stats import rankdata

from approaches.combined_ranking_regression_trees.util import calculate_ranking_with_ties


def calculate_ranking_from_performance_data(performance_data: np.array): 
    ranked_instances = list()
    for instance in performance_data:
        ranked_instances.append(calculate_ranking_with_ties(instance))
    return np.array(ranked_instances)
