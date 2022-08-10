import numpy as np
import pandas as pd


def mean_square_error(performance_data: np.array):
    def mean_by_instance(performance_data: np.array):
        if len(performance_data) > 1:
            return performance_data.var(axis=0)
        else:
            raise ValueError(f"performance_data has not enough instances to calculate variance")

    try:
        return (1 / np.ma.size(performance_data, axis=1)) * mean_by_instance(performance_data).sum()
    except ValueError:
        return 0
