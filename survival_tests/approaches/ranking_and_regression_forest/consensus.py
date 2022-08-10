from typing import List

import numpy as np


def average_runtimes(predictions: List[np.array]):
    return np.average(predictions, axis=0)

def max_runtimes(predictions: List[np.array]):
    return np.max(predictions, axis=0)

def min_runtimes(predictions: List[np.array]):
    return np.min(predictions, axis=0)

