import numpy as np
import pandas as pd


def calculate_ranking_from_performance_data(performance_data: np.array):
    def calculate_ranking_from_instance(instance, worst_performance):
        worst_performance = performance_data.max()
        grouped_algos = {x: [] for x in instance}
        last_value = -1
        for algo_number, x in enumerate(instance):
            grouped_algos[x].append(algo_number)
        ranking = {}
        counter = 0
        for group_number, algo_numbers in sorted(grouped_algos.items(), key=lambda x: x[0]):
            if group_number == worst_performance:
                for algo_number in algo_numbers:
                    ranking[algo_number] = np.ma.size(performance_data, axis=1)
            elif len(algo_numbers) == 1:
                ranking[algo_numbers[0]] = counter
                counter += 1
            else:
                avg = sum(range(counter, len(algo_numbers) + counter)) / len(algo_numbers)
                for x in algo_numbers:
                    ranking[x] = avg
                counter += len(algo_numbers)

        return ranking

    ranked_instances = list()
    worst_performance = performance_data.max()
    for instance in performance_data:
        ranked_instances.append(calculate_ranking_from_instance(instance, worst_performance))
    return ranked_instances
