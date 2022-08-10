import copy
import datetime
import os
import pickle

from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from approaches.combined_ranking_regression_trees.borda_score import borda_score_mean_ranking
from approaches.combined_ranking_regression_trees.ranking_loss import corrected_spearman_footrule
from approaches.combined_ranking_regression_trees.regression_error_loss import mean_square_error
from approaches.combined_ranking_regression_trees.stopping_criteria import max_depth
from approaches.single_best_solver import SingleBestSolver
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from aslib_scenario import ASlibScenario
from baselines.per_algorithm_regressor import PerAlgorithmRegressor

models_to_evaluate = {
    "sbs": SingleBestSolver(),
    "per_algo_regressor": PerAlgorithmRegressor(),
    "survival_forest": SurrogateSurvivalForest(criterion="Expectation"),
}
impact_factor = 0.6
ranking_loss = copy.deepcopy(corrected_spearman_footrule)
regression_loss = copy.deepcopy(mean_square_error)
borda_score = borda_score_mean_ranking
for stopping_threshold in (
    2,
    3,
):
    stopping_criterion = max_depth
    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
    models_to_evaluate[binary_decision_tree.get_name() + "_depth_" + str(binary_decision_tree.stopping_threshold)] = binary_decision_tree

performance_times = {}

for model_name, models_to_evaluate in list(models_to_evaluate.items())[3:]:
    tmp_model = copy.deepcopy(models_to_evaluate)
    performance_times[model_name] = {}
    for scenario_name in [
        "CSP-2010",
        "QBF-2016",
        "ASP-POTASSCO",
        "SAT12-HAND",
        "MAXSAT15-PMS-INDU",
        "SAT12-INDU",
        "CPMP-2015",
    ]:
        performance_times[model_name][scenario_name] = {}
        scenario = ASlibScenario()
        scenario.read_scenario(os.path.join("data", "aslib_data-master", scenario_name))
        for fold in range(1, 11):
            print("evaluating fold", fold, "of scenario", scenario_name, "with model", model_name)
            test_scenario, train_scenario = scenario.get_split(indx=fold)
            threshold = train_scenario.algorithm_cutoff_time
            train_scenario.performance_data = train_scenario.performance_data.clip(upper=threshold)
            start_time = datetime.datetime.now()
            tmp_model.fit(train_scenario, fold, -1)
            end_time = datetime.datetime.now()
            training_time = (end_time - start_time).total_seconds()
            performance_times[model_name][scenario_name][fold] = training_time

with open("performance_data_file.pkl", "wb") as handle:
    pickle.dump(performance_times, handle)
print("finished")
