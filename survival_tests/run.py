import configparser
import copy
import logging
import multiprocessing as mp
import os

import numpy as np

#
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
#

from sklearn.linear_model import Ridge

import database_utils
from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from approaches.combined_ranking_regression_trees.borda_score import borda_score_mean_performance, borda_score_mean_ranking, borda_score_median_ranking, geometric_mean_performance
from approaches.combined_ranking_regression_trees.evaulation_metrices import NDCG, KendallsTau_b, Performance_Regret
from approaches.combined_ranking_regression_trees.ranking_loss import (
    corrected_spearman_footrule,
    modified_position_error,
    number_of_discordant_pairs,
    spearman_footrule,
    spearman_rank_correlation,
    spearman_rankk_correlation_no_normalisation,
    squared_hinge_loss,
)
from approaches.combined_ranking_regression_trees.regression_error_loss import mean_square_error
from approaches.combined_ranking_regression_trees.stopping_criteria import loss_under_threshold, max_depth, same_ranking, same_ranking_percentage
from approaches.oracle import Oracle
from approaches.ranking_and_regression_forest.combined_ranking_and_regression_forest import CombinedRankingAndRegressionForest as Forest
from approaches.ranking_and_regression_forest.consensus import average_runtimes, max_runtimes, min_runtimes
from approaches.single_best_solver import SingleBestSolver
from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla07 import SATzilla07
from baselines.satzilla11 import SATzilla11
from baselines.snnap import SNNAP
from baselines.sunny import SUNNY
from evaluation import evaluate_scenario
from number_unsolved_instances import NumberUnsolvedInstances
from par_10_metric import Par10Metric

logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    logging.basicConfig(filename="logs/log_file.log", filemode="w", format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.DEBUG)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open("conf/experiment_configuration.cfg"))
    return config


def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        logger.info(str(section) + ": " + str(dict(config[section])))


def log_result(result):
    logger.info("Finished experiements for scenario: " + result)


def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:
        if approach_name == "sbs":
            approaches.append(SingleBestSolver())
        if approach_name == "oracle":
            approaches.append(Oracle())
        if approach_name == "ExpectationSurvivalForest":
            approaches.append(SurrogateSurvivalForest(criterion="Expectation"))
        if approach_name == "PolynomialSurvivalForest":
            approaches.append(SurrogateSurvivalForest(criterion="Polynomial"))
        if approach_name == "GridSearchSurvivalForest":
            approaches.append(SurrogateSurvivalForest(criterion="GridSearch"))
        if approach_name == "ExponentialSurvivalForest":
            approaches.append(SurrogateSurvivalForest(criterion="Exponential"))
        if approach_name == "SurrogateAutoSurvivalForest":
            approaches.append(SurrogateAutoSurvivalForest())
        if approach_name == "PAR10SurvivalForest":
            approaches.append(SurrogateSurvivalForest(criterion="PAR10"))
        if approach_name == "per_algorithm_regressor":
            approaches.append(PerAlgorithmRegressor())
        if approach_name == "imputed_per_algorithm_rf_regressor":
            approaches.append(PerAlgorithmRegressor(impute_censored=True))
        if approach_name == "imputed_per_algorithm_ridge_regressor":
            approaches.append(PerAlgorithmRegressor(scikit_regressor=Ridge(alpha=1.0), impute_censored=True))
        if approach_name == "multiclass_algorithm_selector":
            approaches.append(MultiClassAlgorithmSelector())
        if approach_name == "sunny":
            approaches.append(SUNNY())
        if approach_name == "snnap":
            approaches.append(SNNAP())
        if approach_name == "satzilla-11":
            approaches.append(SATzilla11())
        if approach_name == "satzilla-07":
            approaches.append(SATzilla07())
        if approach_name == "isac":
            approaches.append(ISAC())

        if approach_name == "ablation_study_stopping_criteria":
            impact_factor = 0.6
            ranking_loss = corrected_spearman_footrule
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            for stopping_criterion in [loss_under_threshold, max_depth, same_ranking, same_ranking_percentage]:
                if stopping_criterion == max_depth:
                    for stopping_threshold in (0, 1, 2, 3, 4, 5):
                        binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                        approaches.append(binary_decision_tree)
                elif stopping_criterion == loss_under_threshold:
                    for stopping_threshold in np.arange(0, 1.04, 0.04):
                        binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                        approaches.append(binary_decision_tree)
                elif stopping_criterion == same_ranking:
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=None)
                    approaches.append(binary_decision_tree)
                if stopping_criterion == same_ranking_percentage:
                    for stopping_threshold in np.arange(0.1, 0.91, 0.1):
                        binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                        approaches.append(binary_decision_tree)

        if approach_name == "ablation_study_lambda":
            for impact_factor in np.arange(0.0, 1.05, 0.05):
                ranking_loss = copy.deepcopy(corrected_spearman_footrule)
                regression_loss = copy.deepcopy(mean_square_error)
                borda_score = borda_score_mean_ranking
                stopping_threshold = 3
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                approaches.append(binary_decision_tree)

        if approach_name == "ablation_study_borda_score":
            for borda_score in (borda_score_mean_performance, borda_score_mean_ranking, borda_score_median_ranking, geometric_mean_performance):
                impact_factor = 0.6
                ranking_loss = copy.deepcopy(spearman_footrule)
                regression_loss = copy.deepcopy(mean_square_error)
                stopping_threshold = 3
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                approaches.append(binary_decision_tree)

        if approach_name == "ablation_study_ranking_loss":
            for ranking_loss in (modified_position_error, corrected_spearman_footrule, spearman_rank_correlation, squared_hinge_loss, number_of_discordant_pairs):
                borda_score = borda_score_mean_ranking
                regression_loss = copy.deepcopy(mean_square_error)
                stopping_threshold = 3
                stopping_criterion = max_depth
                impact_factor = 0.6
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                approaches.append(binary_decision_tree)

        if approach_name == "ablation_study_lambda_and_ranking":
            for ranking_loss in (modified_position_error, spearman_rank_correlation, number_of_discordant_pairs, squared_hinge_loss):
                for impact_factor in np.arange(0.0, 1.05, 0.1):
                    borda_score = borda_score_mean_ranking
                    regression_loss = copy.deepcopy(mean_square_error)
                    stopping_threshold = 3
                    stopping_criterion = max_depth
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                    approaches.append(binary_decision_tree)

        if approach_name == "loss_analysis_discordant_pairs_and_lambda":
            for impact_factor in np.arange(0.0, 1.025, 0.025):
                ranking_loss = copy.deepcopy(number_of_discordant_pairs)
                regression_loss = copy.deepcopy(mean_square_error)
                borda_score = borda_score_mean_ranking
                stopping_threshold = 3
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                approaches.append(binary_decision_tree)

        if approach_name == "comparison_to_others":
            # corrected_spearman
            for impact_factor in np.arange(0.0, 1.05, 0.1):
                ranking_loss = copy.deepcopy(corrected_spearman_footrule)
                regression_loss = copy.deepcopy(mean_square_error)
                borda_score = borda_score_mean_ranking
                for stopping_threshold in (4,):
                    stopping_criterion = max_depth
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                    approaches.append(binary_decision_tree)

        if approach_name == "ablation_study_mu":
            for mu in (1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2, 3, 4, 5, 7.5, 10, 15, 20):
                for impact_factor in (0, 1):
                    borda_score = borda_score_mean_ranking
                    regression_loss = copy.deepcopy(mean_square_error)
                    ranking_loss = corrected_spearman_footrule
                    stopping_threshold = 3
                    stopping_criterion = max_depth
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold, mu=mu)
                    approaches.append(binary_decision_tree)

        if approach_name == "comparison_before_after_modification":
            ranking_loss = spearman_rankk_correlation_no_normalisation
            for impact_factor in np.arange(0.0, 1.05, 0.1):
                borda_score = borda_score_mean_ranking
                regression_loss = copy.deepcopy(mean_square_error)
                stopping_threshold = 3
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold, do_preprocessing=False)
                approaches.append(binary_decision_tree)

            ranking_loss = spearman_rank_correlation
            for impact_factor in np.arange(0.0, 1.05, 0.1):
                borda_score = borda_score_mean_ranking
                regression_loss = copy.deepcopy(mean_square_error)
                stopping_threshold = 3
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold, do_preprocessing=True)
                approaches.append(binary_decision_tree)

            # modified position error #todo das einmal einkommentieren für das eine scenario in dem die gut sind
            for impact_factor in np.arange(0.0, 1.05, 0.1):
                ranking_loss = copy.deepcopy(modified_position_error)
                regression_loss = copy.deepcopy(mean_square_error)
                borda_score = borda_score_mean_ranking
                for stopping_threshold in (1, 2, 3, 4):
                    stopping_criterion = max_depth
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                    approaches.append(binary_decision_tree)

        if approach_name == "fix_squared_hinge_loss":
            impact_factor = 0.6
            ranking_loss = copy.deepcopy(squared_hinge_loss)
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            stopping_threshold = 3
            stopping_criterion = max_depth
            binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
            approaches.append(binary_decision_tree)

        if approach_name == "evaluate_feature_bagging_with_deep_tree":
            ranking_loss = copy.deepcopy(modified_position_error)
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            impact_factor = 0.5
            stopping_criterion = max_depth
            amount_of_trees = 100
            stopping_depth = 5
            consensus_function = average_runtimes
            for feature_percentage in (0.8, 0.9):
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_depth)
                forest = Forest(amount_of_trees, binary_decision_tree, consensus=consensus_function, feature_percentage=feature_percentage)
                approaches.append(forest)

        if approach_name == "evaluate_tree_depth":
            ranking_loss = copy.deepcopy(modified_position_error)
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            impact_factor = 0.5
            stopping_criterion = max_depth
            amount_of_trees = 30
            feature_percentage = 0.7
            consensus_function = average_runtimes
            for stopping_depth in (3, 4, 5):
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_depth)
                forest = Forest(amount_of_trees, binary_decision_tree, consensus=consensus_function, feature_percentage=feature_percentage)
                approaches.append(forest)

        if approach_name == "debug_forests":
            ranking_loss = copy.deepcopy(modified_position_error)
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            impact_factor = 0.5
            stopping_criterion = max_depth
            stopping_depth = 1
            amount_of_trees = 3
            feature_percentage = 0.7
            consensus_function = average_runtimes
            stopping_criterion = max_depth
            binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_depth)
            forest = Forest(amount_of_trees, binary_decision_tree, consensus=consensus_function, feature_percentage=feature_percentage)
            approaches.append(forest)

        if approach_name == "evaluate_consensus_function":
            ranking_loss = copy.deepcopy(modified_position_error)
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            impact_factor = 0.5
            stopping_criterion = max_depth
            stopping_depth = 5
            amount_of_trees = 100
            feature_percentage = 1
            for consensus_function in (average_runtimes, max_runtimes, min_runtimes):
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_depth)
                forest = Forest(amount_of_trees, binary_decision_tree, consensus=consensus_function, feature_percentage=feature_percentage)
                approaches.append(forest)

        if approach_name == "evaluate_amount_of_trees":
            ranking_loss = copy.deepcopy(modified_position_error)
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            impact_factor = 0.5
            stopping_criterion = max_depth
            feature_percentage = 0.7
            for amount_of_trees in (30, 40, 50):
                stopping_criterion = max_depth
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_depth)
                forest = Forest(amount_of_trees, binary_decision_tree, consensus=consensus_function, feature_percentage=feature_percentage)
                approaches.append(forest)

        if approach_name == "evaluate_ranking_loss_and_lambda":  # todo das hier wird noch nicht ausgewertet
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            stopping_criterion = max_depth
            amount_of_trees = 20
            feature_percentage = 0.7
            for ranking_loss in (modified_position_error, squared_hinge_loss, corrected_spearman_footrule):
                for impact_factor in np.arange(0.0, 1.05, 0.1):
                    stopping_criterion = max_depth
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_depth)
                    forest = Forest(amount_of_trees, binary_decision_tree, consensus=consensus_function, feature_percentage=feature_percentage)
                    approaches.append(forest)

        if approach_name == "poster_forest":
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            stopping_criterion = max_depth
            amount_of_trees = 100
            feature_percentage = 0.7
            ranking_loss = modified_position_error
            consensus_function = average_runtimes
            stopping_threshold = 6
            for impact_factor in np.arange(0.0, 1.05, 0.1):
                binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                forest = Forest(amount_of_trees, binary_decision_tree, consensus=consensus_function, feature_percentage=feature_percentage)
                approaches.append(forest)

        if approach_name == "poster_tree":
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            stopping_criterion = max_depth
            ranking_loss = modified_position_error
            for stopping_threshold in (1, 2, 3, 4, 5):
                for impact_factor in np.arange(0.0, 1.05, 0.1):
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                    approaches.append(binary_decision_tree)

        if approach_name == "evaluate_ranking_loss_and_lambda":
            regression_loss = copy.deepcopy(mean_square_error)
            borda_score = borda_score_mean_ranking
            stopping_criterion = max_depth
            stopping_threshold = 8
            consensus_function = average_runtimes
            for ranking_loss in [corrected_spearman_footrule, squared_hinge_loss, modified_position_error, spearman_rank_correlation]:
                for impact_factor in [0.3, 0.7, 1]:
                    binary_decision_tree = BinaryDecisionTree(ranking_loss, regression_loss, borda_score, impact_factor, stopping_criterion, stopping_threshold=stopping_threshold)
                    forest = Forest(100, binary_decision_tree, consensus=average_runtimes, feature_percentage=0.7)
                    approaches.append(forest)

    return approaches


#######################
#         MAIN        #
#######################

initialize_logging()
config = load_configuration()
logger.info("Running experiments with config:")
print_config(config)
debug_mode = False
if debug_mode:
    logger.setLevel(logging.DEBUG)
# fold = int(sys.argv[1])
# logger.info("Running experiments for fold " + str(fold))

db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(config)
database_utils.create_table_if_not_exists(db_handle, table_name)

amount_of_cpus_to_use = int(config["EXPERIMENTS"]["amount_of_cpus"])
pool = mp.Pool(amount_of_cpus_to_use)


scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
approach_names = config["EXPERIMENTS"]["approaches"].split(",")
amount_of_scenario_training_instances = int(config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
tune_hyperparameters = bool(int(config["EXPERIMENTS"]["tune_hyperparameters"]))


for scenario in scenarios:
    for fold in range(1, 6):
        approaches = create_approach(approach_names)

        if len(approaches) < 1:
            logger.error("No approaches recognized!")
        for approach in approaches:
            metrics = list()
            metrics.append(Par10Metric())
            metrics.append(NDCG())
            metrics.append(KendallsTau_b())
            metrics.append(Performance_Regret())
            if approach.get_name() != "oracle":
                metrics.append(NumberUnsolvedInstances(False))
                metrics.append(NumberUnsolvedInstances(True))

            if debug_mode:
                evaluate_scenario(scenario, approach, metrics, amount_of_scenario_training_instances, fold, config, tune_hyperparameters)
                print("Finished evaluation of fold")
            else:
                logger.info('Submitted pool task for approach "' + str(approach.get_name()) + '" on scenario: ' + scenario)
                pool.apply_async(evaluate_scenario, args=(scenario, approach, metrics, amount_of_scenario_training_instances, fold, config, tune_hyperparameters), callback=log_result)

                print("Finished evaluation of fold")

pool.close()
pool.join()
logger.info("Finished all experiments.")
