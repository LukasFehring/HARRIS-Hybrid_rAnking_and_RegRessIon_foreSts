import configparser
import logging
import multiprocessing as mp
import os

from approaches.combined_ranking_regression_trees.borda_score import borda_score_mean
from approaches.combined_ranking_regression_trees.ranking_loss import spearman_rank_correlation
from approaches.combined_ranking_regression_trees.regression_error_loss import regression_error_loss
from approaches.combined_ranking_regression_trees.stopping_criteria import loss_under_threshold

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sklearn.linear_model import Ridge

import database_utils
from approaches.combined_ranking_regression_trees.binary_decision_tree import BinaryDecisionTree
from approaches.oracle import Oracle
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
        if approach_name == "binary_decision_tree_combined_loss":
            basinc_ranking_loss = spearman_rank_correlation
            basic_regression_loss = regression_error_loss
            basic_borda_score = borda_score_mean
            basic_impact_factor = 0.9
            basic_stopping_criterion = loss_under_threshold
            binary_decision_tree = BinaryDecisionTree(basinc_ranking_loss, basic_regression_loss, basic_borda_score, basic_impact_factor, basic_stopping_criterion)
            approaches.append(binary_decision_tree)

    return approaches


#######################
#         MAIN        #
#######################

initialize_logging()
config = load_configuration()
logger.info("Running experiments with config:")
print_config(config)
debug_mode = True
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

for fold in range(1, 11):

    for scenario in scenarios:
        approaches = create_approach(approach_names)

        if len(approaches) < 1:
            logger.error("No approaches recognized!")
        for approach in approaches:
            metrics = list()
            metrics.append(Par10Metric())
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
