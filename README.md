# Code for paper: "HARRIS: Hybrid Ranking and Regression Forests for Algorithm Selection"

This repository holds the code for our paper "HARRIS: Hybrid Ranking and Regression Forests for Algorithm Selection" by Lukas Fehring, Alexander Tornede, and Jonas Hanselle. It is build on top of https://github.com/alexandertornede/algorithm_survival_analysis. Regarding questions please contact lukas.fehring@stud.uni-hannover.de.

## Abstract
It is well known that different algorithms perform differently well on an instance of an algorithmic problem, motivating algorithm selection (AS): Given an instance of an algorithmic problem, which is the most suitable algorithm to solve it? As such, the AS problem has received considerable attention resulting in various approaches -- many of which either solve a regression or ranking problem under the hood. Although both of these formulations yield very natural ways to tackle AS, they have considerable weaknesses. On the one hand, correctly predicting the performance of an algorithm on an instance is a sufficient, but not a necessary condition to produce a correct ranking over algorithms and in particular ranking the best algorithm first. On the other hand, classical ranking approaches often do not account for concrete performance values available in the training data, but only leverage rankings composed from such data. We propose \name - Hybrid rAnking and RegRessIon foreSts - a new algorithm selector leveraging special forests, combining the strengths of both approaches while alleviating their weaknesses. \name' decisions are based on a forest model, whose trees are created based on splits optimized on a hybrid ranking and regression loss function. As our preliminary experimental study on ASLib shows, \name improves over standard algorithm selection approaches on some scenarios showing that combining ranking and regression in trees is indeed promising for AS.

## Execution Details (Getting the Code To Run)
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below.

### 1. Configuration
In order to reproduce the results by running our code, we assume that you have a MySQL server with version >=5.7.9 running.

As a next step, you have to create a configuration file entitled `experiment_configuration.cfg` in the `conf` folder next to `allocate.txt`. This configuration file should contain the following information:

```
[DATABASE]
host = my.sqlserver.com
username = username
password = password
database = databasename
table = tablename
ssl = true

[EXPERIMENTS]
scenarios=ASP-POTASSCO,CSP-Minizinc-Time-2016,MAXSAT12-PMS,MIP-2016,QBF-2011,QBF-2016,SAT12-ALL,SAT12-HAND,CPMP-2015
approaches=HARRIS,isac,per_algorithm_regressor,satzilla-11
amount_of_training_scenario_instances=-1
amount_of_cpus=16
tune_hyperparameters=0
train_status=all
```

You have to adapt all entries below the `[DATABASE]` tag according to your database server setup. The entries have the following meaning:
* `host`: the address of your database server
* `username`: the username the code can use to access the database
* `password`: the password the code can use to access the database
* `database`: the name of the database where you imported the tables
* `table`: the name of the table, where results should be stored. This is created automatically by the code if it does not exist yet and should NOT be created manually.
* `ssl`: whether ssl should be used or not

Entries below the `[EXPERIMENTS]` define which experiments will be run. The configuration above will produce the main results presented in the paper.

### 2. Packages and Dependencies
For running the code several dependencies have to be fulfilled. The easiest way of getting there is by using [Anaconda](https://anaconda.org/). For this purpose, you find an Anaconda environment definition called `environment.yml` at the top-level of this project.  Assuming that you have Anaconda installed, you can create an according environment with all required packages via

```
conda env create -f environment.yml
``` 

which will create an environment named `HARRIS`. After it has been successfully installed, you can use 
```
conda activate HARRIS
```
to activate the environment and run the code (see step 4).

### 3. ASLib Data
Obviously, the code requires access to the ASLib scenarios in order to run the requested evaluations. It expects the ASLib scenarios (which can be downloaded from [Github](https://github.com/coseal/aslib_data)) to be located in a folder `data/aslib_data-master` on the top-level of your IDE project. I.e. your folder structure should look similar to this: 
```
./survival_tests
./survival_tests/approaches
./survival_tests/approaches/survival_forests
./survival_tests/approaches/combined_ranking_and_regression_forest
./survival_tests/approaches/combined_ranking_and_regerssion_trees
./survival_tests/results
./survival_tests/singularity
./survival_tests/tests
./survival_tests/data
./survival_tests/data/aslib_data-master
./survival_tests/conf
```


### 4. Evaluation Results
At this point you should be good to go and can execute the experiments by running the `run.py` on the top-level of the project. 

 All results will be stored in the table given in the configuration file and has the following columns:

* `scenario_name`: The name of the scenario.
* `fold`: The train/test-fold associated with the scenario which is considered for this experiment
* `approach`: The approach which achieved the reported results, where `Run2SurviveExp := Expectation_algorithm_survival_forest`, `Run2SurvivaPar10 := PAR10_algorithm_survival_forest` and `Run2SurvivePolyLog := SurrogateAutoSurvivalForest`
* `metric`: The metric which was used to generate the result. For the `number_unsolved_instances` metric, the suffix `True` indicates that feature costs are accounted for whereas for `False` this is not the case. All other metrics automatically incorporate feature costs.
* `result`: The output of the corresponding metric.
* `impact_factor`: Null if not Hybrid Forest or Hybrid Tree, else λ from the paper
* `stopping_criteria`: Null if not Hybrid Forest or Hybrid Tree, else stopping criterion of tree(s)
* `ranking_error`: Null iff not Hybrid forest or Hybrid Tree, else ranking error used at training time

### 6. Generating Plots and table
All plots/the table found in the paper can be generated using the self-explanatory Jupyter notebook `visualization.ipynb` in the top-level `results` folder.
