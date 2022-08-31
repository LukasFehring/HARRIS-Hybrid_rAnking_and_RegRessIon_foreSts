import os

import numpy as np
import pandas as pd

from aslib_scenario import ASlibScenario

relevant_data = []
scenarios = []
columns = [
    "Scenario",
    "Problem",
    "Instances",
    "Algorithms",
    "Features",
    "Unsolved Instances",
    "Relative Unsolved Instances",
    "Missing Evaluations",
    "Cutoff",
    "Loss",
]
long_columns = [
    "Scenario Name",
    "Algorithmic Problem",
    "Number of Instances",
    "Number of Algorithms",
    "Number of Features",
    "Number of Unsolved Instances",
    "Relative Number of Unsolved Instances",
    "Relative Number of Missing Evaluations",
    "Cutoff Time",
    "Loss Function",
]

scenarios = list()
for scenarioname in [f.path for f in os.scandir("data/aslib_data-master") if f.is_dir()]:
    scenarios.append(scenarioname.split("/")[-1])
    scenario: ASlibScenario = ASlibScenario()
    scenario.read_scenario(scenarioname)
    relevant_data.append(
        (
            scenarioname.split("/")[-1],
            scenarioname.split("/")[-1].split("-")[0],
            len(scenario.instances),
            len(scenario.algorithms),
            len(scenario.features),
            len(scenario.runstatus_data[np.all(scenario.runstatus_data.values != "ok", axis=1)]),
            np.round(len(scenario.runstatus_data[np.all(scenario.runstatus_data.values != "ok", axis=1)]) / len(scenario.instances), decimals=2),
            np.round(
                len(scenario.performance_data.values.flatten()[scenario.performance_data.values.flatten() >= scenario.algorithm_cutoff_time]) / len(scenario.performance_data.values.flatten()),
                decimals=2,
            ),
            scenario.algorithm_cutoff_time,
            ", ".join(scenario.performance_measure),
        )
    )

print(
    pd.DataFrame(relevant_data, columns=columns)
    .sort_values(["Scenario"])
    .to_latex(index=False, column_format="l|cccccccccc")
    .replace("\\toprule\n", "")
    .replace("\\midrule", "\\hline")
    .replace("\\bottomrule\n", "")
)
