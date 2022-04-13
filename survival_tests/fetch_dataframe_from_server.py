import configparser

import fontTools
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt

import database_utils

plt.rcParams.update({"text.usetex": True})


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open("conf/experiment_configuration.cfg"))
    return config


def fetch_dataframe_from_server(sql_statement):
    db_config = load_configuration()
    db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(db_config)

    return pd.read_sql(sql_statement, db_handle)


def plot_data(sql_statement, x_axis_column, y_axis_column, label, title, sbs_sql_statement, per_algorithm_regressor_sql_statement):
    df = fetch_dataframe_from_server(sql_statement)
    df_sbs = fetch_dataframe_from_server(sbs_sql_statement)
    df_per_algo = fetch_dataframe_from_server(per_algorithm_regressor_sql_statement)
    fig, axs = plt.subplots()
    axs.set_prop_cycle(cycler(color=["c", "m", "y", "k"]))
    fig.set_size_inches(25, 10, forward=True)
    axs.plot(df["impact_factor"], df["result"], "x", label=label)

    axs.axhline(y=df_sbs["result"].iat[0], linestyle="--", label="regression forest")
    axs.axhline(y=df_per_algo["result"].iat[0], linestyle="--", label="single best solver")

    fig.suptitle(title, fontsize=40)
    axs.set_xlabel(x_axis_column, fontsize=30)
    axs.set_ylabel(y_axis_column, fontsize=30)

    plt.legend(loc=(0.5, -0.4), fontsize=20)
    plt.savefig(title + ".png")
