import configparser
from email.mime import base

import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt
from pyparsing import line

plt.style.use("seaborn-whitegrid")

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


def plot_data_with_baseline(sql_statement, x_axis_column, y_axis_column, label, title, sbs_sql_statement, per_algorithm_regressor_sql_statement):

    df = fetch_dataframe_from_server(sql_statement)
    df_sbs = fetch_dataframe_from_server(sbs_sql_statement)
    df_per_algo = fetch_dataframe_from_server(per_algorithm_regressor_sql_statement)

    fig, axs = plt.subplots()
    axs.set_prop_cycle(cycler(color=["r", "b", "c", "k"]))

    fig.set_size_inches(25, 10, forward=True)
    axs.plot(df["impact_factor"], df["result"], "x", label=label, color="r")

    axs.axhline(y=df_sbs["result"].iat[0], linestyle=":", label="per algorithm regressor", linewidth=4, color="b")
    axs.axhline(y=df_per_algo["result"].iat[0], linestyle=":", label="single best solver", linewidth=4, color="c")

    fig.suptitle(title, fontsize=40)
    axs.tick_params(axis="both", which="major", labelsize=25)
    axs.grid(linewidth=2)
    axs.set_xlim(-0.01, 1.01)
    axs.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    axs.set_xlabel(x_axis_column, fontsize=35)
    axs.set_ylabel(y_axis_column, fontsize=35)

    axs.patch.set_edgecolor("black")
    axs.patch.set_linewidth("2")
    fig.legend(loc=1, fontsize=30)
    plt.savefig(title + ".png", bbox_inches="tight")


def compare_losses_and_lambda(sql_statements, legend_names, x_axis_column, y_axis_column, title, baselines):
    fig, axs = plt.subplots()
    fig.suptitle(title, fontsize=40)
    fig.set_size_inches(25, 10, forward=True)
    axs.grid(linewidth=2)
    axs.set_prop_cycle(cycler(color=["red", "green", "blue", "purple", "black", "brown", "pink", "gray", "olive"]))
    axs.set_xlim(-0.01, 1.01)
    axs.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    axs.set_xlabel(x_axis_column, fontsize=35)
    axs.set_ylabel(y_axis_column, fontsize=35)
    axs.patch.set_edgecolor("black")
    axs.patch.set_linewidth("2")
    axs.patch.set_edgecolor("black")
    axs.patch.set_linewidth("2")
    line_types = ["--", "-.", ":"]
    graphnumber = 0
    axs.tick_params(axis="both", which="major", labelsize=25)
    for sql_statement, legend_name in zip(sql_statements, legend_names):
        df = fetch_dataframe_from_server(sql_statement)
        axs.plot(df["impact_factor"], df["result"], "X" + line_types[graphnumber % 3], label=legend_name, linewidth=2, )
        graphnumber += 1

    for baseline_name, sql_statement in baselines:
        df = fetch_dataframe_from_server(sql_statement)
        y = list(df["result"].values) * 21
        axs.plot(list(np.arange(0.0, 1.05, 0.05)), y, ":", label=baseline_name, linewidth=8)

    plt.legend(fontsize=30, loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)

    plt.savefig(title + ".png", bbox_inches="tight")
