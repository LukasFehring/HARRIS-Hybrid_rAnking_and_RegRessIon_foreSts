import configparser

import pandas as pd

import database_utils


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open("conf/experiment_configuration.cfg"))
    return config


def fetch_dataframe_from_server(sql_statement):
    db_config = load_configuration()
    db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(db_config)

    return pd.read_sql(sql_statement, db_handle)
