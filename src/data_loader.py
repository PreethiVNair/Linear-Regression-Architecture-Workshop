import pandas as pd
import sqlite3
import yaml
import requests


def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data


def load_database(database_path, query):
    connection = sqlite3.connect(database_path)

    data = pd.read_sql_query(query, connection)

    connection.close()

    return data


def load_config(config_path):
    """Load project settings from a YAML file."""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def load_api(api_url):
    """Load data from a web API."""

    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    data = response.json()

    return data


if __name__ == "__main__":
    print("data_loader.py is working.")