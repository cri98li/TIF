import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from TCIF.algorithms.utils import prepare


def _prepare_dataset():
    df = pd.read_csv("datasets/vehicles.zip")

    #df["class"] = LabelEncoder().fit_transform(df["class"])

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                 df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 test_size=.2,
                                                 stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 random_state=3)

    df_train = df[df.tid.isin(tid_train)]

    tid_train, tid_validation, _, _ = train_test_split(df_train.groupby(by=["tid"]).max().reset_index()["tid"],
                                                 df_train.groupby(by=["tid"]).max().reset_index()["class"],
                                                 test_size=.2,
                                                 stratify=df_train.groupby(by=["tid"]).max().reset_index()["class"],
                                                 random_state=3)

    train = prepare(df, tid_train, verbose=False)
    validation = prepare(df, tid_validation, verbose=False)
    test = prepare(df, tid_test, verbose=False)
    return train, validation, test


def generate_obs():
    parameters_rp = [
        [3, 5, 10, 20, 50, 100],  # n_interval
        [5, 10, 20, 50, 100, 200],  # min_length
        [5, 10, 20, 50, 100, 200, 500, np.inf],  # max_length
        [None, "reverse_fill"],  # interval_type
        [1, 2, 3, 4, 5],  # seed
    ]

    parameters_p = [
        [3, 5, 10, 20, 50, 100],  # n_interval
        [.05, .10, .25, .50,],  # min_length
        [.05, .10, .25, .50, .75, 1.],  # max_length
        ["percentage"],  # interval_type
        [1, 2, 3, 4, 5],  # seed
    ]

    parameters_names = ["n_interval", "min_length", "max_length", "interval_type", "seed"]

    par = list(itertools.product(*parameters_rp)) + list(itertools.product(*parameters_p))

    par = [x for x in par if x[1] <= x[2]]

    return par, parameters_names, _prepare_dataset()

def generate_time():
    parameters_rp = [
        [3, 5, 10, 20, 50, 100],  # n_interval
        [5, 10, 20, 50, 100],  # min_length
        [5, 10, 20, 50, 100, 200, np.inf],  # max_length
        [None, "reverse_fill"],  # interval_type
        [1, 2, 3, 4, 5],  # seed
    ]

    parameters_p = [
        [3, 5, 10, 20, 50, 100],  # n_interval
        [.05, .10, .25, .50],  # min_length
        [.05, .10, .25, .50, .75, 1.],  # max_length
        ["percentage"],  # interval_type
        [1, 2, 3, 4, 5],  # seed
    ]

    parameters_names = ["n_interval", "min_length", "max_length", "interval_type", "seed"]

    par = list(itertools.product(*parameters_rp)) + list(itertools.product(*parameters_p))

    par = [x for x in par if x[1] <= x[2]]

    return par, parameters_names, _prepare_dataset()

def generate_space():
    parameters_rp = [
        [3, 5, 10, 20, 50, 100],  # n_interval
        [500, 1000, 5000, 20000],  # min_length
        [500, 1000, 5000, 20000, 50000, np.inf],  # max_length
        [None, "reverse_fill"],  # interval_type
        [1, 2, 3, 4, 5],  # seed
    ]

    parameters_p = [
        [3, 5, 10, 20, 50, 100],  # n_interval
        [.05, .10, .25, .50],  # min_length
        [.05, .10, .25, .50, .75, 1.],  # max_length
        ["percentage"],  # interval_type
        [1, 2, 3, 4, 5],  # seed
    ]

    parameters_names = ["n_interval", "min_length", "max_length", "interval_type", "seed"]

    par = list(itertools.product(*parameters_rp)) + list(itertools.product(*parameters_p))

    par = [x for x in par if x[1] <= x[2]]

    return par, parameters_names, _prepare_dataset()
