import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import pandas as pd
import psutil as psutil
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from TCIF.algorithms.utils import preare
from TCIF.classes.T_CIF_features import T_CIF_features

np.seterr(divide='ignore', invalid='ignore')


def run(els, lat_train, lon_train, time_train, lat_test, lon_test, time_test, classe_train):
    tcif = T_CIF_features(n_trees=els[0], n_interval=els[1], min_length=els[2], interval_type=els[3],
                          verbose=False)

    train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
    test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

    start = time.time()
    tcif.fit(train, y=classe_train)
    stop = time.time()

    return [stop - start, tcif.predict(test)]


if __name__ == "__main__":
    datasets = [y for x in os.walk("datasets/") for y in glob(os.path.join(x[0], '*.zip'))]

    parameters_rp = [
        [100, 200, 500, 1000, 5000],  # n_trees
        [3, 5, 10, 20, 50, 100, 200],  # n_interval
        [5, 10, 20, 50, 100, 200],  # min_length
        ["rp"]  # interval_type
    ]

    parameters_p = [
        [100, 200, 500, 1000, 5000],  # n_trees
        [3, 5, 10, 20, 50, 100, 200],  # n_interval
        [.05, .10, .20, .30, .50, .70],  # min_length
        ["p"]  # interval_type
    ]

    parameters_names = ["n_trees", "n_interval", "min_length", "interval_type"]

    pbar_dataset = tqdm(datasets, position=0, leave=False)
    for dataset in pbar_dataset:
        df = pd.read_csv(dataset)
        dataset_name = dataset.split('\\')[-1].split('/')[-1]
        pbar_dataset.set_description(f"Dataset name: {dataset_name}")

        df[["c1", "c2"]] = df[["c1", "c2"]] / 100000

        tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                     df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     test_size=.3,
                                                     stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     random_state=3)

        id_train, classe_train, lat_train, lon_train, time_train = preare(df, tid_train)
        id_test, classe_test, lat_test, lon_test, time_test = preare(df, tid_test)

        pool = ProcessPoolExecutor(max(1, psutil.cpu_count(logical=True)))

        par = list(itertools.product(*parameters_rp)) + list(itertools.product(*parameters_p))

        els_bar = tqdm(pool.map(run,
                                par,
                                itertools.repeat(lat_train),
                                itertools.repeat(lon_train),
                                itertools.repeat(time_train),
                                itertools.repeat(lat_test),
                                itertools.repeat(lon_test),
                                itertools.repeat(time_test),
                                itertools.repeat(classe_train)
                                ), position=1, leave=False, total=len(par))

        result_all = []
        for els, (time, y_pred) in zip(par, els_bar):
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")
            accuracy = accuracy_score(classe_test, y_pred)
            f1 = f1_score(classe_test, y_pred, average="micro")
            recall = recall_score(classe_test, y_pred, average="micro")

            result_all.append(list(els)+[time, accuracy, f1, recall])
        pd.DataFrame(result_all,
                     columns=parameters_names + ["time", "accuracy", "f1", "recall"], index=None)\
            .to_csv("results/"+"features.csv")



