import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pandas as pd

from T_CIF_features import T_CIF_features


def preare(df, tid_list):
    id = []
    lat = []
    lon = []
    time = []
    classe = []

    for _id, _classe in df[df.tid.isin(tid_list)][["tid", "class"]].groupby(
            by=["tid", "class"]).max().reset_index().values:
        df_result = df[df.tid == _id]
        id.append(_id)
        classe.append(_classe)

        _lat = []
        _lon = []
        _time = []
        for _lat_el, _lon_el, _time_el in df_result[["c1", "c2", "t"]].values:
            _time.append(_time_el)
            _lat.append(_lat_el)
            _lon.append(_lon_el)

        lat.append(np.array(_lat))
        lon.append(np.array(_lon))
        time.append(np.array(_time))

    return id, classe, lat, lon, time

if __name__ == "__main__":
    df = pd.read_csv("./vehicles.zip")

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                 df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 test_size=.3,
                                                 stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 random_state=3)

    id_train, classe_train, lat_train, lon_train, time_train = preare(df, tid_train)
    id_test, classe_test, lat_test, lon_test, time_test = preare(df, tid_test)

    tcif = T_CIF_features(n_trees=500, n_interval=5, min_length=10,
                          interval_type="rp")  # n_trees, n_interval, min_length, interval_type="rp", seed=42

    train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
    test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

    tcif.fit(train, y=classe_train)

    y_pred_training = tcif.predict(train)

    y_pred_test = tcif.predict(test)

    print(classification_report(classe_train, y_pred_training))

    print(classification_report(classe_test, y_pred_test))
