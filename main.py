import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pandas as pd

from TCIF.algorithms.utils import preare
from TCIF.classes.T_CIF_features import T_CIF_features



if __name__ == "__main__":
    df = pd.read_csv("vehicles.zip")

    df[["c1", "c2"]] = df[["c1", "c2"]]/100000

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                 df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 test_size=.3,
                                                 stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 random_state=3)

    id_train, classe_train, lat_train, lon_train, time_train = preare(df, tid_train)
    id_test, classe_test, lat_test, lon_test, time_test = preare(df, tid_test)

    tcif = T_CIF_features(n_trees=500, n_interval=20, min_length=10, interval_type="rp", verbose=True)
    #tcif = T_CIF_features(n_trees=1000, n_interval=20, min_length=.05, interval_type="p", verbose=True)

    train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
    test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

    tcif.fit(train, y=classe_train)

    y_pred_training = tcif.predict(train)

    y_pred_test = tcif.predict(test)

    print(classification_report(classe_train, y_pred_training))

    print(classification_report(classe_test, y_pred_test))

    tcif.print_sections()
