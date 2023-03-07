from datetime import datetime

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from TCIF.algorithms.utils import preare
from TCIF.classes.T_CIF_features import T_CIF_features
from TCIF.classes.T_CIF_space import T_CIF_space
from TCIF.classes.T_CIF_time import T_CIF_time

if __name__ == "__main__":
    df = pd.read_csv("Experiments/datasets/taxi.zip")
    df.tid = LabelEncoder().fit_transform(df.tid)
    df = df[df.tid < df.tid.max() * .7]

    #df[["c1", "c2"]] = df[["c1", "c2"]]/100000



    """df = pd.read_csv("Experiments/datasets/geolife.zip").rename(columns={ "label": "class", "lat": "c1", "lon": "c2", "time": "t"})
    df.tid = LabelEncoder().fit_transform(df.tid)
    df = df[df.tid < df.tid.max()*.2]
    remap_class = {
        "subway": "public",
        "motorcycle": "private",
        "run": "private",
        "walk": "private",
        "boat": "private",
        "airplane": "public",
        "train": "public",
        "car": "private",
        "taxi": "public",
        "bike": "private",
        "bus": "public"
    }
    df["class"] = df["class"].apply(lambda x: remap_class[x])"""

    #df = pd.read_csv("Experiments/datasets/seabird_prepared.csv")[["bird", "species", "lat", "lon", "date_time"]].rename(
    #    columns={"lat": "c1", "lon": "c2", "date_time": "t", "species": "class", "bird": "tid"})
    #df.t = df.t.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                 df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 test_size=.3,
                                                 stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 random_state=3)

    id_train, classe_train, lat_train, lon_train, time_train = preare(df, tid_train)
    id_test, classe_test, lat_test, lon_test, time_test = preare(df, tid_test)

    #tcif = T_CIF_features(n_trees=1000, n_interval=20, min_length=10, interval_type="reverse_fill", n_jobs=12, verbose=True)
    #tcif = T_CIF_features(n_trees=1000, n_interval=20, min_length=.05, max_length=.5, interval_type="percentage", n_jobs=12, verbose=True)

    tcif = T_CIF_time(n_trees=1000, n_interval=50, min_length=.05, interval_type="percentage", n_jobs=12, verbose=True)
    #tcif = T_CIF_time(n_trees=1000, n_interval=50, min_length=10, interval_type=None, n_jobs=12, verbose=True)
    #tcif = T_CIF_time(n_trees=1000, n_interval=50, min_length=10, interval_type="reverse_fill", n_jobs=12, verbose=True)

    #tcif = T_CIF_space(n_trees=1000, n_interval=50, min_length=.05, interval_type="percentage", n_jobs=12, verbose=True)
    #tcif = T_CIF_space(n_trees=1000, n_interval=50, min_length=10, interval_type=None, n_jobs=12, verbose=True)
    #tcif = T_CIF_space(n_trees=1000, n_interval=50, min_length=10, interval_type="reverse_fill", n_jobs=12, verbose=True)

    train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
    test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

    start = datetime.now()
    tcif.fit(train, y=classe_train)
    print((datetime.now() - start).total_seconds())

    y_pred_test = tcif.predict(test)

    print(classification_report(classe_test, y_pred_test, digits=3))

    #tcif.print_sections()
