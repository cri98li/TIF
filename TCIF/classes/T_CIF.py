import math
import random
import os

import numpy as np
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

import CaGeo.algorithms.BasicFeatures as bf
import CaGeo.algorithms.AggregateFeatures as af

from abc import ABC, abstractmethod


class T_CIF(BaseEstimator, ClassifierMixin, ABC):
    # interval types: {rp: random padding, p: perc}
    # index criterion: {n: #features, s: space, t: time interval}
    def __init__(self, n_trees, n_interval, min_length, interval_type="rp", X_th=None, measure_f=None, seed=42,
                 verbose=False):
        self.intervals = []
        self.n_trees = n_trees
        self.n_interval = n_interval
        self.min_length = min_length
        self.X_th=X_th
        self.measure_f=measure_f
        self.seed = seed
        self.verbose = verbose

        self.starts = None
        self.stops = None
        self.X = None
        self.clf = None

        if interval_type == "rp":
            self.interval_type = interval_type
            if type(min_length) != int:
                raise ValueError(f"min_length={type(min_length)} unsupported when interval_type={interval_type}. Please"
                                 f" use min_length=int")

        elif interval_type == "p":
            self.interval_type = interval_type
            if type(min_length) != float:
                raise ValueError(
                    f"min_length={type(min_length)} unsupported when interval_type={interval_type}. Please"
                    f" use min_length=float")

        self.interval_type = interval_type if interval_type in ["rp", "p"] else "rp"

    @abstractmethod
    def generate_intervals(self):
        pass

    @abstractmethod
    def get_subset(self, X_row, start, stop, X_th, measure_f):
        pass

    def _transform(self, X, starts, stops):
        features = []

        for (X_lat, X_lon, X_time) in tqdm(X, disable=not self.verbose, desc="Processing TS", leave=False, position=0):
            feature = []
            for start, stop in tqdm(list(zip(starts, stops)), disable=not self.verbose, desc="Processing interval", leave=False, position=1):
                X_lat_sub = self.get_subset(X_lat, start, stop, self.X_th, self.measure_f)
                X_lon_sub = self.get_subset(X_lon, start, stop, self.X_th, self.measure_f)
                X_time_sub = self.get_subset(X_time, start, stop, self.X_th, self.measure_f)

                feature.append(af.max(np.nan_to_num(bf.speed(X_lat_sub, X_lon_sub, X_time_sub)), None))
                feature.append(af.min(np.nan_to_num(bf.speed(X_lat_sub, X_lon_sub, X_time_sub)), None))
                feature.append(af.sum(np.nan_to_num(bf.speed(X_lat_sub, X_lon_sub, X_time_sub))/len(X_lat_sub), None))
                feature.append(af.max(np.nan_to_num(bf.direction(X_lat_sub, X_lon_sub)), None))
                feature.append(af.min(np.nan_to_num(bf.direction(X_lat_sub, X_lon_sub)), None))

                feature.append(af.max(np.nan_to_num(bf.distance(X_lat_sub, X_lon_sub)), None))
            #feature = np.hstack([np.array(x) for x in feature])
            features.append(feature)

        return np.array(features)[:, :, 0]

    def fit(self, X, y):  # list of triplets (lat, lon, time)
        self.X = X

        self.starts, self.stops = self.generate_intervals()

        self.clf = RandomForestClassifier(n_estimators=self.n_trees, max_depth=2)

        self.clf.fit(self._transform(X, self.starts, self.stops), y)

        return self

    def predict(self, X):
        return self.clf.predict(self._transform(X, self.starts, self.stops))

    def print_sections(self):
        width = os.get_terminal_size().columns

        max_w = max([len(x[0]) for x in self.X])

        c = width/max_w

        print("".join(["#" for _ in range(width)]))

        for start, stop in zip(self.starts, self.stops):
            toPrint = []

            if self.interval_type == "rp":

                toPrint = [" " for _ in range(int(start * c))]

                toPrint += ["-" for _ in range(int(start * c), int(stop * c))]

            if self.interval_type == "p":
                toPrint = [" " for _ in range(int(start * width))]

                toPrint += ["-" for _ in range(int(start * width), int(stop * width))]

            print("".join(toPrint))
