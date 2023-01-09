import math
import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

import Cage8.BasicFeatures as bf
import Cage8.AggregateFeatures as af

from abc import ABC, abstractmethod


class T_CIF(BaseEstimator, ClassifierMixin, ABC):
    # interval types: {rp: random padding, p: perc}
    # index criterion: {n: #features, s: space, t: time interval}
    def __init__(self, n_trees, n_interval, min_length, interval_type="rp", seed=42):
        self.intervals = []
        self.n_trees = n_trees
        self.n_interval = n_interval
        self.min_length = min_length
        self.seed = seed

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
    def get_subset(self, X_row, start, stop):
        pass

    def _transform(self, X, starts, stops):
        features = []

        for (X_lat, X_lon, X_time) in X:
            feature = []
            for start, stop in zip(starts, stops):
                X_lat_sub = self.get_subset(X_lat, start, stop)
                X_lon_sub = self.get_subset(X_lon, start, stop)
                X_time_sub = self.get_subset(X_time, start, stop)

                feature.append(af.max(bf.speed(X_lat_sub, X_lon_sub, X_time_sub), None))
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
