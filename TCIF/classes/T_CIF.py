from abc import ABC, abstractmethod

import CaGeo.algorithms.AggregateFeatures as af
import CaGeo.algorithms.BasicFeatures as bf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm


class T_CIF(BaseEstimator, ClassifierMixin, ABC):
    # interval types: {rp: random padding, p: perc}
    def __init__(self, n_trees, n_interval, min_length, seed=42,
                 verbose=False):
        self.intervals = []
        self.n_trees = n_trees
        self.n_interval = n_interval
        self.min_length = min_length
        self.seed = seed
        self.verbose = verbose

        self.starts = None
        self.stops = None
        self.X = None
        self.clf = None

    @abstractmethod
    def generate_intervals(self):
        pass

    @abstractmethod
    def get_subset(self, X_row, start, stop):
        pass

    def _transform(self, X, starts, stops):
        features = []

        for (X_lat, X_lon, X_time) in tqdm(X, disable=not self.verbose, desc="Processing TS", leave=False, position=0):
            feature = []
            for start, stop in tqdm(list(zip(starts, stops)), disable=not self.verbose, desc="Processing interval",
                                    leave=False, position=1):
                X_lat_sub, X_lon_sub, X_time_sub = self.get_subset((X_lat, X_lon, X_time), start, stop)

                transformed = [
                    np.nan_to_num(bf.speed(X_lat_sub, X_lon_sub, X_time_sub)),
                    np.nan_to_num(bf.distance(X_lat_sub, X_lon_sub, accurate=False)),
                    np.nan_to_num(bf.direction(X_lat_sub, X_lon_sub)),
                    np.nan_to_num(bf.acceleration(X_lat_sub, X_lon_sub, X_time_sub)),
                    np.nan_to_num(bf.acceleration2(X_lat_sub, X_lon_sub, X_time_sub))
                ]

                for arr in tqdm(transformed, disable=not self.verbose, desc="computing aggregate features",
                                leave=False, position=2):
                    for f in [af.sum, af.std, af.max, af.min, af.cov, af.var]:
                        feature.append(f(arr, None))
                    feature.append(np.array([arr.mean()]))  # mean
                    feature.append(af.rate_below(arr, arr.mean() * .25, None))
                    feature.append(af.rate_upper(arr, arr.mean() * .75, None))

                # feature.append(af.max(np.nan_to_num(bf.distance(X_lat_sub, X_lon_sub)), None))
                #   feature.append(af.max(np.nan_to_num(bf.distance(X_lat_sub, X_lon_sub)), None))
            # feature = np.hstack([np.array(x) for x in feature])
            features.append(feature)

        return np.array(features)[:, :, 0]

    def fit(self, X, y):  # list of triplets (lat, lon, time)
        self.X = X

        self.starts, self.stops = self.generate_intervals()

        self.clf = RandomForestClassifier(n_estimators=self.n_trees, max_depth=2, bootstrap=False)

        self.clf.fit(self._transform(X, self.starts, self.stops), y)

        return self

    def predict(self, X):
        return self.clf.predict(self._transform(X, self.starts, self.stops))

    @abstractmethod
    def print_sections(self):
        pass
