from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import CaGeo.algorithms.AggregateFeatures as af
import CaGeo.algorithms.BasicFeatures as bf
import CaGeo.algorithms.SegmentFeatures as sf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm
import warnings

class T_CIF(BaseEstimator, ClassifierMixin, ABC):
    # interval types: {None: [a:b] or [0] if a > len(trj), percentage: percentage, reverse_fill: if a | b > len(trj),
    # reverse trj}
    def __init__(self, n_trees, n_interval, min_length, max_length, interval_type=None, seed=42, n_jobs=24,
                 verbose=False):
        self.intervals = []
        self.n_trees = n_trees
        self.n_interval = n_interval
        self.min_length = min_length
        self.max_length = max_length
        self.interval_type = interval_type
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.starts = None
        self.stops = None
        self.X = None
        self.clf = None

        self.executor = ProcessPoolExecutor(n_jobs)

    @abstractmethod
    def generate_intervals(self):
        pass

    @abstractmethod
    def get_subset(self, X_row, start, stop):
        pass

    def _transform(self, X, starts, stops):
        features = []

        # for (X_lat, X_lon, X_time) in tqdm(X, disable=not self.verbose, desc="Processing TS", leave=False,
        # position=0):
        for feature in tqdm(self.executor.map(_transform_inner_loop, X, repeat(starts), repeat(stops),
                                         repeat(str(type(self))), repeat(self.min_length), repeat(self.interval_type),
                                         ), total=len(X), disable=not self.verbose):
            features.append(feature)

        return np.array(features)[:, :, 0]

    def fit(self, X, y):  # list of triplets (lat, lon, time)
        self.X = X

        self.starts, self.stops = self.generate_intervals()

        self.clf = RandomForestClassifier(n_estimators=self.n_trees, max_depth=10, bootstrap=False,
                                          random_state=self.seed, n_jobs=self.n_jobs)

        self.clf.fit(self._transform(X, self.starts, self.stops), y)

        return self

    def predict(self, X):
        return self.clf.predict(self._transform(X, self.starts, self.stops))

    @abstractmethod
    def print_sections(self):
        pass


def _transform_inner_loop(X, starts, stops, tipo, min_length, interval_type):
    verbose = False

    if not verbose:
        warnings.filterwarnings('ignore')

    X_lat, X_lon, X_time = X
    if "time" in tipo:
        from TCIF.classes.T_CIF_time import T_CIF_time
        get_subset = T_CIF_time(None, None, min_length, interval_type=interval_type).get_subset
    elif "space" in tipo:
        from TCIF.classes.T_CIF_space import T_CIF_space
        get_subset = T_CIF_space(None, None, min_length, interval_type=interval_type).get_subset
    else:
        from TCIF.classes.T_CIF_features import T_CIF_features
        get_subset = T_CIF_features(None, None, min_length, interval_type=interval_type).get_subset

    feature = []
    for start, stop in tqdm(list(zip(starts, stops)), disable=not verbose, desc="Processing interval",
                            leave=False, position=1):
        X_lat_sub, X_lon_sub, X_time_sub = get_subset((X_lat, X_lon, X_time), start, stop)
        dist = np.nan_to_num(bf.distance(X_lat_sub, X_lon_sub, accurate=False))

        transformed = [
            np.nan_to_num(bf.speed(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:])),
            dist,
            np.nan_to_num(bf.direction(X_lat_sub, X_lon_sub)),
            np.nan_to_num(bf.turningAngles(X_lat_sub, X_lon_sub)),
            np.nan_to_num(bf.acceleration(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:])),
            np.nan_to_num(bf.acceleration2(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:]))
        ]

        for arr in tqdm(transformed, disable=not verbose, desc="computing aggregate features",
                        leave=False, position=2):
            for f in [af.sum, af.std, af.max, af.min, af.cov, af.var]:
                feature.append(f(arr, None))
            feature.append(np.array([arr.mean()]))  # mean
            feature.append(af.rate_below(arr, arr.mean() * .25, None))
            feature.append(af.rate_upper(arr, arr.mean() * .75, None))

        feature.append([np.nan_to_num(sf.straightness(X_lat_sub, X_lon_sub))])
        feature.append([np.nan_to_num(sf.meanSquaredDisplacement(X_lat_sub, X_lon_sub))])
        feature.append([np.nan_to_num(sf.intensityUse(X_lat_sub, X_lon_sub))])
        feature.append([np.nan_to_num(sf.sinuosity(X_lat_sub, X_lon_sub))])

    for f in feature:
        if len(f) == 0:
            print("HERE")
        if np.isnan(f[0]).all() or not np.isfinite(f[0]).all():
            print("HERE")
    return feature
