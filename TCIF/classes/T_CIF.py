from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import repeat
import math
import gc

from joblib import effective_n_jobs
from joblib import Parallel, delayed
from distutils.version import LooseVersion

import CaGeo.algorithms.AggregateFeatures as af
import CaGeo.algorithms.BasicFeatures as bf
import CaGeo.algorithms.SegmentFeatures as sf
import numpy as np
from memory_profiler import profile
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._base import _partition_estimators
from tqdm.auto import tqdm
import warnings

class T_CIF(BaseEstimator, ClassifierMixin, ABC):
    # interval types: {None: [a:b] or [0] if a > len(trj), percentage: percentage, reverse_fill: if a | b > len(trj),
    # reverse trj}
    def __init__(self, n_trees, n_interval, min_length, max_length, interval_type=None, accurate=False, lat_lon=False,
                 seed=42, n_jobs=24, verbose=False):

        if min_length > max_length:
            raise ValueError(f"min_length must be less then max_length. Values: (min={min_length}, max={max_length})")

        self.intervals = []
        self.n_trees = n_trees
        self.n_interval = n_interval
        self.min_length = min_length
        self.max_length = max_length
        self.interval_type = interval_type
        self.accurate = accurate
        self.include_lat_lon = lat_lon
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

    def _chunkize(self, X, n_chunks):
        chunk_size = max(math.floor(len(X) / n_chunks), 1)

        for i in range(0, len(X), chunk_size):
            yield deepcopy(X[i:i + chunk_size])




    def dividi(self, n_tr, n_chunks):
        chunk_size = max(math.floor(n_tr / n_chunks), 1)

        divisi = []
        for i in range(0, n_tr, chunk_size):
            divisi.append((i, i + chunk_size))

        return divisi
    #@profile
    def transform(self, X):
        self.X = X

        if self.starts is None:
            self.starts, self.stops = self.generate_intervals()

        features = []
        # for (X_lat, X_lon, X_time) in tqdm(X, disable=not self.verbose, desc="Processing TS", leave=False,
        # position=0):

        n_chunk = self.n_jobs

        """for feature in tqdm(self.executor.map(_transform_inner_loop, self._chunkize(X, n_chunk),
                                              repeat(self.starts), repeat(self.stops), repeat(str(type(self))),
                                              repeat(self.min_length), repeat(self.interval_type),
                                              repeat(self.accurate), repeat(self.include_lat_lon)
                                              ),
                            total=n_chunk, disable=not self.verbose):
            features += feature"""

        """chunk_size = max(math.floor(len(X) / self.n_jobs), 1)

        X_chunk_list = []
        for i in range(0, len(X), chunk_size):
            X_chunk_list.append(X[i:i + chunk_size])
        processes = []


        for X_chunk in self._chunkize(X, self.n_jobs):
            processes.append(self.executor.submit(_transform_inner_loop,
                                                  X_chunk,
                                                  self.starts,
                                                  self.stops,
                                                  str(type(self)),
                                                  self.min_length,
                                                  self.interval_type,
                                                  self.accurate,
                                                  self.include_lat_lon,
                                                  "ciao"
                                                  )
                             )
        for process in tqdm(processes):
            res, _ = process.result()
            features += res
            del process"""


        #divisi = self.dividi(len(X), n_chunk)

        chunk_size = max(math.floor(len(X) / self.n_jobs), 1)

        X_chunk_list = []
        for i in range(0, len(X), chunk_size):
            X_chunk_list.append(deepcopy(X[i:i + chunk_size]))

        estimators = Parallel(n_jobs=self.n_jobs, verbose=1, prefer="processes")(
            delayed(_transform_inner_loop)(X_chunk, self.starts, self.stops, str(type(self)), self.min_length,
                                           self.interval_type, self.accurate, self.include_lat_lon)
            for i, X_chunk in enumerate(X_chunk_list))


        for res in estimators:
            features.append(res)

            #print(name)


        

        return np.concatenate(features)#np.array(features)[:, :, 0]

    def fit(self, X, y):  # list of triplets (lat, lon, time)
        self.X = X

        if self.starts is None:
            self.starts, self.stops = self.generate_intervals()

        self.clf = RandomForestClassifier(n_estimators=self.n_trees, max_depth=10, bootstrap=False,
                                          random_state=self.seed, n_jobs=self.n_jobs)

        self.clf.fit(self.transform(X), y)

        return self

    def get_col_names(self):
        final_features = []
        for i in range(len(self.starts)):
            base_features = ["speed", "dist", "direction", "turningAngles", "acceleration", "acceleration2"]
            if self.include_lat_lon:
                base_features += ["lat", "lon", "time"]

            for base_feature in base_features:
                for aggragate_feature in ["sum", "std", "max", "min", "cov", "var", "mean", "rate_b", "rate_u"]:
                    final_features.append(f"{aggragate_feature}({base_feature}_{i})")

            final_features += [f"straightness_{i}", f"meanSquaredDisplacement_{i}", f"intensityUse_{i}", f"sinuosity_{i}"]

        return final_features

    def predict(self, X):
        return self.clf.predict(self.transform(X))

    @abstractmethod
    def print_sections(self):
        pass

#@profile
def _transform_inner_loop(X_list, starts, stops, tipo, min_length, interval_type, accurate, include_lat_lon):
    #import sys
    #print(sys.getsizeof(X_list))
    verbose = False

    if not verbose:
        warnings.filterwarnings('ignore')

    n_base = 6
    n_base += 3 if include_lat_lon else 0

    features = np.zeros((len(X_list), (n_base*9+4)*len(starts)))*np.nan

    for i, X in enumerate(X_list):
        X_lat, X_lon, X_time = X

        if "time" in tipo:
            from TCIF.classes.T_CIF_time import T_CIF_time
            get_subset = T_CIF_time(None, None, min_length, interval_type=interval_type).get_subset
        elif "space" in tipo:
            from TCIF.classes.T_CIF_space import T_CIF_space
            get_subset = T_CIF_space(None, None, min_length, interval_type=interval_type).get_subset
        else:
            from TCIF.classes.T_CIF_observation import T_CIF_observations
            get_subset = T_CIF_observations(None, None, min_length, interval_type=interval_type).get_subset

        feature = []
        for start, stop in tqdm(list(zip(starts, stops)), disable=not verbose, desc="Processing interval",
                                leave=False, position=1):
            X_lat_sub, X_lon_sub, X_time_sub = get_subset((X_lat, X_lon, X_time), start, stop)
            dist = np.nan_to_num(bf.distance(X_lat_sub, X_lon_sub, accurate=accurate))*111.139

            transformed = [
                np.nan_to_num(bf.speed(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:])),
                dist,
                np.nan_to_num(bf.direction(X_lat_sub, X_lon_sub)),
                np.nan_to_num(bf.turningAngles(X_lat_sub, X_lon_sub)),
                np.nan_to_num(bf.acceleration(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:])),
                np.nan_to_num(bf.acceleration2(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:]))
            ]

            if include_lat_lon:
                transformed += [X_lat_sub, X_lon_sub, X_time_sub]

            for arr in tqdm(transformed, disable=not verbose, desc="computing aggregate features",
                            leave=False, position=2):
                for f in [af.sum, af.std, af.max, af.min, af.cov, af.var]:
                    feature.append(f(arr, None))
                feature.append(np.array([arr.mean()]))  # mean
                feature.append(af.rate_below(arr, arr.mean() * .25, None))
                feature.append(af.rate_upper(arr, arr.mean() * .75, None))

            del transformed[:]
            feature.append([np.nan_to_num(sf.straightness(X_lat_sub, X_lon_sub))])
            feature.append([np.nan_to_num(sf.meanSquaredDisplacement(X_lat_sub, X_lon_sub))])
            feature.append([np.nan_to_num(sf.intensityUse(X_lat_sub, X_lon_sub))])
            feature.append([np.nan_to_num(sf.sinuosity(X_lat_sub, X_lon_sub))])

        for f in feature:
            if len(f) == 0:
                print("HERE")
            if np.isnan(f[0]).all() or not np.isfinite(f[0]).all():
                print("HERE")

        #features.append(feature)
        features[i] = np.array(feature).T.reshape(-1)
        del feature


    return features

def _joblib_parallel_args(**kwargs):
    import joblib

    if joblib.__version__ >= LooseVersion('0.12'):
        return kwargs

    extra_args = set(kwargs.keys()).difference({'prefer', 'require'})
    if extra_args:
        raise NotImplementedError('unhandled arguments %s with joblib %s'
                                  % (list(extra_args), joblib.__version__))
        args = {}
        if 'prefer' in kwargs:
            prefer = kwargs['prefer']
        if prefer not in ['threads', 'processes', None]:
            raise ValueError('prefer=%s is not supported' % prefer)
        args['backend'] = {'threads': 'threading',
                           'processes': 'multiprocessing',
                           None: None}[prefer]

        if 'require' in kwargs:
            require = kwargs['require']
        if require not in [None, 'sharedmem']:
            raise ValueError('require=%s is not supported' % require)
        if require == 'sharedmem':
            args['backend'] = 'threading'
        return args

def _partition_estimators(n_estimators, n_jobs):

    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()