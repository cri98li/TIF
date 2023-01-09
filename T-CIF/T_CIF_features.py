import math
import random
from abc import ABC

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

import Cage8.BasicFeatures as bf
import Cage8.AggregateFeatures as af

from T_CIF import T_CIF


class T_CIF_features(T_CIF):

    # interval types: {rp: random padding, p: perc}
    def __init__(self, n_trees, n_interval, min_length, interval_type="rp", seed=42):
        super().__init__(n_trees, n_interval, min_length, interval_type, seed)

    def generate_intervals(self):
        if self.interval_type == "rp":
            max_len = max([len(x[0]) for x in self.X])

            random.seed(self.seed)

            starting_p = random.sample(range(0, max_len - self.min_length), self.n_interval)
            ending_p = []
            for p in starting_p:
                l = random.randint(self.min_length, max_len - p) + p

                ending_p.append(l)

            return starting_p, ending_p

        elif self.interval_type == "p":
            random.seed(self.seed)

            starting_p = random.sample(range(0, 1.0 - self.min_length), self.n_interval)
            ending_p = []
            for p in starting_p:
                l = random.randint(self.min_length, 1.0 - p) + p

                ending_p.append(l)

            return starting_p, ending_p

    def get_subset(self, X_row, start, stop):
        if self.interval_type == "rp":
            return_value = X_row[start:min(stop, len(X_row))]

            return np.hstack((return_value, X_row[-1] * np.ones(max(0, stop-len(X_row)))))


        elif self.interval_type == "p":
            return np.zeros(0)
