import os
import random

import numpy as np

from TCIF.classes.T_CIF import T_CIF


class T_CIF_features(T_CIF):

    # interval types: {rp: random padding, p: perc}
    def __init__(self, n_trees, n_interval, min_length, interval_type="rp", seed=42, verbose=False):
        super().__init__(n_trees, n_interval, min_length, seed, verbose)

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

            starting_p = [random.uniform(0.0, 1.0 - self.min_length) for _ in range(self.n_interval)]

            ending_p = []
            for p in starting_p:
                l = random.uniform(self.min_length, 1.0 - p) + p

                ending_p.append(l)

            return starting_p, ending_p

    def get_subset(self, X_row, start, stop):
        if self.interval_type == "rp":
            return_value = tuple([x[start:min(stop, len(x))] for x in X_row])

            if len(return_value[0]) == 0:  # special case: start > len(trajectory)
                return_value = (X_row[0][-1:], X_row[1][-1:], X_row[2][-1:])

            topad = stop - min(stop, len(X_row[0]))

            return_value = tuple(np.hstack((x, x[-1] * np.ones(topad))) for x in return_value)

            return return_value


        elif self.interval_type == "p":
            length = len(X_row[0])
            start = int(length * start)
            stop = int(length * stop)

            return tuple([x[start:stop] for x in X_row])

    def print_sections(self):
        width = os.get_terminal_size().columns

        max_w = max([len(x[0]) for x in self.X])

        c = width / max_w

        print("".join(["#" for _ in range(width)]))

        for start, stop in zip(self.starts, self.stops):
            to_print = []

            if self.interval_type == "rp":
                to_print = [" " for _ in range(int(start * c))]

                to_print += ["-" for _ in range(int(start * c), int(stop * c))]

            if self.interval_type == "p":
                to_print = [" " for _ in range(int(start * width))]

                to_print += ["-" for _ in range(int(start * width), int(stop * width))]

            print("".join(to_print))
