import hashlib
import os
import random

import numpy as np

from TCIF.classes.T_CIF import T_CIF

from CaGeo.algorithms.BasicFeatures import distance

class T_CIF_space(T_CIF):

    def __init__(self, n_trees, n_interval, min_length, max_length=np.inf, interval_type="percentage", seed=42,
                 verbose=False):
        super().__init__(
            n_trees=n_trees,
            n_interval=n_interval,
            min_length=min_length,
            max_length=max_length,
            interval_type=interval_type,
            seed=seed,
            verbose=verbose
        )
        self.X_distance_dict = dict()

        self.min_s = None
        self.max_s = None

        if verbose:
            print(f"{interval_type}, min:{min_length}, max:{max_length}", flush=True)

        if interval_type is None:
            if type(min_length) != int or (type(max_length) != int and max_length != np.inf):
                raise ValueError(f"min_length={type(min_length)} and max_length={type(min_length)} unsupported when "
                                 f"interval_type={interval_type}. Please use min_length=int and None=[int or inf]")

        elif interval_type == "percentage":
            if type(min_length) != float or type(max_length) != float:
                raise ValueError(f"min_length={type(min_length)} and max_length={type(min_length)} unsupported when "
                                 f"interval_type={interval_type}. Please use min_length=float and None=[float or inf]")

        elif interval_type == "reverse_fill":
            if type(min_length) != int or (type(max_length) != int and max_length != np.inf):
                raise ValueError(f"min_length={type(min_length)} and max_length={type(min_length)} unsupported when "
                                 f"interval_type={interval_type}. Please use min_length=int and None=[int or inf]")
        else:
            raise ValueError(f"interval_type={interval_type} unsupported. supported interval types: [None, percentage, "
                             f"reverse_fill]")

        self.interval_type = interval_type

    def generate_intervals(self):
        random.seed(self.seed)

        self.max_s = int(max([distance(x[0], x[1]).sum() for x in self.X]))
        if self.max_s == 0:
            raise ValueError("int(max_distance) == 0")

        if self.interval_type in [None, "reverse_fill"]:
            starting_p = random.sample(range(0, self.max_s - self.min_length), self.n_interval)
            ending_p = []
            for p in starting_p:
                l = random.randint(self.min_length, self.max_s - p) + p

                ending_p.append(l)

            return starting_p, ending_p

        elif self.interval_type == "percentage":
            starting_p = [random.uniform(0.0, 1.0 - self.min_length) for _ in range(self.n_interval)]

            ending_p = []
            for p in starting_p:
                l = random.uniform(self.min_length, min(1.0 - p, self.max_length)) + p

                ending_p.append(l)

            return starting_p, ending_p

    def get_subset(self, X_row, start, stop):
        key = (
            hashlib.sha1(X_row[0]).hexdigest(),
            hashlib.sha1(X_row[1]).hexdigest(),
            hashlib.sha1(X_row[2]).hexdigest()
        )

        if key not in self.X_distance_dict:
            self.X_distance_dict[key] = distance(X_row[0], X_row[1]).sum()

        X_distance = self.X_distance_dict[key]

        if self.interval_type in [None, "percentage"]:

            if self.interval_type == "percentage":
                start = X_distance * start
                stop = X_distance * stop

            start_idx = None
            stop_idx = None

            cum_dist = 0
            for idx, dist in enumerate(distance(X_row[0], X_row[1])):
                cum_dist += dist
                if cum_dist >= start and start_idx is None:
                    start_idx = idx

                if cum_dist >= stop:
                    stop_idx = idx
                    break

            if start_idx is None:
                return tuple([x[0:1] for x in X_row])
            elif stop_idx is None:
                return tuple([x[start_idx:] for x in X_row])
            elif start_idx == stop_idx:
                return tuple([x[start_idx:stop_idx + 1] for x in X_row])
            else:
                return tuple([x[start_idx:stop_idx] for x in X_row])

        elif self.interval_type == "reverse_fill":
            return_value = ([], [], [])
            X_row_clone = (
                X_row[0],
                X_row[1],
                X_row[2]
            )

            subsequence_space = 0

            while subsequence_space < stop \
                    or len(
                return_value[0]) == 0:  # special case: at least 1 element -> in the case that delta_time > stop-start
                for it, (lat, lon, time, dist) in enumerate(zip(X_row_clone[0],
                                                                     X_row_clone[1],
                                                                     X_row_clone[2],
                                                                     distance(X_row_clone[0], X_row_clone[1]))):

                    if subsequence_space < start:
                        subsequence_space += dist
                        continue

                    if subsequence_space != 0 and it == 0:  # 1st element after a flip -> skip to avoid duplicates
                        continue

                    return_value[0].append(lat)
                    return_value[1].append(lon)
                    return_value[2].append(time)

                    subsequence_space += dist

                    if subsequence_space >= stop \
                            and len(return_value[0]) > 0:  # special case: at least 1 element -> in the case that delta_time > stop-start
                        break

                X_row_clone = (
                    np.flip(X_row_clone[0]),
                    np.flip(X_row_clone[1]),
                    np.flip(X_row_clone[2])
                )

            if len(return_value[0]) == 0:
                print("HERE")

            return (
                np.array(return_value[0]),
                np.array(return_value[1]),
                np.array(return_value[2]),
            )

    def print_sections(self):
        if self.interval_type in [None, "reverse_fill"]:
            width = os.get_terminal_size().columns

            c = width / (self.max_t - self.min_t)

            print("".join(["#" for _ in range(width)]))

            for start, stop in zip(self.starts, self.stops):
                to_print = [" " for _ in range(int((start - self.min_t) * c))]

                to_print += ["-" for _ in range(int((start - self.min_t) * c), int((stop - self.min_t) * c))]

                print("".join(to_print))
        else:
            raise Exception("Unimplemented")
