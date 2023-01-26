import os
import random

from TCIF.classes.T_CIF import T_CIF


class T_CIF_time(T_CIF):

    def __init__(self, n_trees, n_interval, min_length, seed=42,
                 verbose=False):
        super().__init__(n_trees, n_interval, min_length, seed, verbose)
        self.min_t = None
        self.max_t = None

    def generate_intervals(self):
        self.max_t = int(max([max(x[2]) for x in self.X]))
        self.min_t = int(min([min(x[2]) for x in self.X]))

        starting_p = random.sample(range(self.min_t, self.max_t - self.min_length), self.n_interval)
        ending_p = []
        for p in starting_p:
            l = random.randint(self.min_length, self.max_t - p) + p

            ending_p.append(l)

        return starting_p, ending_p

    def get_subset(self, X_row, start, stop):
        start_idx = None
        stop_idx = None

        for idx, (lat, lon, time) in enumerate(zip(X_row[0], X_row[1], X_row[2])):
            if time > start and start_idx is None:
                start_idx = idx

            if time > stop:
                stop_idx = idx
                break

        if start_idx is None:
            return tuple([x[0:1] for x in X_row])
        elif stop_idx is None:
            return tuple([x[start_idx:] for x in X_row])
        elif start_idx == stop_idx:
            return tuple([x[start_idx:stop_idx+1] for x in X_row])
        else:
            return tuple([x[start_idx:stop_idx] for x in X_row])


    def print_sections(self):
        width = os.get_terminal_size().columns

        c = width / (self.max_t - self.min_t)

        print("".join(["#" for _ in range(width)]))

        for start, stop in zip(self.starts, self.stops):

            to_print = [" " for _ in range(int((start - self.min_t) * c))]

            to_print += ["-" for _ in range(int((start - self.min_t) * c), int((stop - self.min_t) * c))]

            print("".join(to_print))
