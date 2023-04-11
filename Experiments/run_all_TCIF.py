import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lineartree import LinearTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm.auto import tqdm
from setuptools.glob import glob
import lightgbm as lgb

from TCIF.classes.T_CIF_observation import T_CIF_observations

import run_animals as animals
import run_vehicles as vehicles

import psutil

import warnings

from TCIF.classes.T_CIF_space import T_CIF_space
from TCIF.classes.T_CIF_time import T_CIF_time

warnings.filterwarnings('ignore')

modules = {
    #"animals": animals,
    "vehicles": vehicles
}

parameters_randomForest_names = ["n_estimators", "max_depth", "bootstrap"]
parameters_randomForest = [
    [20, 100, 500, 1000],  # n_estimators
    [2, 3, 4, 5, 10, 15, 20],  # max_depth
    [True, False],  # bootstrap (False per TCIF)
]


parameters_lightgbm_names = ["n_estimators", "learning_rate", "objective"]
parameters_lightgbm = [
    [20, 100, 500, 1000],  # n_estimators
    [0.01, 0.05, 0.1, 0.2],  # learning_rate
    ["multiclassova"],  # objective
]

parameters_linearTree_names = ["base_estimator", "max_depth", "criterion", "max_bins"]
parameters_linearTree = [
    [RidgeClassifier()],  # base_estimator
    [2, 3, 4, 5, 10, 15, 20],  # max_depth
    ["hamming"],  # criterion crossentropy solo con predict_proba
    [25],  # max_bins
]

def compute_scores(y_true, y_pred):
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    accuracy = accuracy_score(y_true, y_pred)

    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")

    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    return ["accuracy", "f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"], \
        [accuracy, f1_micro, f1_macro, precision_micro, precision_macro, recall_micro, recall_macro]

def train_models(filename, arg, args_names, train_transformed, classe_train, validation_transformed, classe_validation,
                 test_transformed, classe_test, t_feature_extr):
    # Random forest
    bar = tqdm(list(itertools.product(*parameters_randomForest.copy())), desc="Training Random Forest", position=2, leave=False)
    for n_estimators, max_depth, bootstrap in bar:
        complete_filename = f"RF!{n_estimators}_{max_depth}_{bootstrap}!{filename}"
        if os.path.exists("results/" + complete_filename):
            continue
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                     random_state=arg[4], n_jobs=psutil.cpu_count(logical=False))

        start = datetime.now()
        clf.fit(train_transformed, classe_train)
        total_time = (datetime.now() - start).total_seconds() + t_feature_extr

        classe_predicted_validation = clf.predict(validation_transformed)
        classe_predicted_test = clf.predict(test_transformed)

        performance_names, performance_validation = compute_scores(classe_validation, classe_predicted_validation)
        _, performance_test = compute_scores(classe_test, classe_predicted_test)

        pd.DataFrame(
            [
                [n_estimators, max_depth, bootstrap] +
                list(arg) +
                performance_validation +
                performance_test +
                [total_time]
            ],

            columns=
            parameters_randomForest_names +
            args_names +
            [f"validation_{x}" for x in performance_names] +
            [f"test_{x}" for x in performance_names] +
            ["total_time"]
        ).to_csv("results/" + complete_filename, index=False)

    #lightgbm
    bar = tqdm(list(itertools.product(*parameters_lightgbm.copy())), desc="Training lightgbm", position=2, leave=False)
    for n_estimators, learning_rate, objective in bar:
        if len(set(classe_train)) == 2:
            objective = "binary"
        complete_filename = f"LGBM!{n_estimators}_{learning_rate}_{objective}!{filename}"
        if os.path.exists("results/" + complete_filename):
            continue

        clf = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, objective=objective,
                                     random_state=arg[4], n_jobs=psutil.cpu_count(logical=False))

        start = datetime.now()
        clf.fit(train_transformed, classe_train)
        total_time = (datetime.now() - start).total_seconds() + t_feature_extr

        classe_predicted_validation = clf.predict(validation_transformed)
        classe_predicted_test = clf.predict(test_transformed)

        performance_names, performance_validation = compute_scores(classe_validation, classe_predicted_validation)
        _, performance_test = compute_scores(classe_test, classe_predicted_test)

        pd.DataFrame(
            [
                [n_estimators, learning_rate, objective] +
                list(arg) +
                performance_validation +
                performance_test +
                [total_time]
            ],

            columns=
            parameters_lightgbm_names +
            args_names +
            [f"validation_{x}" for x in performance_names] +
            [f"test_{x}" for x in performance_names] +
            ["total_time"]
        ).to_csv("results/" + complete_filename, index=False)

    # linearTree
    """bar = tqdm(list(itertools.product(*parameters_linearTree.copy())), desc="Training LinearTree", position=2, leave=False)
    for base_estimator, max_depth, criterion, max_bins in bar:
        complete_filename = f"LinTree!{str(base_estimator)}_{max_depth}_{criterion}_{max_bins}!{filename}"
        if os.path.exists("results/" + complete_filename):
            continue

        clf = LinearTreeClassifier(base_estimator=base_estimator, max_depth=max_depth, criterion=criterion,
                                   max_bins=max_bins, n_jobs=psutil.cpu_count(logical=False))

        start = datetime.now()
        clf.fit(train_transformed, classe_train)
        total_time = (datetime.now() - start).total_seconds() + t_feature_extr

        classe_predicted_validation = clf.predict(validation_transformed)
        classe_predicted_test = clf.predict(test_transformed)

        performance_names, performance_validation = compute_scores(classe_validation, classe_predicted_validation)
        _, performance_test = compute_scores(classe_test, classe_predicted_test)

        pd.DataFrame(
            [
                [base_estimator, max_depth, criterion, max_bins] +
                list(arg) +
                performance_validation +
                performance_test +
                [total_time]
            ],

            columns=
            parameters_linearTree_names +
            args_names +
            [f"validation_{x}" for x in performance_names] +
            [f"test_{x}" for x in performance_names] +
            ["total_time"]
        ).to_csv("results/" + complete_filename, index=False)"""

def run_obs(train_unpack, validation_unpack, test_unpack, args, args_names, dataset_name):
    bar = tqdm(args, position=1, leave=False)
    for arg in bar:
        try:
            filename = f"OBS!{dataset_name}!{'_'.join([str(x) for x in arg])}.csv"
            bar.set_description(filename)

            tcif = T_CIF_observations(None, arg[0], arg[1], arg[2], arg[3], n_jobs=psutil.cpu_count(logical=False),
                                      seed=arg[4], verbose=False)

            id_train, classe_train, lat_train, lon_train, time_train = train_unpack
            id_validation, classe_validation, lat_validation, lon_validation, time_validation = validation_unpack
            id_test, classe_test, lat_test, lon_test, time_test = test_unpack
            train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
            validation = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_validation, lon_validation, time_validation)]
            test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

            try:
                start = datetime.now()
                train_transformed = tcif.transform(train)
                t_feature_extr = (datetime.now() - start).total_seconds()
            except Exception as e:
                #print(e)
                continue

            validation_transformed = tcif.transform(validation)
            test_transformed = tcif.transform(test)

            train_models(filename, arg, args_names, train_transformed, classe_train, validation_transformed,
                         classe_validation, test_transformed, classe_test, t_feature_extr)
        except:
            pass




def run_time(train_unpack, validation_unpack, test_unpack, args, args_names, dataset_name):
    bar = tqdm(args, position=1, leave=False)
    for arg in bar:
        try:
            filename = f"OBS!{dataset_name}!{'_'.join([str(x) for x in arg])}.csv"
            bar.set_description(filename)

            tcif = T_CIF_time(None, arg[0], arg[1], arg[2], arg[3], n_jobs=psutil.cpu_count(logical=False),
                                      seed=arg[4], verbose=False)

            id_train, classe_train, lat_train, lon_train, time_train = train_unpack
            id_validation, classe_validation, lat_validation, lon_validation, time_validation = validation_unpack
            id_test, classe_test, lat_test, lon_test, time_test = test_unpack
            train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
            validation = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_validation, lon_validation, time_validation)]
            test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

            try:
                start = datetime.now()
                train_transformed = tcif.transform(train)
                t_feature_extr = (datetime.now() - start).total_seconds()
            except Exception as e:
                # print(e)
                continue

            validation_transformed = tcif.transform(validation)
            test_transformed = tcif.transform(test)

            train_models(filename, arg, args_names, train_transformed, classe_train, validation_transformed,
                         classe_validation, test_transformed, classe_test, t_feature_extr)
        except:
            pass


def run_space(train_unpack, validation_unpack, test_unpack, args, args_names, dataset_name):
    bar = tqdm(args, position=1, leave=False)
    for arg in bar:
        try:
            filename = f"OBS!{dataset_name}!{'_'.join([str(x) for x in arg])}.csv"
            bar.set_description(filename)

            tcif = T_CIF_space(None, arg[0], arg[1], arg[2], arg[3], n_jobs=psutil.cpu_count(logical=False),
                                      seed=arg[4], verbose=False)

            id_train, classe_train, lat_train, lon_train, time_train = train_unpack
            id_validation, classe_validation, lat_validation, lon_validation, time_validation = validation_unpack
            id_test, classe_test, lat_test, lon_test, time_test = test_unpack
            train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
            validation = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_validation, lon_validation, time_validation)]
            test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

            try:
                start = datetime.now()
                train_transformed = tcif.transform(train)
                t_feature_extr = (datetime.now() - start).total_seconds()
            except Exception as e:
                # print(e)
                continue

            validation_transformed = tcif.transform(validation)
            test_transformed = tcif.transform(test)

            train_models(filename, arg, args_names, train_transformed, classe_train, validation_transformed,
                         classe_validation, test_transformed, classe_test, t_feature_extr)
        except:
            pass


if __name__ == '__main__':

    bar = tqdm(modules.items(), position=0, leave=False)
    for dataset_name, module in bar:
        bar.set_description(f"Processing dataset {dataset_name}")

        bar_exp = tqdm(range(3), position=1, leave=False)

        bar_exp.set_description("Running TCIF-obs")
        par, parameters_names, (train, validation, test) = module.generate_obs()
        run_obs(train, validation, test, par, parameters_names, dataset_name)
        bar.update(1)

        bar_exp.set_description("Running TCIF-time")
        par, parameters_names, (train, validation, test) = module.generate_time()
        run_time(train, validation, test, par, parameters_names, dataset_name)
        bar.update(2)

        bar_exp.set_description("Running TCIF-time")
        par, parameters_names, (train, validation, test) = module.generate_space()
        run_space(train, validation, test, par, parameters_names, dataset_name)
        bar.update(3)

