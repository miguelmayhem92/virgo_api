from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier

import optuna
import pickle
import os
import pandas as pd
import numpy as np

import mlflow
from pathlib import Path
from  mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature


from .custom_modules import configs, datafuncion, crossvalidation_low_finder


n_days = configs.data_configs.n_days
lag_days = configs.data_configs.lags
window = configs.data_configs.window
ref_price = configs.data_configs.ref_price
std_column = configs.data_configs.std_column
logdif_column = configs.data_configs.logdif_column
dates_to_label = configs.low_finder_configs.dates_to_label
n_trials = configs.low_finder_configs.n_trials
d_types = configs.low_finder_configs.d_types ## for xgboost

datasets = dict()
class train_bidfider(train = False):
    def __init__(self, train_new_model = False):
        self.train_new_model =train_new_model

    def train_new_bidfinder(self):
        if self.train_new_model:
            for code in dates_to_label.keys():

                data_stock = datafuncion.get_stock_data(stock_code = code, n_days = n_days, window = window, lags = lag_days)
                data_stock_ = datafuncion.shape_data(data_stock, code, ref_price, std_column, logdif_column)
                dict_code = dates_to_label[code]
                bids, asks = dict_code['bids'], dict_code['asks']
                data_stock_ = datafuncion.label_Bit_Ask(data = data_stock_, bids = bids, asks = asks)

                object_test = datafuncion.slicer_date_prepa(data = data_stock_,code = code, dates_to_label = dates_to_label)
                object_test.feature_engineering()
                object_test.get_slice()
                #for i in total_indexes:
                X_datas_train = list()
                y_datas_train = list()
                X_datas_test = list()
                y_datas_test = list()
                for index in object_test.total_indexes:
                    
                    object_test.slice_treated(slicex = index)
                    X = object_test.X_train_transformed
                    y = object_test.y_train
                    
                    if index != max(object_test.total_indexes):
                        X_datas_train.append(X)
                        y_datas_train.append(y)
                    elif index == max(object_test.total_indexes):
                        X_datas_test.append(X)
                        y_datas_test.append(y)
                        
                X_datas_train = np.concatenate(X_datas_train, axis=0)
                y_datas_train = np.concatenate(y_datas_train, axis=0)
                X_datas_test = np.concatenate(X_datas_test, axis=0)
                y_datas_test = np.concatenate(y_datas_test, axis=0)
                    
                datasets[code] = {'X_train': X_datas_train, 'y_train' : y_datas_train, 'X_test': X_datas_test, 'y_test' : y_datas_test }
                
                print('{0} code is done'.format(code))

            X_train = list()
            y_train = list()
            X_test = list()
            y_test = list()
            code_list_train = list()
            code_list_test = list()
            for code in dates_to_label.keys():
                
                X_train_array, y_train_array = datasets[code]['X_train'], datasets[code]['y_train']
                X_test_array, y_test_array = datasets[code]['X_test'], datasets[code]['y_test']
                
                X_train.append(X_train_array)
                y_train.append(y_train_array)
                X_test.append(X_test_array)
                y_test.append(y_test_array)
                code_list_train.append(list([code]*X_train_array.shape[0]))
                code_list_test.append(list([code]*X_test_array.shape[0]))
                
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)
            code_list_train = np.concatenate(code_list_train, axis=0)
            code_list_test = np.concatenate(code_list_test, axis=0)

            dataset_train = pd.DataFrame(X_train, columns = object_test.my_features + object_test.exeptions)
            dataset_train['target'] = y_train
            dataset_train['code'] = code_list_train
            dataset_test = pd.DataFrame(X_test, columns = object_test.my_features + object_test.exeptions)
            dataset_test['target'] = y_test
            dataset_test['code'] = code_list_test

            X_train, y_train = dataset_train[object_test.my_features], dataset_train['target']
            X_test, y_test = dataset_test[object_test.my_features], dataset_test['target']

            ### Training models using crossvalidation
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
            )

            func_objective = lambda trial: crossvalidation_low_finder.objective_rf(trial, X_train, y_train)
            optuna.logging.set_verbosity(0)
            study.optimize(func_objective, n_trials=n_trials)

            rf_model = BaggingClassifier(RandomForestClassifier(**study.best_params), n_estimators=5, bootstrap = False, n_jobs = -1).fit(X_train, y_train)

            random_forest_evaluate = datafuncion.evaluate(rf_model, dataset_test, X_test, y_test)
            random_forest_evaluate.get_error()


            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
            )

            func_objective = lambda trial: crossvalidation_low_finder.objective_adb(trial, X_train, y_train)
            optuna.logging.set_verbosity(0)
            study.optimize(func_objective, n_trials=n_trials)

            abc_model = BaggingClassifier(AdaBoostClassifier(**study.best_params), n_estimators=5, bootstrap = False, n_jobs = -1).fit(X_train, y_train)

            adaboost_evaluate = datafuncion.evaluate(abc_model, dataset_test, X_test, y_test)
            adaboost_evaluate.get_error()



            X_train = X_train.astype(d_types)
            X_test = X_test.astype(d_types)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
            )

            func_objective = lambda trial: crossvalidation_low_finder.objective_xgb(trial)
            optuna.logging.set_verbosity(0)
            study.optimize(func_objective, n_trials=n_trials)

            xgb_model = XGBClassifier(**study.best_params).fit(X_train, y_train)

            xgb_evaluate = datafuncion.evaluate(xgb_model, dataset_test, X_test, y_test)
            xgb_evaluate.get_error()

            ### summary
            metrics_json = {'Random_forest' : {
                'Precision':random_forest_evaluate.precision, 
                'Weighted Precision':random_forest_evaluate.weighted_precision, 
                'ROC AUC':random_forest_evaluate.roc_auc,
                },
            'AdaBoost' : {
                'Precision':adaboost_evaluate.precision, 
                'Weighted Precision':adaboost_evaluate.weighted_precision, 
                'ROC AUC':adaboost_evaluate.roc_auc,
                },
            'XGBoost' : {
                'Precision':xgb_evaluate.precision, 
                'Weighted Precision':xgb_evaluate.weighted_precision, 
                'ROC AUC':xgb_evaluate.roc_auc,
                }
            }

            data = dict()
            models = list()
            metrics = list()
            values = list()
            for key in metrics_json.keys():
                for metric in metrics_json[key].keys():
                    value = metrics_json[key][metric]
                    models.append(key)
                    metrics.append(metric)
                    values.append(value)
            data['models'] = models
            data['metrics'] = metrics
            data['values'] = values
            summary = pd.DataFrame(data)

            ## Saving model in MLflow
            ### creation of experiment or skip if it exists
            registered_model_name = f'{self.stock_code}_models'
            tags = {"used_model": "tensorflow"}
            desc = f"in this directory the models for {self.stock_code}"
            best_params = self.study.best_params

            experimet_name = f'{self.stock_code}_experiments'
            run_name = f'{self.stock_code}-run-{today_str}'
            artificat_path_run = f"{self.stock_code}-run"

            try:
                experiment_id = mlflow.create_experiment(
                    experimet_name,
                    artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
                )
            except:
                print('experiment exists')

            ### getting the exp Id of the experiment folder
            client = MlflowClient()
            experiments = client.search_experiments()

            for exp in experiments:
                exp = dict(exp)
                if exp['name'] == f'{self.stock_code}_experiments':
                    exp_id = exp['experiment_id']
                    break
            ### metrics to store
            mae_val = wide_window.get_metrics( self.model, source = 'validation')
            mae_test = wide_window.get_metrics( self.model, source = 'test')

            data_ = wide_window.total_data[-wide_window.input_width:]
            data_ = (data_  - train_mean) / train_std
            data_ = data_.values
            data_ = data_.reshape((1,data_.shape[0],data_.shape[1]))
            signature = infer_signature(data_, best_model.predict(data_))

            with mlflow.start_run(
                run_name = run_name,
                    experiment_id  = exp_id
                    ):

                mlflow.log_param('best_model_framework', self.best_model_framework)
                for paramx in best_params.keys():
                    mlflow.log_param(paramx, best_params[paramx])
                mlflow.log_param("train_mean", train_mean)
                mlflow.log_param("train_std", train_std)

                mlflow.log_metric("mae_val", mae_val)
                mlflow.log_metric("mae_test", mae_test)

                mlflow.tensorflow.log_model(
                    best_model, 
                    artifact_path= artificat_path_run,
                    registered_model_name = registered_model_name ,
                    signature = signature
                )

            ### staging the current model under the latest version
            time.sleep(30)
            print('waiting')
            filter_registered_model_name = "name='{}'".format(registered_model_name)
            versions = list()
            for mv in client.search_model_versions(filter_registered_model_name):
                versions.append(dict(mv)['version'])

            client.transition_model_version_stage(
                name=registered_model_name,
                version=max(versions),
                stage="Production",
                archive_existing_versions = True
                )

            message_result = ' '.join([
                message_result,
                'best model results, metrics and model saved in mlflow '
            ])

            mlflow.end_run()

            return (message_result)
        
