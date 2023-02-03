from .custom_modules import configs, datafuncion, modeling, crossvalidation
import datetime
import tensorflow as tf
import numpy as np
import optuna
import os

import mlflow
from pathlib import Path
from  mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import io
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

today_str = datetime.date.today().strftime("%Y-%m-%d")

n_days = configs.data_configs.n_days
lag_days = configs.data_configs.lags
window = configs.data_configs.window
ref_price = configs.data_configs.ref_price
std_column = configs.data_configs.std_column
logdif_column = configs.data_configs.logdif_column
split_config = configs.data_configs.split_config
OUT_STEPS = configs.data_configs.steps_to_predic
input_length = configs.data_configs.input_length
best_error = configs.data_configs.best_error
save_predictions_path = configs.data_configs.save_predictions_path
save_model_path = configs.data_configs.save_model_path
cross_validation_trials = configs.optimazation_configs.cross_validation_trials

today_str = datetime.date.today().strftime("%Y-%m-%d")

class run_supermodel():
    
    def __init__(self, stock_code, train_predict_bool = False, cross_validation = False, n_trials = 1, save_model = False):
        self.stock_code = stock_code
        self.train_predict_bool = train_predict_bool
        self.n_trials = n_trials
        self.cross_validation = cross_validation
        self.save_model = save_model

    def train_predict(self):

        ### some configs to save the model and study
        model_and_study_path = f'{save_model_path}/{today_str}/{self.stock_code}'

        if not os.path.exists(model_and_study_path):
            os.makedirs(model_and_study_path)

        self.path_best_model = f'{model_and_study_path}/best_model.h5'
        self.study_name = f"{model_and_study_path}/model_study3"  # unique identifier of the study.
        self.storage_name = "sqlite:///{}.db".format(self.study_name)

        self.predictions_path_file = f'{save_predictions_path}/{today_str}-{self.stock_code}-predictions.csv'

        ### getting the data
        raw_stock = datafuncion.get_stock_data(stock_code = self.stock_code, n_days = n_days, window = window, lags = lag_days)
        raw_stock = datafuncion.shape_data(raw_stock , 'stock', ref_price, std_column, logdif_column)

        ### feature engineering
        stock_data = datafuncion.data_eng_features(data = raw_stock)
        
        ### spliting data
        split_object = datafuncion.split_data(stock_data, split_config)

        column_indices = split_object.column_indices
        n = split_object.ndata
        train_df = split_object.train_df
        val_df  = split_object.val_df
        test_df = split_object.test_df
        num_features = split_object.num_features
        ### scaling
        split_object.scaling()
        train_mean = split_object.train_mean
        train_std = split_object.train_std
        train_df = split_object.train_df
        val_df  = split_object.val_df
        test_df = split_object.test_df
        ### train, val and test are already updated, we just need mean and std for later rescaling

        wide_window = datafuncion.WindowGenerator(
            total_data = stock_data, 
            raw_stock = raw_stock,
            train_df=train_df, 
            val_df=val_df, 
            test_df=test_df,
            input_width=input_length, 
            label_width=OUT_STEPS, 
            shift=OUT_STEPS,
            label_columns=['stock_logdif']
        )

        tf_version = tf.__version__
        len_data = len(raw_stock)

        message_result = ' '.join([
            f'the package is tensorflow{tf_version} and the code to search is {self.stock_code}'
            f'with data length {len_data}'
        ])  
        ############################################################
        ###### Train baseline Models  ###############
        ############################################################


        if self.train_predict_bool:

            multi_val_performance = {}
            self.multi_performance = {}
            best_model_framework = None
            best_error_study = None

            multi_linear_model = modeling.initial_models(OUT_STEPS, num_features).multi_linear_model
            history = modeling.compile_and_fit(multi_linear_model, wide_window)
            multi_val_performance['Linear'] = multi_linear_model.evaluate(wide_window.val)
            self.multi_performance['Linear'] = multi_linear_model.evaluate(wide_window.test, verbose=0)

            multi_dense_model = modeling.initial_models(OUT_STEPS, num_features).multi_linear_model
            history = modeling.compile_and_fit(multi_dense_model, wide_window)
            multi_val_performance['Dense'] = multi_dense_model.evaluate(wide_window.val)
            self.multi_performance['Dense'] = multi_dense_model.evaluate(wide_window.test, verbose=0)

            multi_conv_model = modeling.initial_models(OUT_STEPS, num_features).multi_conv_model
            history = modeling.compile_and_fit(multi_conv_model, wide_window)
            multi_val_performance['Conv'] = multi_conv_model.evaluate(wide_window.val)
            self.multi_performance['Conv'] = multi_conv_model.evaluate(wide_window.test, verbose=0)

            multi_lstm_model = modeling.initial_models(OUT_STEPS, num_features).multi_lstm_model
            history = modeling.compile_and_fit(multi_lstm_model, wide_window)
            multi_val_performance['LSTM'] = multi_lstm_model.evaluate(wide_window.val)
            self.multi_performance['LSTM'] = multi_lstm_model.evaluate(wide_window.test, verbose=0)

            feedback_model = modeling.initial_models(OUT_STEPS, num_features).feedback_model
            history = modeling.compile_and_fit(feedback_model, wide_window)
            multi_val_performance['AR LSTM'] = feedback_model.evaluate(wide_window.val)
            self.multi_performance['AR LSTM'] = feedback_model.evaluate(wide_window.test, verbose=0)

            ### base line models error
            
            metric_name = 'mean_absolute_error'
            metric_index = multi_linear_model.metrics_names.index('mean_absolute_error')
            self.val_mae = [v[metric_index] for v in multi_val_performance.values()]
            self.test_mae = [v[metric_index] for v in self.multi_performance.values()]

            best_model_framework =list(self.multi_performance.keys())[np.argmin(self.test_mae)]

            ###storing first training results ### to save in memory
            best_model_framework =list(self.multi_performance.keys())[np.argmin(self.test_mae)]
            baseline_validation_results = dict(zip(self.multi_performance.keys(), self.val_mae))
            baseline_test_results = dict(zip(self.multi_performance.keys(), self.test_mae))

            message_result = ' '.join([ 
                message_result,
                f', the best base-line model is: {best_model_framework}'
            ])

        if self.train_predict_bool and self.cross_validation:
    
            study = optuna.create_study(
                direction='minimize',
                study_name=self.study_name,
                storage=self.storage_name,
                load_if_exists=True,
                )
            objective = crossvalidation.objective_functions.ojective_functions[best_model_framework]
            func = lambda trial: objective(
                trial, OUT_STEPS, num_features, 
                wide_window, best_error, self.save_model, self.path_best_model)
            study.optimize(func, n_trials= self.n_trials)

            best_params_study = study.best_params  #### to save in memory
            best_error_study = study.best_value  #### to save in memory

            best_model = tf.keras.models.load_model(self.path_best_model)

            predictions, data_ = wide_window.expected_return(
                plot_col='stock_logdif',
                model = best_model,
                train_mean = train_mean, 
                train_std = train_std,
                )

            self.predictions = predictions
            self.data_ = data_

            final_result, some_history, prep_predictions = wide_window.get_futur_prices(
                predictions= self.predictions,
                steps_futur = OUT_STEPS, 
                stock_code= self.stock_code,
                OUT_STEPS= OUT_STEPS,
                train_std= train_std,
                train_mean = train_mean,
                lag_days=lag_days,
                )

            self.some_history = some_history
            self.prep_predictions = prep_predictions
            
            self.study = study
            self.model = best_model 
            final_result.to_csv(self.predictions_path_file)
            self.wide_window = wide_window

            message_result = ' '.join([
                message_result,
                'crossvalidation done'

            ])

        if self.save_model:

            ### creation of experiment or skip if it exists
            try:
                experiment_id = mlflow.create_experiment(
                    f'{self.stock_code}_experiments',
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
            mae_val = wide_window.get_metrics(self, self.model, source = 'validation')
            mae_test = wide_window.get_metrics(self, self.model, source = 'test')

            data_ = wide_window.total_data[-wide_window.input_width:]
            data_ = (data_  - train_mean) / train_std
            data_ = data_.values
            data_ = data_.reshape((1,data_.shape[0],data_.shape[1]))
            signature = infer_signature(data_, best_model.predict(data_))

            with mlflow.start_run(
                run_name = f'{self.stock_code}-run-{today_str}',
                 experiment_id  = exp_id
                 ):

                mlflow.log_param("mae", mae_val)
                mlflow.log_param("mae", mae_test)
                mlflow.tensorflow.log_model(
                    best_model, 
                    artifact_path=f"{self.stock_code}-run",
                    signature = signature
                )
            mlflow.end_run()
            
            ### Model Register
            try:
                client = MlflowClient()
                client.create_registered_model(self.stock_code)
            except:
                print('folder already exists')
            
            mlflow.tensorflow.log_model(best_model,
                artifact_path=f"{self.stock_code}",
                registered_model_name=f'{self.stock_code}_model',
                signature=signature)

            ### staging the current model under the latest version
            versions = list()
            for mv in client.search_model_versions(f'{self.stock_code}_model'):
                versions.append(dict(mv)['version'])

            client.transition_model_version_stage(
                name=f'{self.stock_code}_model',
                version=max(versions),
                stage="Production")
            
            message_result = ' '.join([
                message_result,
                'best model results, metrics and model saved in mlflow '
            ])

        return (message_result)

    def baseline_models_plot(self):
            x = np.arange(len(self.multi_performance))
            width = 0.3

            fig, ax = plt.subplots()

            p1 = ax.bar(x - 0.17, self.val_mae, width, label='Validation')
            p2 = ax.bar(x + 0.17, self.test_mae, width, label='Test')

            ax.set_xticks(ticks=x, labels=self.multi_performance.keys(),
                    rotation=45)
            ax.set_ylabel(f'MAE (average over all times and outputs)')
            ax.bar_label(p1, label_type='edge')
            ax.bar_label(p2, label_type='edge')

            ax.set_ylim(min(self.val_mae + self.test_mae)-0.05, max(self.val_mae + self.test_mae) + 0.05)
            _ = plt.legend()

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            return img_buf

    def hiperparamter_plot(self):
        results = self.study.trials_dataframe()

        results['value'].sort_values().reset_index(drop=True).plot()
        plt.title('Convergence plot')
        plt.xlabel('Iteration')
        plt.ylabel('Error')

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        return img_buf

    def return_prediction_plot(self, plot_col='stock_logdif'):

        if self.wide_window.label_columns:
            label_col_index = self.wide_window.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        plot_col_index = self.wide_window.column_indices[plot_col]
        plt.figure(figsize=(8, 4))
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.wide_window.input_indices, self.data_[0, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

        predictions = self.model(self.data_)
        plt.scatter(self.wide_window.label_indices, predictions[0, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        plt.xlabel('Time [d]')
        plt.axhline(0.0, linestyle='--', color = 'pink')

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        return img_buf

    def future_prices_plot(self):
        plt.figure(figsize=(10, 4))
        plt.ylabel(f'stock price')
        plt.plot(self.some_history.Date, self.some_history.stock_price, label='Inputs', marker='.', zorder=-10)
        plt.scatter(self.prep_predictions.Date, self.prep_predictions.stock_price,marker='X', edgecolors='k', label='Predictions',c='#ff7f0e', s=64)
        plt.xlabel('date')

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        return img_buf