import mlflow.pyfunc
from mlflow import MlflowClient
from .custom_modules import  datafuncion,configs
import datetime

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


class get_model_production_results():

    
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.predictions_path_file = f'{save_predictions_path}/{today_str}-{self.stock_code}-predictions.csv'

    def get_model_predictions(self,plot_col='stock_logdif'):
        
        try:

            client = MlflowClient()

            model_name = f'{self.stock_code}_models'
            latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
            latest_production_version = latest_version_info[0].version
            model_version = latest_production_version
            self.model_version = model_version

            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/{model_version}"
            )

            self.model = model 

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

            predictions, data_ = wide_window.expected_return(
                plot_col='stock_logdif',
                model = self.model,
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

            final_result.to_csv(self.predictions_path_file)
            self.wide_window = wide_window

            self.some_history = some_history
            self.prep_predictions = prep_predictions

            message = f'the stock: {self.stock_code} the model version is {self.model_version} predictions are done and plots are: '
            
        except:
            message = 'there is not a registered model or sth went wrong'

        return message
    
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

        predictions = self.model.predict(self.data_)
        plt.scatter(self.wide_window.label_indices, predictions[0, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        plt.xlabel('Time [d]')
        plt.axhline(0.0, linestyle='--', color = 'pink')
        plt.title(f'{self.stock_code}: return prediction plot')

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
        plt.title(f'{self.stock_code}: price prediction plot')

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        return img_buf
