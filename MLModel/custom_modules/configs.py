import os 

root = os.getcwd()

class data_configs:
    n_days = 950
    lags = 4
    window = 7
    std_column = 'Close_roll_std'
    ref_price = 'Close'
    logdif_column = 'Dif'
    split_config = {
        'train':0.65,
        'val':0.85
    }
    steps_to_predic = 4
    input_length = 14
    best_error = 1.1
    save_predictions_path = f'{root}/predictions_csvs'
    save_model_path = f'{root}/models_hpt'
    optuna_trials = 60

class optimazation_configs:
    days_back_predictions = 7
    cross_validation_trials = 50 ### normal 50