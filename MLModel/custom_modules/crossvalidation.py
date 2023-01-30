import tensorflow as tf
import optuna 
import numpy as np

def save_model_fun(save_model, model , error, best_error, best_model_path):
    if round(error,0) < best_error:
        if save_model:
            model.save(best_model_path)
        best_error = error
    
    return best_error


def objective_conv(
    trial,
    OUT_STEPS, 
    num_features, 
    wide_window, 
    best_error,
    save_model = False, 
    best_model_path = None
    ):
    
    patience=2
    MAX_EPOCHS = 20
    
    model = tf.keras.Sequential()
    
    CONV_WIDTH = trial.suggest_categorical('conv_width', [3,4,5,6,7])
    
    model.add(tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]))
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 1) 
    for i in range(num_conv_layers):
        
        model.add(
            tf.keras.layers.Conv1D(
                filters = trial.suggest_categorical(f'conv_1d_filters_{i}', [150,230,250,270,300]),
                activation = trial.suggest_categorical(f'conv_1d_activation_{i}', ['relu', 'tanh', 'sigmoid']),
                kernel_size=(CONV_WIDTH)
            )
        )
        model.add(
            tf.keras.layers.Dropout(
            rate= trial.suggest_categorical(f'dropout_{i}', [0.0,0.2,0.4,0.6])
            )
        )
          
    num_dense_layers = trial.suggest_int('num_dense_layers', 0, 3) 
    for i in range(num_dense_layers):
        
        model.add(
            tf.keras.layers.Dense(
                units = trial.suggest_categorical(f'num_units_layers_{i}', [30,70,100,130,150,170,200]),
                kernel_initializer=tf.initializers.zeros(),
                activation = trial.suggest_categorical(f'dense_activation_{i}', ['relu', 'tanh','sigmoid'])
            )
        )
        model.add(
            tf.keras.layers.Dropout(
            rate= trial.suggest_categorical(f'dropout_{i}', [0.0,0.2,0.4,0.6])
            )
        )
        
    model.add(tf.keras.layers.Dense(OUT_STEPS*num_features,kernel_initializer=tf.initializers.zeros()))
    model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features]))   
    
    optimizer_name =trial.suggest_categorical('optimizer_name', ['Adam', 'RMSprop'])
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate',  1e-6, 1e-2))
    else:
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=trial.suggest_float('learning_rate',  1e-6, 1e-2),
            momentum=trial.suggest_float('momentum',  0.1, 0.9),
        )
    
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer= optimizer,
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience, mode='min')
    
    history = model.fit(
        wide_window.train,
        epochs=MAX_EPOCHS,
        validation_data=wide_window.val,
        callbacks=[early_stopping]
    )
    
    error = model.evaluate(wide_window.test)[1]  
    
    best_error = save_model_fun(save_model, model, error, best_error, best_model_path)

    del model
    return error


def objective_linear(
    trial,
    OUT_STEPS, 
    num_features, 
    wide_window, 
    best_error,
    save_model = False, 
    best_model_path = None):
    
    patience=2
    MAX_EPOCHS = 20
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x[:, -1:, :]))
    
    num_dense_layers = trial.suggest_int('num_dense_layers', 0, 5) 
    for i in range(num_dense_layers):
        
        model.add(
            tf.keras.layers.Dense(
                units = trial.suggest_categorical(f'num_units_layers_{i}', [30,70,100,130,150,170,200]),
                kernel_initializer=tf.initializers.zeros(),
                activation = trial.suggest_categorical(f'dense_activation_{i}', ['relu', 'tanh','sigmoid'])
            )
        )
        model.add(
            tf.keras.layers.Dropout(
            rate= trial.suggest_categorical(f'dropout_{i}', [0.0,0.2,0.4,0.6])
            )
        )
        
        
    model.add(tf.keras.layers.Dense(OUT_STEPS*num_features,kernel_initializer=tf.initializers.zeros()))
    model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features]))
    
    optimizer_name =trial.suggest_categorical('optimizer_name', ['Adam', 'RMSprop'])
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate',  1e-6, 1e-2))
    else:
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=trial.suggest_float('learning_rate',  1e-6, 1e-2),
            momentum=trial.suggest_float('momentum',  0.1, 0.9),
        )
    
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer= optimizer,
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience, mode='min')
    
    history = model.fit(
        wide_window.train,
        epochs=MAX_EPOCHS,
        validation_data=wide_window.val,
        callbacks=[early_stopping]
    )
    
    error = model.evaluate(wide_window.test)[1]  
    
    best_error = save_model_fun(save_model, model, error, best_error, best_model_path)

    del model
    return error

class objective_functions():
    ojective_functions =  {
        'Linear': objective_linear,
        'Dense': objective_linear,
        'Conv': objective_conv,
        'LSTM': objective_conv,
        }

