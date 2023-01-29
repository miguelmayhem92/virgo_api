import tensorflow as tf

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


class initial_models():
    def __init__(self, OUT_STEPS, num_features, CONV_WIDTH = 3):

        self.multi_linear_model = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        self.multi_dense_model =tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        self.multi_conv_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        self.multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        ### autoregresive model
        class FeedBack(tf.keras.Model):
            def __init__(self, units, out_steps):
                super().__init__()
                self.out_steps = out_steps
                self.units = units
                self.lstm_cell = tf.keras.layers.LSTMCell(units)
                # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
                self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
                self.dense = tf.keras.layers.Dense(num_features)

            def warmup(self, inputs):
                # inputs.shape => (batch, time, features)
                # x.shape => (batch, lstm_units)
                x, *state = self.lstm_rnn(inputs)

                # predictions.shape => (batch, features)
                prediction = self.dense(x)
                return prediction, state

            def call(self, inputs, training=None):
                # Use a TensorArray to capture dynamically unrolled outputs.
                predictions = []
                # Initialize the LSTM state.
                prediction, state = self.warmup(inputs)

                # Insert the first prediction.
                predictions.append(prediction)

                # Run the rest of the prediction steps.
                for n in range(1, self.out_steps):
                    # Use the last prediction as input.
                    x = prediction
                    # Execute one lstm step.
                    x, state = self.lstm_cell(x, states=state,
                                            training=training)
                    # Convert the lstm output to a prediction.
                    prediction = self.dense(x)
                    # Add the prediction to the output.
                    predictions.append(prediction)

                # predictions.shape => (time, batch, features)
                predictions = tf.stack(predictions)
                # predictions.shape => (batch, time, features)
                predictions = tf.transpose(predictions, [1, 0, 2])
                
                return predictions

        self.feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)