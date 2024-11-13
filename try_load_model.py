from tensorflow import keras
import tensorflow as tf

# python try_load_model.py

def create_model():
    hidden_layer_size = 80
    dropout_rate = 0.2
    max_gradient_norm = 1.0

    time_steps = 63
    input_size = 8
    output_size = 1 # ????

    input = keras.Input((time_steps, input_size))
    lstm = tf.keras.layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=dropout_rate,
        stateful=False,
        activation="tanh",
        recurrent_activation="sigmoid",
        recurrent_dropout=0,
        unroll=False,
        use_bias=True,
    )(input)
    dropout = keras.layers.Dropout(dropout_rate)(lstm)

    output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            output_size,
            activation=tf.nn.tanh,
            kernel_constraint=keras.constraints.max_norm(3),
        )
    )(dropout[..., :, :]) # WTF is this????

    model = keras.Model(inputs=input, outputs=output)

    return model

def load_model_from_tf_checkpoint(checkpoint_dir):

    # Create a new model
    model = create_model()
    
    # Create checkpoint instance
    checkpoint = tf.train.Checkpoint(model=model)
    
    # Restore the checkpoint
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.expect_partial()  # Suppress warnings about incomplete restoration
    
    # Convert to Keras format and save
    keras_path = "converted_model.keras"
    model.save(keras_path, save_format='keras')
    
    # Load the converted Keras model
    converted_model = keras.models.load_model(keras_path)
    
    return converted_model

def main():

    checkpoint_path = "results/experiment_quandl_100assets_lstm_cpnone_len63_notime_div_v1/2022-2025/best/checkpoints"
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    print(latest_checkpoint)

    model = load_model_from_tf_checkpoint(latest_checkpoint)
    model.summary()

if __name__ == "__main__":
    main()