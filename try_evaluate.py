from empyrical import sharpe_ratio
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from settings.default import QUANDL_TICKERS
from mom_trans.model_inputs import ModelFeatures



# python try_evaluate.py

import json

#with open('data.json', 'r') as file:
#    data = json.load(file)

def create_model():
    hidden_layer_size = 20
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

def get_positions(
        data,
        model,
        sliding_window=True,
        years_geq=np.iinfo(np.int32).min,
        years_lt=np.iinfo(np.int32).max,
    ):
        inputs, outputs, _, identifier, time = ModelFeatures._unpack(data)

        if sliding_window:
            time = pd.to_datetime(
                time[:, -1, 0].flatten()
            )  # TODO to_datetime maybe not needed
            years = time.map(lambda t: t.year)
            identifier = identifier[:, -1, 0].flatten()
            returns = outputs[:, -1, 0].flatten()
        else:
            time = pd.to_datetime(time.flatten())
            years = time.map(lambda t: t.year)
            identifier = identifier.flatten()
            returns = outputs.flatten()
        mask = (years >= years_geq) & (years < years_lt)

        positions = model.predict(inputs)

        if sliding_window:
            positions = positions[:, -1, 0].flatten()
        else:
            positions = positions.flatten()

        captured_returns = returns * positions
        results = pd.DataFrame(
            {
                "identifier": identifier[mask],
                "time": time[mask],
                "returns": returns[mask],
                "position": positions[mask],
                "captured_returns": captured_returns[mask],
            }
        )    

        # don't need to divide sum by n because not storing here
        # mean does not work as well (related to days where no information)
        performance = sharpe_ratio(results.groupby("time")["captured_returns"].sum())

        return results, performance        

def main():

    features_file_path = "data/quandl_cpd_nonelbw.csv"
    raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
    raw_data["date"] = raw_data["date"].astype("datetime64[ns]")

    params = {
        'architecture': 'LSTM', 
        'total_time_steps': 63, 
        'early_stopping_patience': 25, 
        'multiprocessing_workers': 32, 
        'num_epochs': 300, 
        'fill_blank_dates': False, 
        'split_tickers_individually': True, 
        'random_search_iterations': 50, 
        'evaluate_diversified_val_sharpe': True, 
        'train_valid_ratio': 0.9, 
        'time_features': False, 
        'force_output_sharpe_length': None
    }

    train_interval = (2010, 2022, 2025)
    changepoint_lbws = None # depends on selected architecture

    asset_class_dictionary = {
        "QQQ": 'COMB',
        "SMH": 'COMB',
        "SOXX": 'COMB',
        "SPY": 'COMB',
        "XBI": 'COMB',
        "XLC": 'COMB',
        "XLE": 'COMB',
        "XLF": 'COMB',
        "XLK": 'COMB',
        "XLRE": 'COMB',
        "XLU": 'COMB',
        "XLV": 'COMB',
        "XLY": 'COMB',
        "XLP": 'COMB',
        "XLI": 'COMB',
        "XLB": 'COMB'
    }

    model_features = ModelFeatures(
        raw_data,
        params["total_time_steps"],
        start_boundary=train_interval[0],
        test_boundary=train_interval[1],
        test_end=train_interval[2],
        changepoint_lbws=changepoint_lbws,
        split_tickers_individually=params["split_tickers_individually"],
        train_valid_ratio=params["train_valid_ratio"],
        add_ticker_as_static=(params["architecture"] == "TFT"),
        time_features=params["time_features"],
        lags=params["force_output_sharpe_length"],
        asset_class_dictionary=asset_class_dictionary,
    )

    best_model = create_model()
    best_model.load_weights("results/experiment_quandl_100assets_lstm_cpnone_len63_notime_div_v1/2022-2025/best/checkpoints/checkpoint.weights.h5")
    best_model.summary()


    print("Predicting on test set...")

    results_sw, performance_sw = get_positions(
        model_features.test_sliding,
        best_model,
        sliding_window=True
    )
    print(f"performance (sliding window) = {performance_sw}")

    print(results_sw)

if __name__ == "__main__":
    main()