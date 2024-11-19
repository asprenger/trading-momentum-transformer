import os
import json
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from mom_trans.model_inputs import ModelFeatures
from data.symbols import ASSET_CLASS_MAPPING, SYMBOLS
from mom_trans.classical_strategies import (
    calc_net_returns
)
from settings.default import BACKTEST_AVERAGE_BASIS_POINTS

from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
)


# python try_evaluate.py

def create_model(hidden_layer_size:int, dropout_rate:int, time_steps:int, input_size:int):
    time_steps = 63
    input_size = 8
    output_size = 1

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
    input_data_path = "data/quandl_cpd_nonelbw.csv"
    model_path = "results/experiment_quandl_100assets_lstm_cpnone_len63_notime_div_v1/2022-2025/best"

    # load hyper parameter
    hp_path = os.path.join(model_path, "hyperparameters.json")
    with open(hp_path, 'r') as file:
        hyper_params = json.load(file)

    # load input data
    raw_data = pd.read_csv(input_data_path, index_col=0, parse_dates=True)
    raw_data["date"] = raw_data["date"].astype("datetime64[ns]")

    train_interval = (2010, 2022, 2025)
    time_steps = 63

    # Note that many of the ModelFeature parameters
    # depend on the model architecture
    model_features = ModelFeatures(
        raw_data,
        time_steps,
        start_boundary=train_interval[0],
        test_boundary=train_interval[1],
        test_end=train_interval[2],
        changepoint_lbws=None,
        split_tickers_individually=True,
        train_valid_ratio=0.9,
        add_ticker_as_static=False,
        time_features=False,
        lags=None,
        asset_class_dictionary=ASSET_CLASS_MAPPING,
    )

    checkpoint_path = os.path.join(model_path, "checkpoints/checkpoint.weights.h5")
    input_size = model_features.test_fixed['inputs'].shape[2]
    best_model = create_model(hidden_layer_size=hyper_params['hidden_layer_size'], 
                              dropout_rate=hyper_params['dropout_rate'],
                              time_steps=time_steps,
                              input_size=input_size)
    
    best_model.load_weights(checkpoint_path)
    best_model.summary()

    if False:
        date = model_features.test_sliding['date'][0][0][0]
        ticker = model_features.test_sliding['identifier'][0][0][0]
        print(model_features.test_sliding['outputs'][0][0][0])

        foo = raw_data[raw_data['date']==date]
        bar = foo[foo['ticker']==ticker]
        print(bar['target_returns'])

    # Note we are testing on `model_features.test_sliding`. This has as output the target_rate 
    # (as calculated by ModelFeatures) at the given date as output and a window of features up
    # to this date as input.

    results_sw, performance_sw = get_positions(model_features.test_sliding,
                                               best_model,
                                               sliding_window=True)
    print(f"performance (sliding window) = {performance_sw}")
    

    for ticker in SYMBOLS:
        captured_returns = results_sw[results_sw['identifier']==ticker]['captured_returns']
        print(ticker)
        #print(f"Sharp Ratio: {sharpe_ratio(captured_returns)}")
        #print(f"Annual Return: {annual_return(captured_returns)}")
        #print(f"Max Drawdown: {max_drawdown(captured_returns)}")


        perc_pos_return = len(captured_returns[captured_returns > 0.0]) / len(captured_returns)
        profit_loss_ratio = np.mean(captured_returns[captured_returns >= 0.0]) / np.mean(np.abs(captured_returns[captured_returns < 0.0]))

        #print(f"annual_volatility: {annual_volatility(captured_returns)}")
        print(f"perc_pos_return: {perc_pos_return}")
        #print(f"profit_loss_ratio: {profit_loss_ratio}")


        print()

    

    


if __name__ == "__main__":
    main()