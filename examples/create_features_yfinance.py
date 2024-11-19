import argparse
from typing import List
import pandas as pd
from data.symbols import SYMBOLS
from data.pull_data import pull_quandl_sample_data
from settings.default import (
    CPD_QUANDL_OUTPUT_FOLDER,
    FEATURES_QUANDL_FILE_PATH,
)
from mom_trans.data_prep import (
    deep_momentum_strategy_features,
    include_changepoint_features,
)

# python -m examples.create_features_yfinance

def main(
    tickers: List[str],
    cpd_module_folder: str,
    lookback_window_length: int,
    output_file_path: str,
    extra_lbw: List[int],
):
    
    features = pd.concat(
        [
            deep_momentum_strategy_features(pull_quandl_sample_data(ticker)).assign(
                ticker=ticker
            )
            for ticker in tickers
        ]
    )

    features.date = features.index
    features.index.name = "Date"

    if lookback_window_length:
        features_w_cpd = include_changepoint_features(
            features, cpd_module_folder, lookback_window_length
        )

        if extra_lbw:
            for extra in extra_lbw:
                extra_data = pd.read_csv(
                    output_file_path.replace(
                        f"quandl_cpd_{lookback_window_length}lbw.csv",
                        f"quandl_cpd_{extra}lbw.csv",
                    ),
                    index_col=0,
                    parse_dates=True,
                ).reset_index()[
                    ["date", "ticker", f"cp_rl_{extra}", f"cp_score_{extra}"]
                ]
                extra_data["date"] = pd.to_datetime(extra_data["date"])

                features_w_cpd = pd.merge(
                    features_w_cpd.set_index(["date", "ticker"]),
                    extra_data.set_index(["date", "ticker"]),
                    left_index=True,
                    right_index=True,
                ).reset_index()
                features_w_cpd.index = features_w_cpd["date"]
                features_w_cpd.index.name = "Date"
        else:
            features_w_cpd.index.name = "Date"
        features_w_cpd.to_csv(output_file_path)
    else:
        features.to_csv(output_file_path)

    print(f"Save feature file: {output_file_path}")

if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Create Momentum Transformer input features")
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=None,
            # choices=[],
            help="Set lookback window length",
        )
        parser.add_argument(
            "extra_lbw",
            metavar="-e",
            type=int,
            nargs="*",
            default=[],
            # choices=[],
            help="Fill missing prices.",
        )
        args = parser.parse_known_args()[0]

        return (
            SYMBOLS,
            CPD_QUANDL_OUTPUT_FOLDER(args.lookback_window_length),
            args.lookback_window_length,
            FEATURES_QUANDL_FILE_PATH(args.lookback_window_length),
            args.extra_lbw,
        )

    main(*get_args())