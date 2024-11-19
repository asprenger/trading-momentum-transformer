import os
import pandas as pd

def pull_quandl_sample_data(ticker: str) -> pd.DataFrame:
    parquet_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'sectors/{ticker}_daily.parquet')
    return pd.read_parquet(parquet_path)

