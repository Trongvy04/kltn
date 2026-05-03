# data_loader.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from .config import SCALER_PATH

DATA_DIR = "data"
MARKET_FILES = ["GSPC.csv", "IXIC.csv", "VIX.csv"]

def load_market_data():

    market_data = {}
    for file in MARKET_FILES:
        path = os.path.join(DATA_DIR, "market", file)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        market_data[file.split(".")[0]] = df
    return market_data

def compute_market_features(df):
    df = df.copy()
    df['log_ret'] = np.log(df['close']).diff()
    df['ret_5'] = df['log_ret'].rolling(5).sum()
    return df[['log_ret', 'ret_5']]

def build_features(asset_df, market_data):

    df = asset_df.copy()

    df['log_ret'] = np.log(df['close']).diff()
    df['ret_5'] = df['log_ret'].rolling(5).sum()
    df['ret_20'] = df['log_ret'].rolling(20).sum()
    df['volatility'] = df['log_ret'].rolling(20).std()
    df['ma10'] = df['close'].rolling(10).mean() / df['close'] - 1.0
    df['ma50'] = df['close'].rolling(50).mean() / df['close'] - 1.0
    df['range'] = (df['high'] - df['low']) / df['close']
    df['log_vol'] = np.log(df['volume'] + 1).diff()

    sp = compute_market_features(market_data["GSPC"])
    nas = compute_market_features(market_data["IXIC"])
    vix = compute_market_features(market_data["VIX"])

    sp.columns = ["sp_ret1", "sp_ret5"]
    nas.columns = ["nas_ret1", "nas_ret5"]
    vix.columns = ["vix_ret1", "vix_ret5"]

    df = df.merge(sp, left_index=True, right_index=True, how="inner")
    df = df.merge(nas, left_index=True, right_index=True, how="inner")
    df = df.merge(vix, left_index=True, right_index=True, how="inner")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = [
        'log_ret','ret_5','ret_20',
        'volatility','ma10','ma50',
        'range','log_vol',
        'sp_ret1','sp_ret5',
        'nas_ret1','nas_ret5',
        'vix_ret1','vix_ret5'
    ]

    return (
        df[features].values.astype(np.float32),
        df['close'].values.astype(np.float32)
    )

def load_stocks(mode="train"):
    folder = os.path.join(DATA_DIR, mode)
    market_data = load_market_data()
    states_list = []
    prices_list = []
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(
            os.path.join(folder, file),
            index_col=0,
            parse_dates=True
        )
        states, prices = build_features(df, market_data)
        states_list.append(states)
        prices_list.append(prices)
    if mode == "train":
        scaler = StandardScaler()
        scaler.fit(np.concatenate(states_list, axis=0))
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
    states_list = [scaler.transform(s) for s in states_list]
    return states_list, prices_list