import numpy as np
import pandas as pd
from transformer_sac.data_loader import compute_market_features, load_market_data

SEQ_LEN = 30

class FeatureEngine:

    def __init__(self, scaler):
        self.scaler = scaler
        self.market_data = load_market_data()

    # ==========================
    # Build sequence state
    # ==========================
    def build_sequence(self, asset_df, position):

        df = asset_df.copy()

        # ===== Asset features =====
        df['log_ret'] = np.log(df['close']).diff()
        df['ret_5'] = df['log_ret'].rolling(5).sum()
        df['ret_20'] = df['log_ret'].rolling(20).sum()

        df['volatility'] = df['log_ret'].rolling(20).std()
        df['ma10'] = df['close'].rolling(10).mean() / df['close'] - 1.0
        df['ma50'] = df['close'].rolling(50).mean() / df['close'] - 1.0

        df['range'] = (df['high'] - df['low']) / df['close']
        df['log_vol'] = np.log(df['volume'] + 1).diff()

        # ===== Market regime =====
        sp = compute_market_features(self.market_data["GSPC"])
        nas = compute_market_features(self.market_data["IXIC"])
        vix = compute_market_features(self.market_data["VIX"])

        sp.columns = ["sp_ret1", "sp_ret5"]
        nas.columns = ["nas_ret1", "nas_ret5"]
        vix.columns = ["vix_ret1", "vix_ret5"]

        df = df.merge(sp, left_index=True, right_index=True, how="inner")
        df = df.merge(nas, left_index=True, right_index=True, how="inner")
        df = df.merge(vix, left_index=True, right_index=True, how="inner")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) < SEQ_LEN:
            return None

        # ===== EXACT feature order like data_loader =====
        features = df[[
            'log_ret','ret_5','ret_20',
            'volatility','ma10','ma50',
            'range','log_vol',
            'sp_ret1','sp_ret5',
            'nas_ret1','nas_ret5',
            'vix_ret1','vix_ret5'
        ]].values.astype(np.float32)

        # Lấy 30 bước cuối
        features = features[-SEQ_LEN:]

        # Scale features
        features = self.scaler.transform(features)

        # ===== Portfolio state =====
        portfolio = np.tile(
            np.array([1 - position, position], dtype=np.float32),
            (SEQ_LEN, 1)
        )

        state_seq = np.concatenate([features, portfolio], axis=1)

        return state_seq