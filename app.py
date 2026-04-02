from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import joblib

from model_service import ModelService
from feature_engine import FeatureEngine
from env_engine import TradingEnvEngine
from transformer_sac.config import MODEL_PATH, SCALER_PATH, DEVICE

app = Flask(__name__)

model_service = ModelService(str(MODEL_PATH), DEVICE)
scaler = joblib.load(SCALER_PATH)

feature_engine = FeatureEngine(scaler)
env = TradingEnvEngine()

data_buffer = []


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/reset", methods=["POST"])
def reset():
    global data_buffer
    global env
    data_buffer = []
    env.reset()
    return {"status": "reset_done"}


@app.route("/step", methods=["POST"])
def step():

    global data_buffer

    new_bar = request.json

    required_cols = ["date", "open", "high", "low", "close", "volume"]

    # Validate input
    if not all(k in new_bar for k in required_cols):
        return jsonify({"ready": False})

    try:
        clean_bar = {
            "date": pd.to_datetime(new_bar["date"]),
            "open": float(new_bar["open"]),
            "high": float(new_bar["high"]),
            "low": float(new_bar["low"]),
            "close": float(new_bar["close"]),
            "volume": float(new_bar["volume"])
        }
    except:
        return jsonify({"ready": False})

    data_buffer.append(clean_bar)

    # ===== BUILD DF SAFELY =====
    df = pd.DataFrame.from_records(data_buffer)

    # ép đúng thứ tự cột
    df = df[["date", "open", "high", "low", "close", "volume"]]

    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # ===== BUILD STATE =====
    state_seq = feature_engine.build_sequence(
        df,
        env.shares > 0.0
    )

    if state_seq is None:
        return jsonify({
            "ready": False,
            "cash": float(env.cash),
            "shares": float(env.shares)
        })

    action = model_service.predict(state_seq)
    result = env.step(action, clean_bar["close"])

    return jsonify({
        "ready": True,
        "date": clean_bar["date"].strftime("%Y-%m-%d"),
        "action": int(action),
        "action_name": result["action_name"],
        "shares": float(result["shares"]),
        "cash": float(result["cash"]),
        "portfolio_value": float(result["portfolio_value"]),
        "close_price": float(clean_bar["close"])
    })


if __name__ == "__main__":
    app.run(debug=True)