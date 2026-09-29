from pathlib import Path
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow Python to import src/predict_world_model.py
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict_world_model import (
    load_model,
    load_scaler,
    predict_next_state,
    forecast_multiple_steps,
)

app = Flask(__name__)
CORS(app)

print("Loading Network Attack Forecasting model...")

model = load_model()
scaler = load_scaler()

print("Model loaded successfully.")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "multitask_world_model_final",
        "history": 5,
        "features": 15,
        "classes": [
            "BENIGN",
            "DOS_DDOS",
            "RECONNAISSANCE",
        ],
    })


@app.route("/api/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json()

        if not data or "states" not in data:
            return jsonify({
                "error": "Request must contain 'states'."
            }), 400

        states = data["states"]

        # One-step prediction
        result = predict_next_state(
            model,
            scaler,
            states,
        )

        # Five-step autoregressive forecast
        forecast = forecast_multiple_steps(
            model,
            scaler,
            states,
            steps=5,
        )

        return jsonify({
            "success": True,

            "prediction": {
                "class": result["predicted_class"],
                "class_id": result["predicted_class_id"],
                "probabilities": result["probabilities"],
                "future_state": result["future_state"].tolist(),
            },

            "forecast": [
                {
                    "horizon": item["horizon"],
                    "predicted_class": item["predicted_class"],
                    "predicted_class_id": item["predicted_class_id"],
                    "probabilities": {
                        "BENIGN": item["BENIGN_probability"],
                        "DOS_DDOS": item["DOS_DDOS_probability"],
                        "RECONNAISSANCE": item["RECONNAISSANCE_probability"],
                    },
                    "future_state": item["future_state"].tolist(),
                }
                for item in forecast
            ],
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("NETWORK ATTACK FORECASTING API")
    print("=" * 60)
    print("API: http://127.0.0.1:5000")
    print("Health: http://127.0.0.1:5000/api/health")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )