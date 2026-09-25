import sys
import os
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Allow importing predict.py from the ../models folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from predict import predict  # noqa: E402

app = FastAPI(title="Wind Turbine Bearing Fault Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.npz")
_dataset = None


def get_dataset():
    global _dataset
    if _dataset is None:
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"{FEATURES_PATH} not found. Run data/preprocess.py first.")
        _dataset = np.load(FEATURES_PATH)
    return _dataset


def pick_sample():
    """Pick one random real sample and derive everything (sensors + diagnosis) from it,
    so the numbers shown on the dashboard are consistent with each other."""
    dataset = get_dataset()
    idx = random.randint(0, len(dataset["labels"]) - 1)

    acoustic_sample = dataset["acoustic"][idx]
    vibration_sample = dataset["vibration"][idx]

    # Turn the raw MFCC/STFT feature arrays into single representative
    # numbers for display — this is real signal energy from the actual
    # recording, not a fixed constant.
    acoustic_level = round(float(np.mean(np.abs(acoustic_sample))), 2)
    vibration_level = round(float(np.mean(np.abs(vibration_sample))), 3)

    # Temperature isn't in this dataset (KAIST has no temperature channel),
    # so we simulate a plausible reading tied to severity for demo purposes.
    # This is clearly a placeholder — real temperature needs a thermal sensor.
    severity_val = int(dataset["severity"][idx])
    simulated_temperature = round(45 + severity_val * 8 + random.uniform(-2, 2), 1)

    diagnosis = predict(acoustic_sample, vibration_sample)

    true_fault = ["Normal", "Inner Race", "Outer Race"][int(dataset["labels"][idx])]
    diagnosis["true_fault_type_for_demo"] = true_fault

    sensors = {
        "acoustic": acoustic_level,
        "vibration": vibration_level,
        "temperature": simulated_temperature,
        "current": "coming soon",
    }

    return sensors, diagnosis


@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/api/dashboard")
def get_dashboard():
    """Single endpoint returning sensors + diagnosis from the SAME sample."""
    sensors, diagnosis = pick_sample()
    return {"sensors": sensors, "diagnosis": diagnosis}


@app.get("/api/sensors")
def get_sensors():
    sensors, _ = pick_sample()
    return sensors


@app.get("/api/diagnosis")
def get_diagnosis():
    _, diagnosis = pick_sample()
    return diagnosis


@app.get("/api/trend")
def get_trend():
    # Still illustrative — a real trend needs a run-to-failure time series,
    # which the current dataset doesn't provide (noted as future work).
    trend = []
    score = 100
    for day in range(1, 31):
        score -= random.uniform(1.5, 2.5)
        trend.append({"day": day, "score": round(max(score, 0), 1)})
    return trend
@app.get("/api/health")
def health_check():
    """Quick check: is the backend up, and is the model file present?"""
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "model.pth")
    dataset_ready = os.path.exists(FEATURES_PATH)
    model_ready = os.path.exists(model_path)

    return {
        "status": "ok",
        "dataset_loaded": dataset_ready,
        "model_loaded": model_ready,
    }