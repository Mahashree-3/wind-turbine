import sys
import os
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

import mock_data

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

# Load the processed dataset once at startup so we have real samples to run
# through the model. In a real deployment this would come from a live sensor
# feed instead — for the prototype, we pick from real recorded samples.
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.npz")
_dataset = None


def get_dataset():
    global _dataset
    if _dataset is None:
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(
                f"{FEATURES_PATH} not found. Run data/preprocess.py first."
            )
        _dataset = np.load(FEATURES_PATH)
    return _dataset


@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/api/sensors")
def get_sensors():
    # Sensor readings are still illustrative display values —
    # only /api/diagnosis uses the real trained model.
    return mock_data.SENSORS


@app.get("/api/diagnosis")
def get_diagnosis():
    """
    Picks a random real sample from the processed dataset and runs it
    through the trained model, returning a REAL prediction instead of
    hardcoded mock data.
    """
    dataset = get_dataset()
    idx = random.randint(0, len(dataset["labels"]) - 1)

    acoustic_sample = dataset["acoustic"][idx]
    vibration_sample = dataset["vibration"][idx]

    result = predict(acoustic_sample, vibration_sample)

    # Also include the true label for demo/debugging purposes (optional,
    # remove if you don't want to show this on the dashboard)
    true_fault = ["Normal", "Inner Race", "Outer Race"][int(dataset["labels"][idx])]
    result["true_fault_type_for_demo"] = true_fault

    return result


@app.get("/api/trend")
def get_trend():
    return mock_data.TREND