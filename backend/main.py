import sys
import os
import random
import random
import tempfile
import librosa
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Allow importing predict.py from the ../models folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from predict import predict  # noqa: E402
# Allow importing extract_mfcc from the ../data folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from preprocess_mafaulda import extract_mfcc, SAMPLE_RATE, SEGMENT_LENGTH_SEC  # noqa: E402
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
            raise RuntimeError(
                f"Dataset not found at {FEATURES_PATH}. Run data/preprocess.py first."
            )
        try:
            _dataset = np.load(FEATURES_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    return _dataset


def pick_sample():
    """Pick one random real sample and derive everything (sensors + diagnosis) from it,
    so the numbers shown on the dashboard are consistent with each other."""
    dataset = get_dataset()
    idx = random.randint(0, len(dataset["labels"]) - 1)

    acoustic_sample = dataset["acoustic"][idx]
    vibration_sample = dataset["vibration"][idx]

    acoustic_level = round(float(np.mean(np.abs(acoustic_sample))), 2)
    vibration_level = round(float(np.mean(np.abs(vibration_sample))), 3)

    severity_val = int(dataset["severity"][idx])
    simulated_temperature = round(45 + severity_val * 8 + random.uniform(-2, 2), 1)

    diagnosis = predict(acoustic_sample, vibration_sample)

    true_fault = ["Normal", "Outer Race", "Ball Fault", "Cage Fault"][int(dataset["labels"][idx])]
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
    try:
        sensors, diagnosis = pick_sample()
        return {"sensors": sensors, "diagnosis": diagnosis}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/api/sensors")
def get_sensors():
    try:
        sensors, _ = pick_sample()
        return sensors
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/api/diagnosis")
def get_diagnosis():
    try:
        _, diagnosis = pick_sample()
        return diagnosis
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/api/trend")
def get_trend():
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
@app.post("/api/predict-audio")
async def predict_from_audio(file: UploadFile = File(...)):
    """
    Accepts a real user-uploaded audio file, extracts MFCC features from it,
    pairs it with a REAL vibration sample from the dataset (since the model
    needs both modalities), and returns a prediction. Response clearly marks
    which parts came from the user's upload vs. the dataset.
    """
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        signal, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        os.unlink(tmp_path)

        segment_len = int(SAMPLE_RATE * SEGMENT_LENGTH_SEC)
        if len(signal) < segment_len:
            signal = np.pad(signal, (0, segment_len - len(signal)))
        else:
            signal = signal[:segment_len]

        acoustic_features = extract_mfcc(signal, SAMPLE_RATE)
        acoustic_features = (acoustic_features - acoustic_features.mean()) / (acoustic_features.std() + 1e-8)

        dataset = get_dataset()
        vib_idx = random.randint(0, len(dataset["vibration"]) - 1)
        vibration_features = dataset["vibration"][vib_idx]

        result = predict(acoustic_features, vibration_features)

        result["acoustic_source"] = f"user_uploaded_file ({file.filename})"
        result["vibration_source"] = f"dataset_sample (index {vib_idx}) — no vibration data was uploaded"
        result["transparency_note"] = (
            "This model requires both acoustic and vibration input. The acoustic "
            "features came from your uploaded audio; the vibration features came "
            "from a real recording in the training dataset, since no vibration "
            "sensor data was provided."
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio file: {e}")