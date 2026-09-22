"""
predict.py — P2 (Model Lead)

Loads the trained model and exposes a predict() function in EXACTLY the
format CONTRACT.md specifies, so P3's backend can call it directly.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from model import CAVFNet

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "model.pth")

FAULT_NAMES = ["Normal", "Inner Race", "Outer Race"]
SEVERITY_NAMES = ["Minor", "Moderate", "Severe"]

# Rule-based RUL mapping — a real RUL model needs run-to-failure labeled
# data we don't have yet. This is explicitly called out as future work.
RUL_MAP = {
    "Minor": 75,
    "Moderate": 30,
    "Severe": 10,
}

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _model
    if _model is None:
        _model = CAVFNet()
        # LazyLinear layers need one dummy forward pass before weights can be loaded
        dummy_acoustic = torch.zeros(1, 40, 128)
        dummy_vibration = torch.zeros(1, 64, 128)
        _model(dummy_acoustic, dummy_vibration)

        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        _model.to(_device)
        _model.eval()
    return _model


def predict(acoustic, vibration):
    """
    acoustic: numpy array of shape (40, 128)
    vibration: numpy array of shape (64, 128)

    Returns a dictionary with exactly these four keys, per CONTRACT.md:
        fault_type: str  — "Normal" | "Inner Race" | "Outer Race"
        severity:   str  — "Normal" | "Minor" | "Moderate" | "Severe"
        confidence: float in [0, 1]
        rul_days:   int
    """
    model = _load_model()

    acoustic_t = torch.tensor(acoustic, dtype=torch.float32).unsqueeze(0).to(_device)   # (1, 40, 128)
    vibration_t = torch.tensor(vibration, dtype=torch.float32).unsqueeze(0).to(_device)  # (1, 64, 128)

    with torch.no_grad():
        fault_logits, severity_logits = model(acoustic_t, vibration_t)
        fault_probs = F.softmax(fault_logits, dim=1)[0]
        severity_probs = F.softmax(severity_logits, dim=1)[0]

    fault_idx = int(torch.argmax(fault_probs))
    severity_idx = int(torch.argmax(severity_probs))

    fault_type = FAULT_NAMES[fault_idx]
    severity = "Normal" if fault_type == "Normal" else SEVERITY_NAMES[severity_idx]
    confidence = float(fault_probs[fault_idx])
    rul_days = 365 if fault_type == "Normal" else RUL_MAP[severity]

    return {
        "fault_type": fault_type,
        "severity": severity,
        "confidence": round(confidence, 4),
        "rul_days": rul_days,
    }


if __name__ == "__main__":
    # Quick manual test using a random sample from the processed dataset
    data = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.npz"))
    idx = 0
    result = predict(data["acoustic"][idx], data["vibration"][idx])
    print("Sample prediction:", result)
    print("True label:", data["labels"][idx], "True severity:", data["severity"][idx])