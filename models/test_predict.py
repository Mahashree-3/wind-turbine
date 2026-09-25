"""
test_predict.py — Run a prediction on ONE specific real file you choose.

Usage:
    python models/test_predict.py 0Nm_BPFI_03.mat
    python models/test_predict.py 4Nm_BPFO_10.mat
    python models/test_predict.py 0Nm_Normal.mat

This takes the filename (must exist in BOTH data/raw/acoustic/ and
data/raw/vibration/), runs it through the SAME feature extraction as
preprocess.py (MFCC + STFT), then runs it through the trained model.

This proves the model works on a specific real file you pick, not just
whatever preprocess.py already segmented into features.npz.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from predict import predict  # noqa: E402
from preprocess import (  # noqa: E402
    load_mat_signal,
    extract_mfcc,
    extract_stft,
    SAMPLE_RATE,
)

ACOUSTIC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "acoustic")
VIBRATION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "vibration")


def run_on_file(filename):
    acoustic_path = os.path.join(ACOUSTIC_DIR, filename)
    vibration_path = os.path.join(VIBRATION_DIR, filename)

    if not os.path.exists(acoustic_path):
        print(f"File not found: {acoustic_path}")
        return
    if not os.path.exists(vibration_path):
        print(f"File not found: {vibration_path}")
        return

    print(f"Loading {filename} ...")
    acoustic_signal = load_mat_signal(acoustic_path)
    vibration_signal = load_mat_signal(vibration_path)

    # Take the first 1-second segment for a quick single prediction
    segment_len = int(SAMPLE_RATE * 1.0)
    acoustic_segment = acoustic_signal[:segment_len]
    vibration_segment = vibration_signal[:segment_len]

    acoustic_features = extract_mfcc(acoustic_segment, SAMPLE_RATE)
    vibration_features = extract_stft(vibration_segment, SAMPLE_RATE)

    # Normalize the same way train.py does
    acoustic_features = (acoustic_features - acoustic_features.mean()) / (acoustic_features.std() + 1e-8)
    vibration_features = (vibration_features - vibration_features.mean()) / (vibration_features.std() + 1e-8)

    result = predict(acoustic_features, vibration_features)

    print(f"\nFile: {filename}")
    print(f"Prediction: {result}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python models/test_predict.py <filename.mat>")
        print("Example: python models/test_predict.py 0Nm_BPFI_03.mat")
        print("\nAvailable files in data/raw/acoustic/:")
        if os.path.exists(ACOUSTIC_DIR):
            for f in os.listdir(ACOUSTIC_DIR):
                print(f"  {f}")
        sys.exit(1)

    run_on_file(sys.argv[1])