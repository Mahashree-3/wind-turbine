"""
preprocess_mafaulda.py — New dataset preprocessing (MaFaulDa)

Folder structure expected (as confirmed from your download):
    data/raw_mafaulda/normal/normal/*.csv
    data/raw_mafaulda/underhang/underhang/{outer_race,cage_fault,ball_fault}/{0g,6g,20g,35g}/*.csv
    data/raw_mafaulda/overhang/overhang/{outer_race,cage_fault,ball_fault}/{0g,6g,20g,35g}/*.csv

CSV format (no header, 8 columns, 50kHz, 5 seconds = 250,000 rows):
    col 1: tachometer
    col 2-4: underhang bearing accelerometer (axial, radial, tangential)
    col 5-7: overhang bearing accelerometer (axial, radial, tangential)
    col 8: microphone (acoustic)

IMPORTANT LABEL CHANGE FROM THE KAIST VERSION:
MaFaulDa does not have an "Inner Race" fault category. Its fault types are
Normal, Outer Race, Ball Fault, and Cage Fault. This is now a 4-CLASS problem
instead of 3. predict.py and the backend have been updated to match.

Output: data/processed/features.npz containing:
    acoustic  -> shape (N, 40, 128)
    vibration -> shape (N, 64, 128)
    labels    -> shape (N,)  0=Normal, 1=Outer Race, 2=Ball Fault, 3=Cage Fault
    severity  -> shape (N,)  0=Minor, 1=Moderate, 2=Severe
    file_id   -> shape (N,)  groups samples from the same source file/segment

INSTALL REQUIRED PACKAGES:
    pip install numpy scipy librosa pandas
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from scipy.signal import stft

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RAW_DIR = "data/raw_mafaulda"
OUT_DIR = "data/processed"
OUT_FILE = os.path.join(OUT_DIR, "features.npz")

SAMPLE_RATE = 50000            # MaFaulDa sampling rate (Hz)
SEGMENT_LENGTH_SEC = 1.0
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 200
TARGET_TIME_FRAMES = 128

# How many CSV files to sample per (fault, load) folder — each folder has ~49
# files (one per rotation frequency), we don't need all of them and it would
# take very long to process. This still gives plenty of independent files.
N_FILES_PER_FOLDER = 6
N_NORMAL_FILES = 10

FAULT_NAMES = ["Normal", "Outer Race", "Ball Fault", "Cage Fault"]
FAULT_FOLDER_MAP = {"outer_race": 1, "ball_fault": 2, "cage_fault": 3}

LOAD_SEVERITY_MAP = {"0g": 0, "6g": 0, "20g": 1, "35g": 2}  # Minor, Minor, Moderate, Severe

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# STEP 1 — Discover files
# ---------------------------------------------------------------------------

def find_files():
    """Returns a list of (csv_path, label, severity, bearing_position) tuples."""
    rng = random.Random(RANDOM_SEED)
    entries = []

    # Normal files
    normal_dir = os.path.join(RAW_DIR, "normal", "normal", "normal")
    normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.csv")))
    if not normal_files:
        print(f"WARNING: no files found in {normal_dir}")
    sampled_normal = rng.sample(normal_files, min(N_NORMAL_FILES, len(normal_files)))
    for f in sampled_normal:
        entries.append((f, 0, 0, "underhang"))  # label=Normal, severity=Minor default

    # Bearing fault files, both positions
    for position in ["underhang", "overhang"]:
        position_dir = os.path.join(RAW_DIR, position, position, position)
        if not os.path.exists(position_dir):
            print(f"WARNING: {position_dir} not found, skipping")
            continue

        for fault_folder, label in FAULT_FOLDER_MAP.items():
            fault_dir = os.path.join(position_dir, fault_folder)
            if not os.path.exists(fault_dir):
                print(f"WARNING: {fault_dir} not found, skipping")
                continue

            for load_folder, severity in LOAD_SEVERITY_MAP.items():
                load_dir = os.path.join(fault_dir, load_folder)
                if not os.path.exists(load_dir):
                    continue

                csv_files = sorted(glob.glob(os.path.join(load_dir, "*.csv")))
                sampled = rng.sample(csv_files, min(N_FILES_PER_FOLDER, len(csv_files)))
                for f in sampled:
                    entries.append((f, label, severity, position))

    return entries


# ---------------------------------------------------------------------------
# STEP 2 — Load signals from CSV
# ---------------------------------------------------------------------------

def load_signals(csv_path, position):
    """Returns (acoustic_signal, vibration_signal) as 1D numpy arrays."""
    df = pd.read_csv(csv_path, header=None)

    acoustic_signal = df.iloc[:, 7].to_numpy(dtype=np.float32)  # column 8: microphone

    if position == "underhang":
        vibration_signal = df.iloc[:, 1].to_numpy(dtype=np.float32)  # underhang axial
    else:
        vibration_signal = df.iloc[:, 4].to_numpy(dtype=np.float32)  # overhang axial

    return acoustic_signal, vibration_signal


def segment_signal(signal, sr, segment_length_sec):
    segment_len = int(sr * segment_length_sec)
    n_segments = len(signal) // segment_len
    return [signal[i * segment_len:(i + 1) * segment_len] for i in range(n_segments)]


# ---------------------------------------------------------------------------
# STEP 3 — Feature extraction (same as before)
# ---------------------------------------------------------------------------

def extract_mfcc(signal, sr):
    import librosa
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return fix_time_frames(mfcc, TARGET_TIME_FRAMES)


def extract_stft(signal, sr):
    _, _, Zxx = stft(signal, fs=sr, nperseg=126, noverlap=63)
    magnitude = np.abs(Zxx)[:64, :]
    return fix_time_frames(magnitude, TARGET_TIME_FRAMES)


def fix_time_frames(feature_2d, target_frames):
    current = feature_2d.shape[1]
    if current == target_frames:
        return feature_2d
    elif current > target_frames:
        return feature_2d[:, :target_frames]
    else:
        return np.pad(feature_2d, ((0, 0), (0, target_frames - current)), mode="constant")


# ---------------------------------------------------------------------------
# STEP 4 — Build the dataset
# ---------------------------------------------------------------------------

def build_dataset():
    entries = find_files()
    print(f"Found {len(entries)} source files to process")

    if not entries:
        raise RuntimeError(
            f"No files found under {RAW_DIR}. Check that the folder structure matches "
            "what's documented at the top of this script."
        )

    acoustic_list, vibration_list, label_list, severity_list, file_id_list = [], [], [], [], []

    for file_idx, (csv_path, label, severity, position) in enumerate(entries):
        if file_idx % 10 == 0:
            print(f"Processing file {file_idx + 1}/{len(entries)}: {os.path.basename(csv_path)}")

        try:
            acoustic_signal, vibration_signal = load_signals(csv_path, position)
        except Exception as e:
            print(f"  Skipping {csv_path}: {e}")
            continue

        acoustic_segments = segment_signal(acoustic_signal, SAMPLE_RATE, SEGMENT_LENGTH_SEC)
        vibration_segments = segment_signal(vibration_signal, SAMPLE_RATE, SEGMENT_LENGTH_SEC)
        n_segments = min(len(acoustic_segments), len(vibration_segments))

        for i in range(n_segments):
            acoustic_list.append(extract_mfcc(acoustic_segments[i], SAMPLE_RATE))
            vibration_list.append(extract_stft(vibration_segments[i], SAMPLE_RATE))
            label_list.append(label)
            severity_list.append(severity)
            file_id_list.append(file_idx * 1000 + i)  # unique per (source file, segment)

    acoustic_arr = np.array(acoustic_list, dtype=np.float32)
    vibration_arr = np.array(vibration_list, dtype=np.float32)
    labels_arr = np.array(label_list, dtype=np.int64)
    severity_arr = np.array(severity_list, dtype=np.int64)
    file_id_arr = np.array(file_id_list, dtype=np.int64)

    print(f"\nFinal shapes:")
    print(f"  acoustic:  {acoustic_arr.shape}")
    print(f"  vibration: {vibration_arr.shape}")
    print(f"  labels:    {labels_arr.shape}  (distribution: {np.bincount(labels_arr)})")
    print(f"  severity:  {severity_arr.shape}  (distribution: {np.bincount(severity_arr)})")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(OUT_FILE, acoustic=acoustic_arr, vibration=vibration_arr,
              labels=labels_arr, severity=severity_arr, file_id=file_id_arr)
    print(f"\nSaved to {OUT_FILE}")

    write_readme(labels_arr, severity_arr, len(entries))


def write_readme(labels_arr, severity_arr, n_files):
    severity_names = {0: "Minor", 1: "Moderate", 2: "Severe"}

    lines = [
        "# Data Summary (MaFaulDa)\n\n",
        f"Total samples: {len(labels_arr)} (from {n_files} source recordings)\n\n",
        "## Fault type distribution\n",
    ]
    for label_id, name in enumerate(FAULT_NAMES):
        count = int(np.sum(labels_arr == label_id))
        lines.append(f"- {name}: {count} samples\n")

    lines.append("\n## Severity distribution\n")
    for sev_id, name in severity_names.items():
        count = int(np.sum(severity_arr == sev_id))
        lines.append(f"- {name}: {count} samples\n")

    lines.append("\n## Important note on labels\n")
    lines.append(
        "MaFaulDa does not have an 'Inner Race' fault category like KAIST did. "
        "This is now a 4-class problem: Normal, Outer Race, Ball Fault, Cage Fault. "
        "Severity is derived from the imbalance load added during recording "
        "(0g/6g=Minor, 20g=Moderate, 35g=Severe) as a proxy, since MaFaulDa "
        "does not label bearing damage severity directly the way KAIST did.\n"
    )
    lines.append(f"\n- Sample rate: {SAMPLE_RATE} Hz\n")
    lines.append(f"- Segment length: {SEGMENT_LENGTH_SEC} second chunks\n")

    with open("data/README.md", "w") as f:
        f.writelines(lines)
    print("Also wrote data/README.md")


if __name__ == "__main__":
    build_dataset()