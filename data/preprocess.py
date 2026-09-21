"""
preprocess.py — P1 (Data & Preprocessing Lead)

Automatically scans data/raw/acoustic and data/raw/vibration for KAIST
bearing .mat files, matches acoustic+vibration pairs by filename, and
builds features.npz for P2 (Model Lead), exactly as defined in CONTRACT.md.

Expected filenames (from the actual KAIST download):
    {Load}Nm_Normal.mat
    {Load}Nm_BPFI_{severity}.mat   -> Inner race fault
    {Load}Nm_BPFO_{severity}.mat   -> Outer race fault
Where severity is 03 (0.3mm=Minor), 10 (1.0mm=Moderate), 30 (3.0mm=Severe).
Unbalance and Misalign files are ignored — this project only uses bearing
faults (Normal, BPFI, BPFO).

FOLDER SETUP (do this first):
    data/raw/acoustic/   <- copy the contents of your downloaded "acoustic" folder here
    data/raw/vibration/  <- copy the contents of your downloaded "vibration" folder here

Output: data/processed/features.npz containing:
    acoustic  -> shape (N, 40, 128)   MFCC features
    vibration -> shape (N, 64, 128)   STFT magnitude features
    labels    -> shape (N,)           0=Normal, 1=Inner race, 2=Outer race
    severity  -> shape (N,)           0=Minor, 1=Moderate, 2=Severe

INSTALL REQUIRED PACKAGES:
    pip install numpy scipy librosa
"""

import os
import re
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import stft

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ACOUSTIC_DIR = "data/raw/acoustic"
VIBRATION_DIR = "data/raw/vibration"
OUT_DIR = "data/processed"
OUT_FILE = os.path.join(OUT_DIR, "features.npz")

SAMPLE_RATE = 25600           # KAIST sampling rate (Hz)
SEGMENT_LENGTH_SEC = 1.0      # length of each chunk we turn into one sample
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 200
TARGET_TIME_FRAMES = 128      # contract requires both feature types to match this

LABEL_MAP = {"Normal": 0, "BPFI": 1, "BPFO": 2}
SEVERITY_CODE_MAP = {"03": 0, "10": 1, "30": 2}   # 0.3mm=Minor, 1.0mm=Moderate, 3.0mm=Severe
DEFAULT_SEVERITY_FOR_NORMAL = 0  # "Minor" bucket, since Normal has no damage code

# Filename pattern: e.g. "0Nm_BPFI_03.mat" or "4Nm_Normal.mat"
FILENAME_PATTERN = re.compile(r"^\d+Nm_(Normal|BPFI|BPFO)(?:_(\d+))?\.mat$")


# ---------------------------------------------------------------------------
# STEP 1 — Discover and pair files
# ---------------------------------------------------------------------------

def parse_filename(filename):
    """
    Returns (fault_type, severity_id) if this file is a bearing fault file
    we care about, or None if it should be skipped (e.g. Unbalance/Misalign).
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    condition, severity_code = match.groups()
    if condition not in LABEL_MAP:
        return None

    if condition == "Normal":
        severity_id = DEFAULT_SEVERITY_FOR_NORMAL
    else:
        if severity_code not in SEVERITY_CODE_MAP:
            return None
        severity_id = SEVERITY_CODE_MAP[severity_code]

    return LABEL_MAP[condition], severity_id


def find_matched_pairs():
    """Find files that exist in BOTH acoustic and vibration folders with the same name."""
    acoustic_files = {os.path.basename(p) for p in glob.glob(os.path.join(ACOUSTIC_DIR, "*.mat"))}
    vibration_files = {os.path.basename(p) for p in glob.glob(os.path.join(VIBRATION_DIR, "*.mat"))}

    common_files = sorted(acoustic_files & vibration_files)

    pairs = []
    for filename in common_files:
        parsed = parse_filename(filename)
        if parsed is None:
            continue  # skip Unbalance/Misalign or anything unrecognized
        label, severity = parsed
        pairs.append((
            os.path.join(ACOUSTIC_DIR, filename),
            os.path.join(VIBRATION_DIR, filename),
            label,
            severity,
            filename,
        ))

    return pairs


# ---------------------------------------------------------------------------
# STEP 2 — Load .mat signals
# ---------------------------------------------------------------------------

def _drill_for_numeric_array(obj, depth=0, path_so_far=""):
    """
    KAIST .mat files were exported from LabVIEW/DIAdem, which saves signals
    as nested structs: root.y_values.values (or similar), alongside
    x_values and function_record which we don't need. This recursively
    digs through structs/object arrays until it finds the actual numeric
    array of samples.
    """
    if depth > 8:
        raise ValueError(f"Structure too deep to resolve (stopped at {path_so_far})")

    # A plain numeric numpy array -> this is the data we want
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        return np.squeeze(obj).astype(np.float32)

    # An object array wrapping something else -> unwrap one level
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.size == 0:
            raise ValueError(f"Empty object array at {path_so_far}")
        return _drill_for_numeric_array(np.ravel(obj)[0], depth + 1, path_so_far + "[0]")

    # A MATLAB struct (loaded with struct_as_record=False) -> has named fields
    if hasattr(obj, "_fieldnames"):
        fields = obj._fieldnames
        # Prefer fields that typically hold the actual sample values
        for preferred in ("values", "Values", "Data", "data", "y_values", "Y", "y"):
            if preferred in fields:
                return _drill_for_numeric_array(
                    getattr(obj, preferred), depth + 1, path_so_far + f".{preferred}"
                )
        # Fall back to the first field if nothing obvious matches
        return _drill_for_numeric_array(
            getattr(obj, fields[0]), depth + 1, path_so_far + f".{fields[0]}"
        )

    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        return _drill_for_numeric_array(obj[0], depth + 1, path_so_far + "[0]")

    raise ValueError(
        f"Could not find a numeric array (stopped at {path_so_far}, "
        f"got type {type(obj)}). Run the debug snippet at the bottom of "
        f"this file on one .mat file and share the printed structure."
    )


def load_mat_signal(path):
    """Load a KAIST .mat file (LabVIEW/DIAdem struct format) and return a 1D numpy signal."""
    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    keys = [k for k in data.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No data variables found in {path}")

    root = data[keys[0]]
    signal = _drill_for_numeric_array(root, path_so_far=keys[0])

    if signal.ndim > 1:
        signal = signal[:, 0] if signal.shape[0] >= signal.shape[1] else signal[0, :]

    return signal.astype(np.float32)


def segment_signal(signal, sr, segment_length_sec):
    segment_len = int(sr * segment_length_sec)
    n_segments = len(signal) // segment_len
    return [signal[i * segment_len:(i + 1) * segment_len] for i in range(n_segments)]


# ---------------------------------------------------------------------------
# STEP 3 — Feature extraction
# ---------------------------------------------------------------------------

def extract_mfcc(signal, sr):
    import librosa
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
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
    pairs = find_matched_pairs()

    if not pairs:
        raise RuntimeError(
            "No matching acoustic+vibration file pairs found. Check that "
            f"{ACOUSTIC_DIR} and {VIBRATION_DIR} both exist and contain "
            "the .mat files with matching names (e.g. 0Nm_BPFI_03.mat in both)."
        )

    print(f"Found {len(pairs)} matched file pairs:")
    for _, _, label, severity, filename in pairs:
        print(f"  {filename}  -> label={label}, severity={severity}")

    acoustic_list, vibration_list, label_list, severity_list = [], [], [], []

    for acoustic_path, vibration_path, label, severity, filename in pairs:
        print(f"\nProcessing {filename} ...")
        acoustic_signal = load_mat_signal(acoustic_path)
        vibration_signal = load_mat_signal(vibration_path)

        acoustic_segments = segment_signal(acoustic_signal, SAMPLE_RATE, SEGMENT_LENGTH_SEC)
        vibration_segments = segment_signal(vibration_signal, SAMPLE_RATE, SEGMENT_LENGTH_SEC)
        n_pairs = min(len(acoustic_segments), len(vibration_segments))
        print(f"  {n_pairs} one-second samples extracted")

        for i in range(n_pairs):
            acoustic_list.append(extract_mfcc(acoustic_segments[i], SAMPLE_RATE))
            vibration_list.append(extract_stft(vibration_segments[i], SAMPLE_RATE))
            label_list.append(label)
            severity_list.append(severity)

    acoustic_arr = np.array(acoustic_list, dtype=np.float32)
    vibration_arr = np.array(vibration_list, dtype=np.float32)
    labels_arr = np.array(label_list, dtype=np.int64)
    severity_arr = np.array(severity_list, dtype=np.int64)

    print(f"\nFinal shapes:")
    print(f"  acoustic:  {acoustic_arr.shape}   (expected (N, 40, 128))")
    print(f"  vibration: {vibration_arr.shape}   (expected (N, 64, 128))")
    print(f"  labels:    {labels_arr.shape}")
    print(f"  severity:  {severity_arr.shape}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(OUT_FILE, acoustic=acoustic_arr, vibration=vibration_arr,
              labels=labels_arr, severity=severity_arr)
    print(f"\nSaved to {OUT_FILE}")

    write_readme(labels_arr, severity_arr)


def write_readme(labels_arr, severity_arr):
    label_names = {v: k for k, v in LABEL_MAP.items()}
    severity_names = {0: "Minor", 1: "Moderate", 2: "Severe"}

    lines = ["# Data Summary\n", f"Total samples: {len(labels_arr)}\n", "\n## Fault type distribution\n"]
    for label_id, name in label_names.items():
        count = int(np.sum(labels_arr == label_id))
        lines.append(f"- {name}: {count} samples\n")

    lines.append("\n## Severity distribution\n")
    for sev_id, name in severity_names.items():
        count = int(np.sum(severity_arr == sev_id))
        lines.append(f"- {name}: {count} samples\n")

    lines.append("\n## Preprocessing applied\n")
    lines.append(f"- Segment length: {SEGMENT_LENGTH_SEC} second chunks\n")
    lines.append(f"- Acoustic -> MFCC, {N_MFCC} coefficients, {TARGET_TIME_FRAMES} time frames\n")
    lines.append(f"- Vibration -> STFT magnitude, 64 frequency bins, {TARGET_TIME_FRAMES} time frames\n")
    lines.append(f"- Sample rate: {SAMPLE_RATE} Hz\n")
    lines.append(f"- Source: KAIST bearing dataset (Normal, BPFI, BPFO only; Unbalance/Misalign excluded)\n")

    with open("data/README.md", "w") as f:
        f.writelines(lines)
    print("Also wrote data/README.md")


if __name__ == "__main__":
    build_dataset()


# ---------------------------------------------------------------------------
# DEBUG HELPER — only run this manually if load_mat_signal() still fails.
# In a Python shell:
#   from preprocess import debug_mat_structure
#   debug_mat_structure("data/raw/acoustic/0Nm_BPFI_03.mat")
# It prints the nested field names so we can see exactly where the numbers are.
# ---------------------------------------------------------------------------

def debug_mat_structure(path, obj=None, depth=0, label="root"):
    if obj is None:
        data = loadmat(path, struct_as_record=False, squeeze_me=True)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print(f"Top-level keys: {keys}")
        obj = data[keys[0]]
        label = keys[0]

    indent = "  " * depth
    if isinstance(obj, np.ndarray):
        print(f"{indent}{label}: ndarray, shape={obj.shape}, dtype={obj.dtype}")
        if obj.dtype == object and obj.size > 0 and depth < 6:
            debug_mat_structure(path, np.ravel(obj)[0], depth + 1, label + "[0]")
    elif hasattr(obj, "_fieldnames"):
        print(f"{indent}{label}: struct with fields {obj._fieldnames}")
        if depth < 6:
            for f in obj._fieldnames:
                debug_mat_structure(path, getattr(obj, f), depth + 1, f)
    else:
        print(f"{indent}{label}: {type(obj)} = {obj}")