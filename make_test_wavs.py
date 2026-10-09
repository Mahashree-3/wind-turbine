import glob, os
import numpy as np
from scipy.io import wavfile

root = os.path.join("data", "raw_mafaulda")
os.makedirs("test_wavs", exist_ok=True)

def convert(path, out):
    data = np.loadtxt(path, delimiter=",", usecols=7)   # 8th column = microphone
    data = data / (np.max(np.abs(data)) + 1e-9)
    wavfile.write(out, 50000, data.astype(np.float32))
    print("Saved", out, "from", path)

# Normal
normal = glob.glob(os.path.join(root, "normal", "**", "*.csv"), recursive=True)
if normal:
    convert(normal[0], "test_wavs/normal.wav")

# One file per fault type
for fault in ["ball_fault", "cage_fault", "outer_race"]:
    files = glob.glob(os.path.join(root, "**", fault, "**", "*.csv"), recursive=True)
    if files:
        convert(files[0], f"test_wavs/{fault}.wav")
    else:
        print("No CSV found for", fault)