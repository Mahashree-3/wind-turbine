import sys
import numpy as np
from scipy.io import wavfile

csv_path = sys.argv[1]
out_path = sys.argv[2]

data = np.loadtxt(csv_path, delimiter=",", usecols=7)  # 8th column = microphone
data = data / (np.max(np.abs(data)) + 1e-9)            # normalize
wavfile.write(out_path, 50000, data.astype(np.float32))  # MaFaulDa sampling rate = 50 kHz
print("Saved", out_path)