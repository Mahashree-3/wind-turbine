# Data Summary
Total samples: 360

## Fault type distribution
- Normal: 120 samples
- BPFI: 120 samples
- BPFO: 120 samples

## Severity distribution
- Minor: 240 samples
- Moderate: 120 samples
- Severe: 0 samples

## Preprocessing applied
- Segment length: 1.0 second chunks
- Acoustic -> MFCC, 40 coefficients, 128 time frames
- Vibration -> STFT magnitude, 64 frequency bins, 128 time frames
- Sample rate: 25600 Hz
- Source: KAIST bearing dataset (Normal, BPFI, BPFO only; Unbalance/Misalign excluded)
