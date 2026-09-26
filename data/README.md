# Data Summary (MaFaulDa)

Total samples: 770 (from 154 source recordings)

## Fault type distribution
- Normal: 50 samples
- Outer Race: 240 samples
- Ball Fault: 240 samples
- Cage Fault: 240 samples

## Severity distribution
- Minor: 410 samples
- Moderate: 180 samples
- Severe: 180 samples

## Important note on labels
MaFaulDa does not have an 'Inner Race' fault category like KAIST did. This is now a 4-class problem: Normal, Outer Race, Ball Fault, Cage Fault. Severity is derived from the imbalance load added during recording (0g/6g=Minor, 20g=Moderate, 35g=Severe) as a proxy, since MaFaulDa does not label bearing damage severity directly the way KAIST did.

- Sample rate: 50000 Hz
- Segment length: 1.0 second chunks
