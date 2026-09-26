# Model Training Results

- Total samples: 770 (train: 616, test: 154)
- Epochs: 40
- Test fault classification accuracy: **91.56%**
- Test severity classification accuracy: **83.77%**

## Fault Confusion Matrix

```
[[ 8  1  0  1]
 [ 0 41  0  7]
 [ 0  1 47  0]
 [ 1  2  0 45]]
```
Labels in order: ['Normal', 'Outer Race', 'Ball Fault', 'Cage Fault']

See training_curve.png and confusion_matrix.png in model/saved/ for plots.

## Notes
- Model: CAVF-Net style — CNN encoders + bidirectional cross-attention + causal weighted fusion.
- RUL (Remaining Useful Life) is NOT predicted by this model — it uses a rule-based mapping from severity (see predict.py), since no run-to-failure labeled data was available.
