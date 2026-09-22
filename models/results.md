# Model Training Results

- Total samples: 360 (train: 288, test: 72)
- Epochs: 40
- Test fault classification accuracy: **100.00%**
- Test severity classification accuracy: **100.00%**

## Fault Confusion Matrix

```
[[24  0  0]
 [ 0 24  0]
 [ 0  0 24]]
```
Labels in order: ['Normal', 'Inner Race (BPFI)', 'Outer Race (BPFO)']

See training_curve.png and confusion_matrix.png in model/saved/ for plots.

## Notes
- Model: CAVF-Net style — CNN encoders + bidirectional cross-attention + causal weighted fusion.
- RUL (Remaining Useful Life) is NOT predicted by this model — it uses a rule-based mapping from severity (see predict.py), since no run-to-failure labeled data was available.
