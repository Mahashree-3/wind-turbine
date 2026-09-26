"""
train.py — P2 (Model Lead)

Loads data/processed/features.npz (produced by P1's preprocess.py),
trains CAVFNet on fault type + severity classification, evaluates on a
held-out test split, and saves:
    model/saved/model.pth   — trained weights
    model/results.md        — accuracy, confusion matrix, training curve

RUN FROM THE PROJECT ROOT (wind-turbine7), not from inside model/:
    python model/train.py

INSTALL REQUIRED PACKAGES:
    pip install torch scikit-learn matplotlib numpy
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from model import CAVFNet

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FEATURES_FILE = "data/processed/features.npz"
MODEL_OUT_DIR = "model/saved"
MODEL_OUT_FILE = os.path.join(MODEL_OUT_DIR, "model.pth")
RESULTS_FILE = "model/results.md"
CURVE_IMAGE = "model/saved/training_curve.png"
CONFUSION_IMAGE = "model/saved/confusion_matrix.png"

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_SEED = 42

FAULT_NAMES = ["Normal", "Outer Race", "Ball Fault", "Cage Fault"]
SEVERITY_NAMES = ["Minor", "Moderate", "Severe"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BearingDataset(Dataset):
    def __init__(self, acoustic, vibration, labels, severity):
        self.acoustic = torch.tensor(acoustic, dtype=torch.float32)
        self.vibration = torch.tensor(vibration, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.severity = torch.tensor(severity, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.acoustic[idx], self.vibration[idx], self.labels[idx], self.severity[idx]


def normalize(arr):
    """Simple per-array standardization so the CNN trains more stably."""
    mean = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mean) / std


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_model():
    print(f"Using device: {DEVICE}")

    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            f"{FEATURES_FILE} not found. Run data/preprocess.py first (P1's part)."
        )

    data = np.load(FEATURES_FILE, allow_pickle=True)
    acoustic = normalize(data["acoustic"])
    vibration = normalize(data["vibration"])
    labels = data["labels"]
    severity = data["severity"]
    file_id = data["file_id"] if "file_id" in data else None

    print(f"Loaded {len(labels)} samples")
    print(f"  Fault distribution: {np.bincount(labels)}")
    print(f"  Severity distribution: {np.bincount(severity)}")

    if file_id is not None:
        # Leakage-free split: all augmented copies of the same 1-second segment
        # (file_id = source_file * 1000 + second_index) stay together on one
        # side of the split. We split WITHIN each fault class so every class
        # remains represented in both train and test — with only 5 source
        # recordings total (1 per class), holding out entire files would wipe
        # whole classes from training, so we hold out segments instead.
        rng = np.random.RandomState(RANDOM_SEED)
        train_idx_list, test_idx_list = [], []

        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_segment_ids = np.unique(file_id[cls_mask])
            rng.shuffle(cls_segment_ids)
            n_test_segments = max(1, int(len(cls_segment_ids) * TEST_SIZE))
            test_segments = set(cls_segment_ids[:n_test_segments])

            for seg_id in cls_segment_ids:
                seg_indices = np.where(file_id == seg_id)[0]
                if seg_id in test_segments:
                    test_idx_list.extend(seg_indices)
                else:
                    train_idx_list.extend(seg_indices)

        train_idx = np.array(train_idx_list)
        test_idx = np.array(test_idx_list)
        print(f"  Segment-level split (leakage-free): train={len(train_idx)}, test={len(test_idx)}. "
              f"All noisy copies of a given 1-second segment stay on one side of the split.")
    else:
        # Fallback if features.npz doesn't have file_id (older preprocess.py version)
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
        )
        print("  WARNING: no file_id found in features.npz — using a random chunk split, "
              "which may leak information between train and test. Re-run the updated "
              "preprocess.py to fix this.")

    train_ds = BearingDataset(acoustic[train_idx], vibration[train_idx], labels[train_idx], severity[train_idx])
    test_ds = BearingDataset(acoustic[test_idx], vibration[test_idx], labels[test_idx], severity[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CAVFNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for acoustic_batch, vibration_batch, label_batch, severity_batch in train_loader:
            acoustic_batch = acoustic_batch.to(DEVICE)
            vibration_batch = vibration_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            severity_batch = severity_batch.to(DEVICE)

            optimizer.zero_grad()
            fault_logits, severity_logits = model(acoustic_batch, vibration_batch)

            loss = criterion(fault_logits, label_batch) + criterion(severity_logits, severity_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * label_batch.size(0)
            correct += (fault_logits.argmax(dim=1) == label_batch).sum().item()
            total += label_batch.size(0)

        avg_loss = epoch_loss / total
        train_acc = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {avg_loss:.4f} | train fault acc: {train_acc:.3f}")

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    model.eval()
    all_fault_preds, all_fault_true = [], []
    all_severity_preds, all_severity_true = [], []

    with torch.no_grad():
        for acoustic_batch, vibration_batch, label_batch, severity_batch in test_loader:
            acoustic_batch = acoustic_batch.to(DEVICE)
            vibration_batch = vibration_batch.to(DEVICE)

            fault_logits, severity_logits = model(acoustic_batch, vibration_batch)

            all_fault_preds.extend(fault_logits.argmax(dim=1).cpu().numpy())
            all_fault_true.extend(label_batch.numpy())
            all_severity_preds.extend(severity_logits.argmax(dim=1).cpu().numpy())
            all_severity_true.extend(severity_batch.numpy())

    fault_accuracy = accuracy_score(all_fault_true, all_fault_preds)
    severity_accuracy = accuracy_score(all_severity_true, all_severity_preds)
    fault_cm = confusion_matrix(all_fault_true, all_fault_preds)

    print(f"\nTest fault classification accuracy:    {fault_accuracy:.4f}")
    print(f"Test severity classification accuracy: {severity_accuracy:.4f}")
    print(f"Confusion matrix (fault):\n{fault_cm}")

    # -----------------------------------------------------------------
    # Save model + plots + results.md
    # -----------------------------------------------------------------
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT_FILE)
    print(f"\nSaved model to {MODEL_OUT_FILE}")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(train_accuracies, label="Train fault accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training curve")
    plt.tight_layout()
    plt.savefig(CURVE_IMAGE)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(fault_cm, cmap="Blues")
    plt.title("Fault Type Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(FAULT_NAMES)), FAULT_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(FAULT_NAMES)), FAULT_NAMES)
    for i in range(len(FAULT_NAMES)):
        for j in range(len(FAULT_NAMES)):
            plt.text(j, i, str(fault_cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CONFUSION_IMAGE)
    plt.close()

    write_results_md(fault_accuracy, severity_accuracy, fault_cm, len(labels), len(train_idx), len(test_idx))


def write_results_md(fault_acc, severity_acc, fault_cm, total_samples, n_train, n_test):
    lines = [
        "# Model Training Results\n\n",
        f"- Total samples: {total_samples} (train: {n_train}, test: {n_test})\n",
        f"- Epochs: {EPOCHS}\n",
        f"- Test fault classification accuracy: **{fault_acc*100:.2f}%**\n",
        f"- Test severity classification accuracy: **{severity_acc*100:.2f}%**\n\n",
        "## Fault Confusion Matrix\n\n",
        "```\n",
        f"{fault_cm}\n",
        "```\n",
        f"Labels in order: {FAULT_NAMES}\n\n",
        "See training_curve.png and confusion_matrix.png in model/saved/ for plots.\n\n",
        "## Notes\n",
        "- Model: CAVF-Net style — CNN encoders + bidirectional cross-attention + causal weighted fusion.\n",
        "- RUL (Remaining Useful Life) is NOT predicted by this model — it uses a rule-based "
        "mapping from severity (see predict.py), since no run-to-failure labeled data was available.\n",
    ]
    with open(RESULTS_FILE, "w") as f:
        f.writelines(lines)
    print(f"Wrote {RESULTS_FILE}")


if __name__ == "__main__":
    train_model()