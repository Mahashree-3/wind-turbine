"""
model.py — P2 (Model Lead)

Defines the CAVF-Net-style architecture:
    Acoustic -> CNN encoder ---\
                                 Cross-Attention -> Causal Weighted Fusion -> Fault + Severity heads
    Vibration -> CNN encoder --/

Input shapes (must match what P1's preprocess.py produces):
    acoustic:  (batch, 40, 128)
    vibration: (batch, 64, 128)

Outputs:
    fault_logits:    (batch, 3)   -> Normal, Inner race (BPFI), Outer race (BPFO)
    severity_logits: (batch, 3)   -> Minor, Moderate, Severe
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Turns a (1, F, T) spectrogram-like input into a sequence of embeddings over time."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        out = self.conv(x)                       # (B, 32, F/4, T/4)
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T, C*F) — sequence over time
        return out


class CAVFNet(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_fault_classes=4, num_severity_classes=3):
        super().__init__()

        self.acoustic_encoder = CNNEncoder()
        self.vibration_encoder = CNNEncoder()

        # LazyLinear auto-detects the flattened CNN output size on first forward pass,
        # so we don't need to hand-calculate it for both branches.
        self.acoustic_proj = nn.LazyLinear(embed_dim)
        self.vibration_proj = nn.LazyLinear(embed_dim)

        # Bidirectional cross-attention: acoustic attends to vibration, and vice versa
        self.cross_attn_a2v = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_v2a = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Causal weighting: learns how much to trust acoustic vs vibration per sample
        self.causal_weight = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fault_head = nn.Linear(128, num_fault_classes)
        self.severity_head = nn.Linear(128, num_severity_classes)

    def forward(self, acoustic, vibration):
        # acoustic: (B, 40, 128), vibration: (B, 64, 128)
        a = acoustic.unsqueeze(1)  # (B, 1, 40, 128)
        v = vibration.unsqueeze(1)  # (B, 1, 64, 128)

        a_feat = self.acoustic_proj(self.acoustic_encoder(a))   # (B, T, embed_dim)
        v_feat = self.vibration_proj(self.vibration_encoder(v))  # (B, T, embed_dim)

        # Bidirectional cross-attention
        a_enh, _ = self.cross_attn_a2v(a_feat, v_feat, v_feat)
        v_enh, _ = self.cross_attn_v2a(v_feat, a_feat, a_feat)

        a_pool = a_enh.mean(dim=1)  # (B, embed_dim)
        v_pool = v_enh.mean(dim=1)

        # Causal weighted fusion
        weights = self.causal_weight(torch.cat([a_pool, v_pool], dim=-1))  # (B, 2)
        w_acoustic = weights[:, 0:1]
        w_vibration = weights[:, 1:2]
        fused = w_acoustic * a_pool + w_vibration * v_pool  # (B, embed_dim)

        x = self.fusion(fused)
        fault_logits = self.fault_head(x)
        severity_logits = self.severity_head(x)

        return fault_logits, severity_logits