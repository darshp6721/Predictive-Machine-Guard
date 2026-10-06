"""
Deep Learning Architectures for Predictive-Machine Guard (`ml_engine/deep_learning.py`).

Implements 3 PyTorch Deep Learning Models for Industrial Failure Classification:
1. ANN (Artificial Neural Network / Deep Multi-Layer Perceptron with BatchNorm & Dropout)
2. 1D-CNN (1D Convolutional Neural Network extracting local multi-sensor patterns)
3. BiLSTM (Bidirectional Long Short-Term Memory Network capturing sequential feature dependencies)

Each model exposes a unified scikit-learn compatible interface (`fit`, `predict`, `predict_proba`)
plus training & validation loss history curves for the Web Dashboard.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class _ANNNet(nn.Module):
    """Deep Multi-Layer Perceptron (ANN) with BatchNorm and Dropout."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _Conv1DCNNNet(nn.Module):
    """1D Convolutional Neural Network (1D-CNN) over multi-sensor channels."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_features) -> (batch, 1, in_features)
        x_3d = x.unsqueeze(1)
        features = self.conv_block(x_3d)
        return self.head(features).squeeze(-1)


class _BiLSTMNet(nn.Module):
    """Bidirectional LSTM (BiLSTM) recurrent network across ordered sensor features."""

    def __init__(self, in_features: int, hidden_size: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 24),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len=in_features) -> (batch, seq_len, 1)
        x_seq = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x_seq)
        # Pool across sequence steps (mean + max context fusion)
        pooled = lstm_out.mean(dim=1)
        normed = self.norm(pooled)
        return self.fc(normed).squeeze(-1)


class DeepLearningClassifierWrapper:
    """
    Scikit-learn compatible wrapper around PyTorch ANN, 1D-CNN, and BiLSTM models.
    Includes automatic StandardScaler normalization, class-weighted BCEWithLogitsLoss,
    and epoch-level training/validation loss logging.
    """

    def __init__(
        self,
        arch: str = "ANN",
        epochs: int = 18,
        batch_size: int = 128,
        lr: float = 0.0035,
        seed: int = 42,
    ) -> None:
        self.arch = arch.upper()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.scaler = StandardScaler()
        self.model: Optional[nn.Module] = None
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def _build_net(self, in_features: int) -> nn.Module:
        torch.manual_seed(self.seed)
        if self.arch == "ANN":
            return _ANNNet(in_features)
        elif self.arch in {"CNN", "1D-CNN"}:
            return _Conv1DCNNNet(in_features)
        elif self.arch == "BILSTM":
            return _BiLSTMNet(in_features)
        raise ValueError(f"Unsupported architecture: {self.arch}")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "DeepLearningClassifierWrapper":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

        X_scaled = self.scaler.fit_transform(X_arr).astype(np.float32)
        y_float = y_arr.astype(np.float32)

        in_features = X_scaled.shape[1]
        self.model = self._build_net(in_features)

        # Compute mild positive class weight to handle residual class imbalance
        n_pos = max(float(np.sum(y_float == 1.0)), 1.0)
        n_neg = max(float(np.sum(y_float == 0.0)), 1.0)
        pos_weight_val = min(max(n_neg / n_pos, 1.0), 4.5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_float))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_tensors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if X_val is not None and y_val is not None:
            Xv_arr = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
            yv_arr = y_val.values if isinstance(y_val, pd.Series) else np.asarray(y_val)
            Xv_scaled = self.scaler.transform(Xv_arr).astype(np.float32)
            val_tensors = (
                torch.from_numpy(Xv_scaled),
                torch.from_numpy(yv_arr.astype(np.float32)),
            )

        self.train_loss_history = []
        self.val_loss_history = []

        for _ in range(self.epochs):
            self.model.train()
            batch_losses = []
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_train_loss = round(float(np.mean(batch_losses)), 4)
            self.train_loss_history.append(epoch_train_loss)

            if val_tensors is not None:
                self.model.eval()
                with torch.no_grad():
                    v_logits = self.model(val_tensors[0])
                    v_loss = float(criterion(v_logits, val_tensors[1]).item())
                    self.val_loss_history.append(round(v_loss, 4))

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Deep learning model has not been fitted yet.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_arr).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_scaled))
            probs_pos = torch.sigmoid(logits).cpu().numpy()
        probs_pos = np.clip(probs_pos, 1e-5, 1.0 - 1e-5)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
