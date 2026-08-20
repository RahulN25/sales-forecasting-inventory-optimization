"""
Deep Learning Model Training & Evaluation Module for Walmart Sales Forecasting.

Provides deep learning model architectures (ResidualAttentionLSTM, LSTM, BiLSTM+Attention, 
1D-CNN, LSTM-MLP Hybrid, TabularMLP, TCN-Transformer), dataset building, loss objectives,
training loops, residual alpha calibration, and weighted ensembling.
"""

import copy
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Reproducibility & Device Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Evaluation Metrics
# ============================================================

def wmae_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
    holiday_weight: float = 5.0,
) -> float:
    """Calculates Walmart Weighted Mean Absolute Error (WMAE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    is_holiday = np.asarray(is_holiday, dtype=bool)

    if not (len(y_true) == len(y_pred) == len(is_holiday)):
        raise ValueError("Metric arrays must have equal length.")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Target or prediction array contains NaN/Inf.")

    weights = np.where(is_holiday, holiday_weight, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


# ============================================================
# PyTorch Dataset
# ============================================================

class ForecastDataset(Dataset):
    """Dataset holding input sequence tensors, scaled residual targets, holiday flags, and original baseline values."""

    def __init__(self, X: np.ndarray, y_scaled: np.ndarray, holiday: np.ndarray, baseline: np.ndarray):
        arrs = [np.asarray(a, dtype=np.float32) for a in (X, y_scaled, holiday, baseline)]
        for arr in arrs:
            if not np.isfinite(arr).all():
                raise ValueError("Dataset array contains NaN/Inf.")
        self.X, self.y, self.holiday, self.baseline = [
            torch.from_numpy(np.ascontiguousarray(a)) for a in arrs
        ]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.holiday[idx], self.baseline[idx]


# ============================================================
# Neural Network Architectures
# ============================================================

class ResidualAttentionLSTM(nn.Module):
    """
    Residual Attention LSTM combining Store/Dept embeddings, bidirectional sequence encoding,
    temporal attention pooling, current-row MLP feature processing, and gated residual fusion.
    """

    def __init__(
        self,
        input_dim: int,
        store_idx: Optional[int],
        dept_idx: Optional[int],
        type_idx: Optional[int] = None,
        n_stores: int = 64,
        n_depts: int = 128,
        n_types: int = 8,
        store_emb_dim: int = 12,
        dept_emb_dim: int = 16,
        type_emb_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        current_hidden: int = 128,
        attention_hidden: int = 64,
        dropout: float = 0.15,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.store_idx, self.dept_idx, self.type_idx = store_idx, dept_idx, type_idx
        self.store_emb = nn.Embedding(n_stores + 1, store_emb_dim) if store_idx is not None else None
        self.dept_emb = nn.Embedding(n_depts + 1, dept_emb_dim) if dept_idx is not None else None
        self.type_emb = nn.Embedding(n_types + 1, type_emb_dim) if type_idx is not None else None

        emb_dim = (store_emb_dim if self.store_emb else 0) + \
                  (dept_emb_dim if self.dept_emb else 0) + \
                  (type_emb_dim if self.type_emb else 0)
        seq_input_dim = input_dim + emb_dim

        self.input_norm = nn.LayerNorm(seq_input_dim)
        self.lstm = nn.LSTM(
            seq_input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        seq_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.attn = nn.Sequential(
            nn.Linear(seq_out_dim, attention_hidden), nn.Tanh(), nn.Linear(attention_hidden, 1, bias=False)
        )
        self.current_mlp = nn.Sequential(
            nn.Linear(seq_input_dim, current_hidden), nn.LayerNorm(current_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(current_hidden, current_hidden), nn.GELU()
        )
        fusion_dim = seq_out_dim + current_hidden
        self.gate = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout / 2), nn.Linear(64, 1)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _append_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x]
        for emb, idx in [(self.store_emb, self.store_idx), (self.dept_emb, self.dept_idx), (self.type_emb, self.type_idx)]:
            if emb is not None and idx is not None:
                ids = x[:, :, idx].round().long().clamp(0, emb.num_embeddings - 1)
                parts.append(emb(ids))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_norm(self._append_embeddings(x))
        seq_out, _ = self.lstm(z)
        attn_weights = torch.softmax(self.attn(seq_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(attn_weights * seq_out, dim=1)
        current = self.current_mlp(z[:, -1, :])
        fused = torch.cat([context, current], dim=-1)
        return self.head(fused * self.gate(fused)).squeeze(-1)


class LSTMModel(nn.Module):
    """Standard multi-layer LSTM sequence encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class BiLSTMWithAttention(nn.Module):
    """Bidirectional LSTM with Bahdanau self-attention mechanism."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_dim * 2
        self.attention_dense = nn.Linear(out_dim, 64)
        self.attention_v = nn.Linear(64, 1, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attention_v(torch.tanh(self.attention_dense(lstm_out))), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return self.head(context).squeeze(-1)


class CNN1DModel(nn.Module):
    """1D Temporal Convolutional Neural Network with skip connection and dual avg+max pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.skip = nn.Conv1d(input_dim, hidden_dim * 2, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        residual = self.skip(x)
        z = F.gelu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(self.drop(z)))
        z = F.gelu(z + residual)
        pooled = 0.5 * (z.mean(dim=-1) + z.amax(dim=-1))
        return self.head(pooled).squeeze(-1)


class LSTMMLPHybrid(nn.Module):
    """Hybrid architecture combining recurrent sequence extraction and tabular MLP features."""

    def __init__(self, input_dim: int, lstm_hidden: int = 128, mlp_hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, mlp_hidden), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden + mlp_hidden), nn.Linear(lstm_hidden + mlp_hidden, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        current = self.mlp(x[:, -1, :])
        return self.head(torch.cat([lstm_out[:, -1, :], current], dim=1)).squeeze(-1)


class TabularMLP(nn.Module):
    """Dense Multi-Layer Perceptron operating on the final sequence timestep."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -1, :]).squeeze(-1)


class CausalConv1d(nn.Module):
    """1D Causal Convolution layer preventing future lookahead leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Residual Causal TCN block with batch normalization and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(F.gelu(self.norm(self.conv(x))))
        return F.gelu(z + self.skip(x))


class TCNTransformerHybrid(nn.Module):
    """Temporal Convolutional Network + Multi-Head Self-Attention Transformer Hybrid (Optimized & Fast)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_dim, hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout),
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=False, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_tcn = self.tcn(x.permute(0, 2, 1))
        z_tcn = self.pool(z_tcn).permute(0, 2, 1)
        z_trans = self.transformer(z_tcn)
        pooled = torch.cat([z_trans.mean(dim=1), z_trans[:, -1, :]], dim=1)
        return self.head(pooled).squeeze(-1)


# ============================================================
# Sequence Building & Data Preparation
# ============================================================

def build_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "_residual_scaled",
    holiday_col: str = "_holiday_target",
    baseline_col_internal: str = "_baseline_sales",
    seq_len: int = 26,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constructs sliding window sequence tensors grouped by (Store, Dept) using edge padding."""
    X_all = df[list(feature_cols)].to_numpy(dtype=np.float32)
    y_all = df[target_col].to_numpy(dtype=np.float32)
    holiday_all = df[holiday_col].to_numpy(dtype=np.float32)
    baseline_all = df[baseline_col_internal].to_numpy(dtype=np.float32)
    row_indices = df.index.to_numpy()

    n_rows, n_features = X_all.shape
    X_seq = np.empty((n_rows, seq_len, n_features), dtype=np.float32)

    group_sizes = df.groupby(["Store", "Dept"], sort=False).size().to_numpy()
    ends = np.cumsum(group_sizes)
    starts = ends - group_sizes

    for s, e in zip(starts, ends):
        padded = np.pad(X_all[s:e], ((seq_len - 1, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=(seq_len, n_features))[:, 0, :, :]
        X_seq[s:e] = windows

    return X_seq, y_all, holiday_all, baseline_all, row_indices


def _clean_holiday_column(data: pd.DataFrame) -> None:
    if "IsHoliday" not in data.columns:
        data["IsHoliday"] = 0.0
        return
    if data["IsHoliday"].dtype == "object":
        mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
        data["IsHoliday"] = data["IsHoliday"].astype(str).str.strip().str.lower().map(mapping)
    else:
        data["IsHoliday"] = pd.to_numeric(data["IsHoliday"], errors="coerce")
    data["IsHoliday"] = data["IsHoliday"].fillna(0.0).clip(0.0, 1.0).astype(np.float32)


def _choose_baseline_column(data: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    candidates = ([preferred] if preferred else []) + ["Weekly_Sales_lag_52", "lag_52", "Weekly_Sales_lag_1", "lag_1"]
    for col in candidates:
        if col in data.columns:
            return col
    return None


def prepare_deep_learning_data(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    val_start: str = "2012-08-17",
    seq_len: int = 26,
    batch_size: int = 512,
    target_col: str = "Weekly_Sales",
    baseline_col: Optional[str] = None,
    num_workers: int = 0,
) -> Dict:
    """Prepares feature scalers, residual targets, historical sequences, and PyTorch DataLoaders."""
    data = df.copy()
    _clean_holiday_column(data)
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    val_start_ts = pd.Timestamp(val_start)
    train_mask = data["Date"] < val_start_ts
    if train_mask.sum() == 0 or (~train_mask).sum() == 0:
        raise ValueError("Training or validation partition is empty.")

    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    train_medians = data.loc[train_mask, list(feature_cols)].replace([np.inf, -np.inf], np.nan).median()
    data[list(feature_cols)] = data[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(train_medians).fillna(0.0)

    selected_baseline = _choose_baseline_column(data, preferred=baseline_col)
    if selected_baseline is not None:
        raw_baseline = pd.to_numeric(data[selected_baseline], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        raw_baseline = pd.Series(np.nan, index=data.index)

    train_group_mean = data.loc[train_mask].groupby(["Store", "Dept"])[target_col].mean()
    global_train_mean = float(data.loc[train_mask, target_col].mean())
    fallback = np.array([train_group_mean.get((s, d), global_train_mean) for s, d in zip(data["Store"], data["Dept"])])

    baseline_values = raw_baseline.to_numpy(dtype=np.float64)
    baseline_values[~np.isfinite(baseline_values)] = fallback[~np.isfinite(baseline_values)]
    data["_baseline_sales"] = baseline_values

    residual = data[target_col].to_numpy(dtype=np.float64) - baseline_values
    residual_scaler = StandardScaler()
    residual_scaler.fit(residual[train_mask.to_numpy()].reshape(-1, 1))
    data["_residual_scaled"] = residual_scaler.transform(residual.reshape(-1, 1)).reshape(-1).astype(np.float32)

    id_cols = {"Store", "Dept", "Type_Encoded", "IsHoliday"}
    scale_cols = [c for c in feature_cols if c not in id_cols]
    if scale_cols:
        x_scaler = StandardScaler()
        x_scaler.fit(data.loc[train_mask, scale_cols].astype(np.float64))
        scaled_vals = x_scaler.transform(data[scale_cols].astype(np.float64)).astype(np.float32)
        data[scale_cols] = scaled_vals
    else:
        x_scaler = None

    for col in id_cols:
        if col in feature_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(np.float32)
    data["_holiday_target"] = data["IsHoliday"].astype(np.float32)

    X_seq, y_seq, holiday_seq, baseline_seq, row_indices = build_sequences(
        df=data, feature_cols=feature_cols, target_col="_residual_scaled",
        holiday_col="_holiday_target", baseline_col_internal="_baseline_sales", seq_len=seq_len
    )

    seq_dates = data.loc[row_indices, "Date"].to_numpy()
    seq_train = seq_dates < np.datetime64(val_start_ts)
    seq_val = ~seq_train

    X_train, y_train, h_train, b_train = X_seq[seq_train], y_seq[seq_train], holiday_seq[seq_train], baseline_seq[seq_train]
    X_val, y_val_scaled, h_val, b_val = X_seq[seq_val], y_seq[seq_val], holiday_seq[seq_val], baseline_seq[seq_val]
    train_indices, val_indices = row_indices[seq_train], row_indices[seq_val]

    pin_memory = get_device() == "cuda"
    train_loader = DataLoader(
        ForecastDataset(X_train, y_train, h_train, b_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        ForecastDataset(X_val, y_val_scaled, h_val, b_val),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    val_df = data.loc[val_indices].copy().reset_index(drop=True)
    val_df["IsHoliday"] = val_df["IsHoliday"] > 0.5

    baseline_wmae = wmae_numpy(
        val_df[target_col].to_numpy(dtype=np.float64),
        val_df["_baseline_sales"].to_numpy(),
        val_df["IsHoliday"].to_numpy()
    )

    feature_cols = list(feature_cols)
    return {
        "data": data, "train_loader": train_loader, "val_loader": val_loader,
        "X_train_seq": X_train, "y_train_seq": y_train, "X_val_seq": X_val, "y_val_seq": y_val_scaled,
        "train_indices": train_indices, "val_indices": val_indices, "val_df": val_df,
        "x_scaler": x_scaler, "y_scaler": residual_scaler, "residual_scaler": residual_scaler,
        "scale_cols": scale_cols, "feature_cols": feature_cols, "target_col": target_col,
        "seq_len": seq_len, "val_start": val_start, "baseline_col": selected_baseline,
        "baseline_wmae": baseline_wmae,
        "store_idx": feature_cols.index("Store") if "Store" in feature_cols else None,
        "dept_idx": feature_cols.index("Dept") if "Dept" in feature_cols else None,
        "type_idx": feature_cols.index("Type_Encoded") if "Type_Encoded" in feature_cols else None,
        "holiday_col_idx": feature_cols.index("IsHoliday") if "IsHoliday" in feature_cols else None,
    }


def print_data_diagnostics(dl_data: Dict) -> None:
    """Prints diagnostic statistics for deep learning dataset tensors."""
    print("\nDeep Learning Data Diagnostics")
    print("-" * 60)
    for name in ["X_train_seq", "y_train_seq", "X_val_seq", "y_val_seq"]:
        arr = dl_data[name]
        print(f"{name:<16} shape={str(arr.shape):<22} finite={np.isfinite(arr).all()!s:<6} min={arr.min():.4f} max={arr.max():.4f}")
    print(f"Sequence length:   {dl_data['seq_len']}")
    print(f"Baseline feature:  {dl_data['baseline_col']}")
    print(f"Baseline WMAE:     {dl_data['baseline_wmae']:,.2f}")
    print(f"Store index:       {dl_data['store_idx']}")
    print(f"Dept index:        {dl_data['dept_idx']}")
    print(f"Holiday index:     {dl_data['holiday_col_idx']}")
    print("-" * 60)


# ============================================================
# Loss Functions & Model Training Loop
# ============================================================

def weighted_forecast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    holiday_flag: torch.Tensor,
    holiday_weight: float = 5.0,
    huber_beta: float = 0.5,
    mae_fraction: float = 0.65,
) -> torch.Tensor:
    """Combines holiday-weighted MAE loss and Huber loss for stable training."""
    weights = torch.where(holiday_flag > 0.5, holiday_weight, 1.0)
    err = torch.abs(pred - target)
    huber = torch.where(err < huber_beta, 0.5 * (err ** 2) / huber_beta, err - 0.5 * huber_beta)
    combined = mae_fraction * err + (1.0 - mae_fraction) * huber
    return torch.sum(weights * combined) / torch.sum(weights)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    residual_scaler: StandardScaler,
    max_epochs: int = 60,
    patience: int = 12,
    lr: float = 0.0008,
    weight_decay: float = 2e-5,
    holiday_weight: float = 5.0,
    huber_beta: float = 0.5,
    mae_fraction: float = 0.65,
    grad_clip: float = 1.0,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-5,
    device: Optional[str] = None,
    verbose: bool = True,
    use_amp: bool = False,
) -> Tuple[nn.Module, Dict]:
    """Trains neural network with early stopping based on original-scale validation WMAE."""
    device = torch.device(device or get_device())
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    residual_mean, residual_scale = float(residual_scaler.mean_[0]), float(residual_scaler.scale_[0])
    best_wmae, best_epoch, no_improve = float("inf"), 0, 0
    best_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "val_wmae": [], "lr": []}

    if verbose:
        print(f"\nTraining Configuration")
        print("-" * 42)
        print(f"Model:       {model.__class__.__name__}")
        print(f"Epochs:      {max_epochs}")
        print(f"Patience:    {patience}")
        print(f"Initial LR:  {lr}")
        print(f"Batch size:  {train_loader.batch_size}")
        print(f"Device:      {device}")
        print(f"AMP:         {amp_enabled}")
        print(f"Holiday wt.: {holiday_weight}")
        print("-" * 42)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_count = 0.0, 0

        for bx, by, holiday, _ in train_loader:
            bx, by, holiday = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), holiday.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                pred = model(bx)
                loss = weighted_forecast_loss(pred, by, holiday, holiday_weight, huber_beta, mae_fraction)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_loss_sum += float(loss.detach().cpu()) * len(bx)
            train_count += len(bx)

        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum, val_count = 0.0, 0
        val_true_orig, val_pred_orig, val_hld = [], [], []

        with torch.inference_mode():
            for bx, by, holiday, baseline in val_loader:
                bx_d, by_d, hld_d = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), holiday.to(device, non_blocking=True)
                pred = model(bx_d)
                loss = weighted_forecast_loss(pred, by_d, hld_d, holiday_weight, huber_beta, mae_fraction)

                val_loss_sum += float(loss.detach().cpu()) * len(bx)
                val_count += len(bx)

                pred_np = pred.detach().float().cpu().numpy().astype(np.float64)
                by_np = by.detach().float().cpu().numpy().astype(np.float64)
                base_np = baseline.numpy().astype(np.float64)

                val_pred_orig.append(base_np + pred_np * residual_scale + residual_mean)
                val_true_orig.append(base_np + by_np * residual_scale + residual_mean)
                val_hld.append(holiday.numpy())

        val_loss = val_loss_sum / val_count
        val_wmae = wmae_numpy(np.concatenate(val_true_orig), np.concatenate(val_pred_orig), np.concatenate(val_hld).astype(bool), holiday_weight)

        scheduler.step(val_wmae)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_wmae"].append(val_wmae)
        history["lr"].append(current_lr)

        if verbose:
            print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val WMAE: {val_wmae:,.2f} | LR: {current_lr:.2e}")

        if val_wmae < best_wmae - 0.5:
            best_wmae, best_epoch, no_improve = val_wmae, epoch, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping | Best epoch: {best_epoch} | Best WMAE: {best_wmae:,.2f}")
                break

    model.load_state_dict(best_weights)
    return model, history


# ============================================================
# Inference & Calibration Helpers
# ============================================================

def predict_scaled(model: nn.Module, loader: DataLoader, device: Optional[str] = None) -> np.ndarray:
    """Predicts scaled residual outputs for a DataLoader."""
    device = torch.device(device or get_device())
    model = model.to(device)
    model.eval()
    out = []
    with torch.inference_mode():
        for bx, _, _, _ in loader:
            pred = model(bx.to(device, non_blocking=True)).detach().float().cpu().numpy()
            out.append(pred)
    return np.concatenate(out)


def inverse_transform_predictions(
    scaled_residual_predictions: np.ndarray,
    residual_scaler: StandardScaler,
    baseline_predictions: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Converts scaled residual predictions back to the original dollar sales scale."""
    scaled_res = np.asarray(scaled_residual_predictions, dtype=np.float64).reshape(-1, 1)
    res_pred = residual_scaler.inverse_transform(scaled_res).reshape(-1)
    return np.asarray(baseline_predictions, dtype=np.float64) + alpha * res_pred


def get_loader_baselines(loader: DataLoader) -> np.ndarray:
    """Extracts baseline forecast vector from a DataLoader."""
    return np.concatenate([b.numpy().astype(np.float64) for _, _, _, b in loader])


def get_loader_holidays(loader: DataLoader) -> np.ndarray:
    """Extracts boolean holiday flags from a DataLoader."""
    return np.concatenate([h.numpy() for _, _, h, _ in loader]).astype(bool)


def predict_original_scale(
    model: nn.Module,
    loader: DataLoader,
    residual_scaler: StandardScaler,
    device: Optional[str] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """Generates predictions on the original dollar sales scale."""
    scaled_res = predict_scaled(model, loader, device=device)
    baselines = get_loader_baselines(loader)
    return inverse_transform_predictions(scaled_res, residual_scaler, baselines, alpha=alpha)


def tune_residual_alpha(
    model: nn.Module,
    val_loader: DataLoader,
    residual_scaler: StandardScaler,
    y_true: np.ndarray,
    is_holiday: np.ndarray,
    device: Optional[str] = None,
    alpha_grid: Optional[Iterable[float]] = None,
) -> Tuple[float, float]:
    """Finds the optimal residual shrinkage scalar alpha maximizing validation WMAE."""
    if alpha_grid is None:
        alpha_grid = np.arange(0.40, 1.31, 0.05)
    scaled_pred = predict_scaled(model, val_loader, device=device)
    residual_pred = residual_scaler.inverse_transform(scaled_pred.reshape(-1, 1)).reshape(-1)
    baseline = get_loader_baselines(val_loader)

    best_alpha, best_score = 1.0, float("inf")
    for alpha in alpha_grid:
        pred = baseline + float(alpha) * residual_pred
        score = wmae_numpy(y_true, pred, is_holiday)
        if score < best_score:
            best_score, best_alpha = score, float(alpha)
    return best_alpha, best_score


# ============================================================
# Ensemble & Model Builder Functions
# ============================================================

def weighted_ensemble(
    predictions: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Computes a weighted average prediction matrix."""
    pred_matrix = np.vstack([np.asarray(p, dtype=np.float64) for p in predictions])
    if not np.isfinite(pred_matrix).all():
        raise ValueError("Ensemble predictions contain NaN/Inf.")
    n_models = pred_matrix.shape[0]
    weights = np.ones(n_models, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        raise ValueError("Weights must sum to > 0")
    return np.average(pred_matrix, axis=0, weights=weights / weights.sum())


def inverse_wmae_weights(scores: Sequence[float], power: float = 3.0) -> np.ndarray:
    """Computes normalized inverse-power WMAE ensemble weights."""
    scores = np.asarray(scores, dtype=np.float64)
    if np.any(scores <= 0):
        raise ValueError("WMAE scores must be positive.")
    inv = 1.0 / np.power(scores, power)
    return inv / inv.sum()


def build_best_model(dl_data: Dict, hidden_dim: int = 128, dropout: float = 0.15) -> ResidualAttentionLSTM:
    """Constructs ResidualAttentionLSTM with maximum embedding bounds derived from dataset."""
    data = dl_data["data"]
    max_store = int(pd.to_numeric(data["Store"], errors="coerce").fillna(0).max())
    max_dept = int(pd.to_numeric(data["Dept"], errors="coerce").fillna(0).max())
    max_type = int(pd.to_numeric(data["Type_Encoded"], errors="coerce").fillna(0).max()) if "Type_Encoded" in data.columns else 0

    return ResidualAttentionLSTM(
        input_dim=len(dl_data["feature_cols"]),
        store_idx=dl_data["store_idx"],
        dept_idx=dl_data["dept_idx"],
        type_idx=dl_data["type_idx"],
        n_stores=max_store + 2,
        n_depts=max_dept + 2,
        n_types=max_type + 2,
        store_emb_dim=12,
        dept_emb_dim=16,
        type_emb_dim=4,
        hidden_dim=hidden_dim,
        num_layers=2,
        current_hidden=128,
        attention_hidden=64,
        dropout=dropout,
        bidirectional=True,
    )


def recommended_configs() -> List[Dict]:
    """Returns curated hyperparameter configuration presets."""
    return [
        {"seq_len": 26, "hidden_dim": 128, "dropout": 0.10, "lr": 8e-4},
        {"seq_len": 26, "hidden_dim": 160, "dropout": 0.15, "lr": 7e-4},
        {"seq_len": 52, "hidden_dim": 128, "dropout": 0.15, "lr": 7e-4},
    ]
