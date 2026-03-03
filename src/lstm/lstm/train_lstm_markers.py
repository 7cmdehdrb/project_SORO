# train_lstm_from_two_logs.py
# Build train/val/test from TWO continuous 10-min logs, then train LSTM.
# Center point: marker_2 (relative positions marker_3..7 w.r.t marker_2)

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


@dataclass
class TrainConfig:
    window: int = 60
    horizon: int = 1
    stride: int = 1
    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    pos_loss_weight: float = 1.0
    valid_loss_weight: float = 0.2
    grad_clip: float = 1.0
    seed: int = 42


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_continuous_df(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time order (no shuffle) to preserve temporal continuity.
    """
    if (
        not (0 < train_ratio < 1)
        or not (0 <= val_ratio < 1)
        or (train_ratio + val_ratio >= 1)
    ):
        raise ValueError("Invalid ratios: require 0<train<1, 0<=val<1, train+val<1")

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # remainder -> test
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


class SlidingWindowDataset(Dataset):
    """
    X: [window, 1] from data[1]
    Y: relative markers (dx,dy,dz,valid)
       center marker: marker_2, predict marker_3..7 -> 5 markers
       dims = 5*(3+1)=20
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: TrainConfig,
        input_col: str,
        center_xyz_cols: Tuple[str, str, str],  # marker_2_x/y/z
        out_marker_ids: List[int],  # [3,4,5,6,7]
        normalize_stats: Optional[dict] = None,
        fit_stats: bool = False,
    ):
        self.cfg = cfg

        # Required columns
        pos_cols = []
        valid_cols = []
        for mid in out_marker_ids:
            pos_cols.extend([f"marker_{mid}_x", f"marker_{mid}_y", f"marker_{mid}_z"])
            valid_cols.append(f"marker_{mid}_valid")

        required = [input_col, *center_xyz_cols, *pos_cols, *valid_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        x = df[input_col].astype(np.float32).to_numpy()  # [T]
        c = df[list(center_xyz_cols)].astype(np.float32).to_numpy()  # [T,3]
        out_pos = df[pos_cols].astype(np.float32).to_numpy()  # [T, 15]
        out_valid = df[valid_cols].astype(np.float32).to_numpy()  # [T, 5]

        # relative positions: marker_i - center(marker_2)
        out_pos = out_pos.reshape(len(df), len(out_marker_ids), 3)  # [T,5,3]
        rel_pos = out_pos - c[:, None, :]  # [T,5,3]
        rel_pos = rel_pos.reshape(len(df), len(out_marker_ids) * 3)  # [T,15]

        y = np.concatenate([rel_pos, out_valid], axis=1).astype(np.float32)  # [T,20]

        T = len(df)
        self.window = cfg.window
        self.horizon = cfg.horizon
        self.stride = cfg.stride

        self.max_start = T - (self.window + self.horizon)
        if self.max_start < 0:
            raise ValueError(
                f"Too short: T={T}, window={self.window}, horizon={self.horizon}"
            )

        self.starts = np.arange(0, self.max_start + 1, self.stride, dtype=np.int64)

        # normalization
        if normalize_stats is None:
            normalize_stats = {}

        if fit_stats:
            x_mean = float(x.mean())
            x_std = float(x.std() + 1e-8)

            y_pos = y[:, : len(out_marker_ids) * 3]  # 15
            y_mean = y_pos.mean(axis=0).astype(np.float32)
            y_std_pos = (y_pos.std(axis=0) + 1e-8).astype(np.float32)

            normalize_stats = {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std_pos": y_std_pos,
            }

        self.stats = normalize_stats
        self.x = x
        self.y = y
        self.pos_dim = len(out_marker_ids) * 3  # 15
        self.total_dim = self.pos_dim + len(out_marker_ids)  # 20

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        w = self.window
        h = self.horizon
        t_y = i + w - 1 + h

        x_seq = self.x[i : i + w]  # [w]
        y_t = self.y[t_y]  # [20]

        # normalize x
        x_seq = (x_seq - self.stats["x_mean"]) / self.stats["x_std"]
        x_seq = x_seq[:, None].astype(np.float32)  # [w,1]

        # normalize position part only
        y_pos = y_t[: self.pos_dim]
        y_valid = y_t[self.pos_dim :]
        y_pos = (y_pos - self.stats["y_mean"]) / self.stats["y_std_pos"]

        y_out = np.concatenate([y_pos, y_valid], axis=0).astype(np.float32)
        return torch.from_numpy(x_seq), torch.from_numpy(y_out)


class LSTMRegressor(nn.Module):
    def __init__(
        self, pos_dim: int, valid_dim: int, hidden_size=128, num_layers=2, dropout=0.1
    ):
        super().__init__()
        self.pos_dim = pos_dim
        self.valid_dim = valid_dim

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pos_dim),
        )
        self.valid_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, valid_dim),
        )

    def forward(self, x):
        # x: [B,T,1]
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        pos = self.pos_head(h)
        valid_logits = self.valid_head(h)
        return pos, valid_logits


@torch.no_grad()
def evaluate(model, loader, device, pos_dim: int, cfg: TrainConfig):
    model.eval()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pos = y[:, :pos_dim]
        y_valid = y[:, pos_dim:]

        pred_pos, pred_valid_logits = model(x)
        loss_pos = mse(pred_pos, y_pos)
        loss_valid = bce(pred_valid_logits, y_valid)
        loss = cfg.pos_loss_weight * loss_pos + cfg.valid_loss_weight * loss_valid

        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / n


def train_one_epoch(model, loader, optimizer, device, pos_dim: int, cfg: TrainConfig):
    model.train()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pos = y[:, :pos_dim]
        y_valid = y[:, pos_dim:]

        pred_pos, pred_valid_logits = model(x)
        loss_pos = mse(pred_pos, y_pos)
        loss_valid = bce(pred_valid_logits, y_valid)
        loss = cfg.pos_loss_weight * loss_pos + cfg.valid_loss_weight * loss_valid

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / n


def main():
    ap = argparse.ArgumentParser()

    import datetime

    ap.add_argument(
        "--out_dir",
        default=f"./runs/lstm_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)

    # split strategy:
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--input_col", default="data[1]")

    args: argparse.Namespace = ap.parse_args()

    print(type(args))

    cfg = TrainConfig(
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    df1 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/merged_data.csv"
    )
    df2 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_24_27/merged_data.csv"
    )

    center_cols = ("marker_2_x", "marker_2_y", "marker_2_z")
    out_marker_ids = [3, 4, 5, 6, 7]  # predict others relative to marker_2

    t1, v1, te1 = split_continuous_df(df1, args.train_ratio, args.val_ratio)
    t2, v2, te2 = split_continuous_df(df2, args.train_ratio, args.val_ratio)

    train_df = pd.concat([t1, t2], axis=0, ignore_index=True)
    val_df = pd.concat([v1, v2], axis=0, ignore_index=True)
    test_df = pd.concat([te1, te2], axis=0, ignore_index=True)

    # Fit stats on train only
    train_ds = SlidingWindowDataset(
        train_df,
        cfg,
        args.input_col,
        center_cols,
        out_marker_ids,
        normalize_stats=None,
        fit_stats=True,
    )
    stats = train_ds.stats
    val_ds = SlidingWindowDataset(
        val_df,
        cfg,
        args.input_col,
        center_cols,
        out_marker_ids,
        normalize_stats=stats,
        fit_stats=False,
    )
    test_ds = SlidingWindowDataset(
        test_df,
        cfg,
        args.input_col,
        center_cols,
        out_marker_ids,
        normalize_stats=stats,
        fit_stats=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pos_dim = len(out_marker_ids) * 3
    valid_dim = len(out_marker_ids)

    model = LSTMRegressor(
        pos_dim,
        valid_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float("inf")
    train_losses = []
    val_losses = []

    for ep in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        tr = train_one_epoch(model, train_loader, optimizer, device, pos_dim, cfg)
        va = evaluate(model, val_loader, device, pos_dim, cfg)
        train_losses.append(tr)
        val_losses.append(va)
        print(f"[E{ep:03d}] train={tr:.6f} val={va:.6f}")

        torch.save(
            {"model": model.state_dict(), "stats": stats, "cfg": cfg.__dict__},
            os.path.join(args.out_dir, "last.pt"),
        )

        if va < best_val:
            best_val = va
            torch.save(
                {"model": model.state_dict(), "stats": stats, "cfg": cfg.__dict__},
                os.path.join(args.out_dir, "best.pt"),
            )
            print(f"[INFO] saved best (val={best_val:.6f})")

    test_loss = evaluate(model, test_loader, device, pos_dim, cfg)
    print(f"[TEST] loss={test_loss:.6f}")

    np.savez(
        os.path.join(args.out_dir, "norm_stats.npz"),
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std_pos=stats["y_std_pos"],
    )
    print(f"[INFO] saved stats -> {os.path.join(args.out_dir, 'norm_stats.npz')}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, cfg.epochs + 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="s", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] saved training plot -> {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
