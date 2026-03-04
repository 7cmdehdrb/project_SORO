import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import os


# LSTM model definition (must match training)
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


# Configuration
MODEL_PATH = "/home/min/project_SORO/runs/lstm_20260304-164414/best.pt"
CSV_FILE = "/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv"
INPUT_COL = "current_pressure"
CENTER_MARKER = 0
OUT_MARKER_IDS = [1, 2, 3, 4, 5]
NOISE_LEVEL = 0.01  # Standard deviation of noise to add to input

# Load model and stats
print(f"Loading model from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
stats = checkpoint["stats"]
cfg_dict = checkpoint["cfg"]

# Reconstruct model
pos_dim = len(OUT_MARKER_IDS) * 3
valid_dim = len(OUT_MARKER_IDS)

model = LSTMRegressor(
    pos_dim,
    valid_dim,
    hidden_size=cfg_dict["hidden_size"],
    num_layers=cfg_dict["num_layers"],
    dropout=cfg_dict["dropout"],
)
model.load_state_dict(checkpoint["model"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded successfully on {device}")
print(f"Window size: {cfg_dict['window']}")
print(f"Horizon: {cfg_dict['horizon']}")

# Load data
print(f"\nLoading data from: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)
print(f"Data shape: {df.shape}")

# Extract window size
window = cfg_dict["window"]
horizon = cfg_dict["horizon"]

# Maximum valid time index (we need window + horizon entries)
max_time_idx = len(df) - window - horizon

print(f"Valid time range: 0 to {max_time_idx}")

# Calculate fixed axis limits from all data
all_marker_ids = [CENTER_MARKER] + OUT_MARKER_IDS
all_x, all_y, all_z = [], [], []
for marker_id in all_marker_ids:
    valid_mask = df[f"marker_{marker_id}_valid"] > 0
    all_x.extend(df.loc[valid_mask, f"marker_{marker_id}_x"].tolist())
    all_y.extend(df.loc[valid_mask, f"marker_{marker_id}_y"].tolist())
    all_z.extend(df.loc[valid_mask, f"marker_{marker_id}_z"].tolist())

margin = 0.15
axis_limits = {
    "x": [min(all_x) - margin, max(all_x) + margin],
    "y": [min(all_y) - margin, max(all_y) + margin],
    "z": [min(all_z) - margin, max(all_z) + margin],
}
print(
    f"Fixed axis limits: X={axis_limits['x']}, Y={axis_limits['y']}, Z={axis_limits['z']}"
)

# Setup figure
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.15, right=0.85)

# Add slider
ax_slider = plt.axes([0.15, 0.05, 0.6, 0.03])
time_slider = Slider(
    ax_slider, "Time Index", 0, max_time_idx, valinit=max_time_idx // 2, valstep=1
)

# Color for ground truth and prediction
color_gt = plt.cm.Set1(0)  # Red for ground truth
color_pred = plt.cm.Set1(1)  # Blue for prediction


def predict_at_time(time_idx):
    """Run prediction at given time index"""
    time_idx = int(time_idx)

    # Extract input window
    input_data = (
        df[INPUT_COL].iloc[time_idx : time_idx + window].to_numpy().astype(np.float32)
    )

    # Add noise to input
    noisy_input = input_data + np.random.normal(
        0, NOISE_LEVEL, input_data.shape
    ).astype(np.float32)

    # Normalize
    x_norm = (noisy_input - stats["x_mean"]) / stats["x_std"]
    x_tensor = (
        torch.from_numpy(x_norm[:, None]).unsqueeze(0).float().to(device)
    )  # [1, window, 1]

    # Predict
    with torch.no_grad():
        pred_pos, pred_valid_logits = model(x_tensor)

    # Denormalize position prediction
    pred_pos = pred_pos.cpu().numpy().squeeze(0)  # [15]
    pred_pos = pred_pos * stats["y_std_pos"] + stats["y_mean"]
    pred_pos = pred_pos.reshape(len(OUT_MARKER_IDS), 3)  # [5, 3]

    # Get valid predictions
    pred_valid = torch.sigmoid(pred_valid_logits).cpu().numpy().squeeze(0)  # [5]

    # Target time for ground truth
    target_idx = time_idx + window - 1 + horizon

    # Get center marker position (marker_2)
    center_pos = (
        df[
            [
                f"marker_{CENTER_MARKER}_x",
                f"marker_{CENTER_MARKER}_y",
                f"marker_{CENTER_MARKER}_z",
            ]
        ]
        .iloc[target_idx]
        .to_numpy()
        .astype(np.float32)
    )

    # Convert relative predictions to absolute positions
    pred_abs = pred_pos + center_pos[None, :]  # [5, 3]

    # Get ground truth absolute positions
    gt_abs = []
    gt_valid = []
    for marker_id in OUT_MARKER_IDS:
        gt_abs.append(
            [
                df[f"marker_{marker_id}_x"].iloc[target_idx],
                df[f"marker_{marker_id}_y"].iloc[target_idx],
                df[f"marker_{marker_id}_z"].iloc[target_idx],
            ]
        )
        gt_valid.append(df[f"marker_{marker_id}_valid"].iloc[target_idx])

    gt_abs = np.array(gt_abs, dtype=np.float32)  # [5, 3]
    gt_valid = np.array(gt_valid, dtype=bool)

    return {
        "pred_abs": pred_abs,
        "pred_valid": pred_valid,
        "gt_abs": gt_abs,
        "gt_valid": gt_valid,
        "center_pos": center_pos,
        "target_idx": target_idx,
    }


def update_plot(time_idx):
    """Update plot with predictions at given time index"""
    ax.clear()

    time_idx = int(time_idx)
    result = predict_at_time(time_idx)

    # Plot center marker (marker_2)
    ax.scatter(
        result["center_pos"][0],
        result["center_pos"][1],
        result["center_pos"][2],
        c="green",
        s=200,
        marker="*",
        label=f"Center (Marker {CENTER_MARKER})",
        edgecolors="black",
        linewidth=2,
    )

    # Plot ground truth and predictions
    for i, marker_id in enumerate(OUT_MARKER_IDS):
        # Ground truth
        if result["gt_valid"][i]:
            ax.scatter(
                result["gt_abs"][i, 0],
                result["gt_abs"][i, 1],
                result["gt_abs"][i, 2],
                c=[color_gt],
                s=150,
                marker="o",
                label=f"GT Marker {marker_id}" if i == 0 else "",
                edgecolors="black",
                linewidth=1.5,
                alpha=0.7,
            )

        # Prediction (always shown if confidence > 0.3)
        if result["pred_valid"][i] > 0.3:
            ax.scatter(
                result["pred_abs"][i, 0],
                result["pred_abs"][i, 1],
                result["pred_abs"][i, 2],
                c=[color_pred],
                s=150,
                marker="^",
                label=f"Pred Marker {marker_id}" if i == 0 else "",
                edgecolors="black",
                linewidth=1.5,
                alpha=0.7,
            )

            # Draw line connecting GT and prediction if both valid
            if result["gt_valid"][i]:
                ax.plot(
                    [result["gt_abs"][i, 0], result["pred_abs"][i, 0]],
                    [result["gt_abs"][i, 1], result["pred_abs"][i, 1]],
                    [result["gt_abs"][i, 2], result["pred_abs"][i, 2]],
                    "k--",
                    alpha=0.3,
                    linewidth=1,
                )

    # Calculate and display error statistics
    valid_mask = result["gt_valid"] & (result["pred_valid"] > 0.3)
    if valid_mask.any():
        errors = np.linalg.norm(
            result["gt_abs"][valid_mask] - result["pred_abs"][valid_mask], axis=1
        )
        mean_error = errors.mean()
        max_error = errors.max()
        error_text = f"Mean Error: {mean_error:.4f}m\nMax Error: {max_error:.4f}m"
    else:
        error_text = "No valid predictions"

    # Add error text to plot
    ax.text2D(
        0.02,
        0.98,
        error_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Set labels and title
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_zlabel("Z (m)", fontsize=11)
    ax.set_title(
        f"LSTM Prediction vs Ground Truth\n"
        f'Time Index: {time_idx} → Target: {result["target_idx"]} '
        f"(Window: {window}, Horizon: {horizon})",
        fontsize=12,
        fontweight="bold",
    )

    # Set fixed axis limits
    ax.set_xlim(axis_limits["x"])
    ax.set_ylim(axis_limits["y"])
    ax.set_zlim(axis_limits["z"])

    # Legend
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    fig.canvas.draw_idle()


# Connect slider to update function
time_slider.on_changed(update_plot)

# Initial plot
print("\n=== Starting interactive visualization ===")
print("Use the slider to change the time index")
print("- Green star: Center marker (Marker 2)")
print("- Red circles: Ground truth")
print("- Blue triangles: LSTM predictions")
print("- Dashed lines: Connection between GT and prediction\n")

update_plot(max_time_idx // 2)

plt.show()
