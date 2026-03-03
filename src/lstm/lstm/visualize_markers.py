import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# CSV 파일 읽기
csv_file = "/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/tracked_markers_cleaned.csv"
df = pd.read_csv(csv_file)

print(f"Loaded data from: {csv_file}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 시간 데이터 추출
times = df["time"].to_numpy()

# 마커 정보 추출
marker_columns = [
    col for col in df.columns if col.startswith("marker_") and col.endswith("_x")
]
marker_indices = [int(col.split("_")[1]) for col in marker_columns]
n_markers = len(marker_indices)

print(f"\nNumber of markers: {n_markers}")
print(f"Marker indices: {marker_indices}")

# 각 마커의 데이터를 배열로 저장
marker_data = {}
for marker_idx in marker_indices:
    marker_data[marker_idx] = {
        "x": df[f"marker_{marker_idx}_x"].to_numpy(),
        "y": df[f"marker_{marker_idx}_y"].to_numpy(),
        "z": df[f"marker_{marker_idx}_z"].to_numpy(),
        "valid": df[f"marker_{marker_idx}_valid"].to_numpy().astype(bool),
    }

# 3D 시각화 설정
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.15)

# 슬라이더를 위한 axes
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
time_slider = Slider(ax_slider, "Time Index", 0, len(times) - 1, valinit=0, valstep=1)

# 색상 맵 생성 (각 마커마다 다른 색상)
colors = plt.cm.tab10(np.linspace(0, 1, n_markers))


def update_plot(time_idx):
    """주어진 시간 인덱스에서 마커 위치를 업데이트"""
    ax.clear()

    time_idx = int(time_idx)
    current_time = times[time_idx]

    # 각 마커에 대해
    for idx, marker_idx in enumerate(marker_indices):
        data = marker_data[marker_idx]

        if data["valid"][time_idx]:
            x = data["x"][time_idx]
            y = data["y"][time_idx]
            z = data["z"][time_idx]

            ax.scatter(
                x,
                y,
                z,
                c=[colors[idx]],
                s=100,
                marker="o",
                label=f"Marker {marker_idx}",
                edgecolors="black",
                linewidth=1.5,
            )

    # 축 레이블 및 제목
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Marker Positions at Time Index {time_idx} (t={current_time:.3f}s)")

    # 축 범위 설정 (전체 데이터 기준)
    all_x = []
    all_y = []
    all_z = []

    for marker_idx in marker_indices:
        data = marker_data[marker_idx]
        valid_indices = data["valid"]
        all_x.extend(data["x"][valid_indices])
        all_y.extend(data["y"][valid_indices])
        all_z.extend(data["z"][valid_indices])

    if len(all_x) > 0:
        margin = 0.1
        ax.set_xlim([min(all_x) - margin, max(all_x) + margin])
        ax.set_ylim([min(all_y) - margin, max(all_y) + margin])
        ax.set_zlim([min(all_z) - margin, max(all_z) + margin])

    # 범례
    if n_markers <= 10:
        ax.legend(loc="upper right", fontsize=8)

    fig.canvas.draw_idle()


# 슬라이더 이벤트 연결
time_slider.on_changed(update_plot)

# 초기 플롯
update_plot(0)

plt.show()
