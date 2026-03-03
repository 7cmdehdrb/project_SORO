# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# %%
import re

arduino_file = (
    "/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/arduino_data.csv"
)
marker_file = "/home/min/project_SORO/data/rosbag2_2026_03_03-16_24_27/natnet_client_node_unlabeled.csv"


df: pd.DataFrame = pd.read_csv(marker_file, on_bad_lines="skip")

column_names = df.columns.tolist()
header_indices = sorted(
    {
        int(m.group(1))
        for col in column_names
        if (m := re.match(r"^markers\[(\d+)\]\.header(?:\.|$)", col))
    }
)
i_min, i_max = header_indices[0], header_indices[-1]

# %%
data = df.to_numpy()

times = df["time"].to_numpy()
positions = np.empty(((i_max - i_min + 1) * 3, len(df)), dtype=np.float64)
labels = df[[f"markers[{i}].id" for i in range(i_min, i_max + 1)]].to_numpy()

for i in range(i_min, i_max + 1):
    x = df[f"markers[{i}].pose.position.x"].to_numpy()
    y = df[f"markers[{i}].pose.position.y"].to_numpy()
    z = df[f"markers[{i}].pose.position.z"].to_numpy()

    positions[(3 * i) + 0, :] = x
    positions[(3 * i) + 1, :] = y
    positions[(3 * i) + 2, :] = z

print(labels.shape)
print(np.unique(labels))


# %%
def track_markers_with_id_recovery(positions, labels):
    """
    마커 추적 데이터에서 ID 재할당 문제를 해결하여 완벽하게 추적된 배열을 반환

    Parameters:
    - positions: (N*3, time_steps) 배열 - x1,y1,z1,x2,y2,z2... 형태
    - labels: (time_steps, N) 배열 - 각 마커의 ID

    Returns:
    - tracked_positions: (M*3, time_steps) 배열 - 일관되게 추적된 M개 마커의 위치
    - valid_mask: (M, time_steps) 배열 - 각 마커가 유효한 시점 표시
    """
    n_markers = labels.shape[1]
    n_timesteps = labels.shape[0]

    # 각 타임스텝별로 마커 위치를 추출 (N, 3, time_steps)
    marker_positions = np.zeros((n_markers, 3, n_timesteps))
    for i in range(n_markers):
        marker_positions[i, 0, :] = positions[3 * i + 0, :]
        marker_positions[i, 1, :] = positions[3 * i + 1, :]
        marker_positions[i, 2, :] = positions[3 * i + 2, :]

    # NaN이나 0인 위치를 invalid로 처리
    valid_positions = ~(
        np.isnan(marker_positions).any(axis=1)
        | (np.abs(marker_positions).sum(axis=1) < 1e-6)
    )

    # 논리적 트랙 관리 (각 트랙은 하나의 물리적 마커를 나타냄)
    tracks = (
        []
    )  # 각 트랙: {'positions': (3, time_steps), 'valid': (time_steps,), 'last_valid_pos': [x,y,z], 'last_valid_time': int}

    # 거리 임계값 (같은 마커로 판단할 최대 거리)
    distance_threshold = 0.1  # 10cm, 필요시 조정

    for t in range(n_timesteps):
        current_markers = []

        # 현재 프레임에서 유효한 마커들 수집
        for i in range(n_markers):
            if valid_positions[i, t]:
                pos = marker_positions[i, :, t]
                marker_id = labels[t, i]
                current_markers.append({"index": i, "position": pos, "id": marker_id})

        # 기존 트랙들을 업데이트
        unmatched_markers = list(range(len(current_markers)))
        matched_tracks = set()

        if len(tracks) > 0 and len(current_markers) > 0:
            # 각 활성 트랙에 대해 가장 가까운 마커 찾기
            for track_idx, track in enumerate(tracks):
                if track["last_valid_time"] is None:
                    continue

                # 최근 위치와의 거리 계산
                last_pos = track["last_valid_pos"]
                min_dist = float("inf")
                best_marker_idx = None

                for marker_idx in unmatched_markers:
                    marker = current_markers[marker_idx]
                    dist = np.linalg.norm(marker["position"] - last_pos)

                    if dist < min_dist:
                        min_dist = dist
                        best_marker_idx = marker_idx

                # 거리 임계값 내에 있으면 매칭
                if best_marker_idx is not None and min_dist < distance_threshold:
                    marker = current_markers[best_marker_idx]
                    track["positions"][:, t] = marker["position"]
                    track["valid"][t] = True
                    track["last_valid_pos"] = marker["position"].copy()
                    track["last_valid_time"] = t

                    unmatched_markers.remove(best_marker_idx)
                    matched_tracks.add(track_idx)

        # 매칭되지 않은 마커들은 새로운 트랙 생성
        for marker_idx in unmatched_markers:
            marker = current_markers[marker_idx]
            new_track = {
                "positions": np.zeros((3, n_timesteps)),
                "valid": np.zeros(n_timesteps, dtype=bool),
                "last_valid_pos": marker["position"].copy(),
                "last_valid_time": t,
            }
            new_track["positions"][:, t] = marker["position"]
            new_track["valid"][t] = True
            tracks.append(new_track)

    # 결과 배열 생성
    n_tracks = len(tracks)
    tracked_positions = np.zeros((n_tracks * 3, n_timesteps))
    valid_mask = np.zeros((n_tracks, n_timesteps), dtype=bool)

    for i, track in enumerate(tracks):
        tracked_positions[3 * i : 3 * i + 3, :] = track["positions"]
        valid_mask[i, :] = track["valid"]

    # 너무 짧은 트랙 제거 (전체 시간의 5% 미만)
    min_valid_frames = max(10, int(n_timesteps * 0.05))
    valid_tracks = [
        i for i in range(n_tracks) if valid_mask[i].sum() >= min_valid_frames
    ]

    if len(valid_tracks) < n_tracks:
        print(f"Filtering short tracks: {n_tracks} -> {len(valid_tracks)}")
        tracked_positions = tracked_positions[
            [3 * i + j for i in valid_tracks for j in range(3)], :
        ]
        valid_mask = valid_mask[valid_tracks, :]

    return tracked_positions, valid_mask


# %%
# 완벽하게 추적된 마커 데이터 생성
tracked_positions, valid_mask = track_markers_with_id_recovery(positions, labels)

print(f"Original markers: {positions.shape[0]//3}")
print(f"Tracked markers: {tracked_positions.shape[0]//3}")
print(f"Time steps: {tracked_positions.shape[1]}")
print(f"\nValidity per marker:")
for i in range(tracked_positions.shape[0] // 3):
    valid_ratio = valid_mask[i].sum() / len(valid_mask[i]) * 100
    print(
        f"  Marker {i}: {valid_mask[i].sum()}/{len(valid_mask[i])} frames ({valid_ratio:.1f}%)"
    )

# %%
# 상호작용 가능한 3D 시각화
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.15)

# 슬라이더를 위한 axes
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
time_slider = Slider(ax_slider, "Time Index", 0, len(times) - 1, valinit=0, valstep=1)

# 색상 맵 생성 (각 마커마다 다른 색상)
n_tracked_markers = tracked_positions.shape[0] // 3
colors = plt.cm.tab10(np.linspace(0, 1, n_tracked_markers))

# 초기 scatter plot
scatter_plots = []


def update_plot(time_idx):
    """주어진 시간 인덱스에서 마커 위치를 업데이트"""
    ax.clear()

    time_idx = int(time_idx)
    current_time = times[time_idx]

    # 각 트랙된 마커에 대해
    for marker_idx in range(n_tracked_markers):
        if valid_mask[marker_idx, time_idx]:
            x = tracked_positions[3 * marker_idx + 0, time_idx]
            y = tracked_positions[3 * marker_idx + 1, time_idx]
            z = tracked_positions[3 * marker_idx + 2, time_idx]

            ax.scatter(
                x,
                y,
                z,
                c=[colors[marker_idx]],
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
    all_x = tracked_positions[0::3, :][valid_mask].flatten()
    all_y = tracked_positions[1::3, :][valid_mask].flatten()
    all_z = tracked_positions[2::3, :][valid_mask].flatten()

    if len(all_x) > 0:
        margin = 0.1
        ax.set_xlim([all_x.min() - margin, all_x.max() + margin])
        ax.set_ylim([all_y.min() - margin, all_y.max() + margin])
        ax.set_zlim([all_z.min() - margin, all_z.max() + margin])

    # 범례 (마커가 너무 많으면 생략)
    if n_tracked_markers <= 10:
        ax.legend(loc="upper right", fontsize=8)

    fig.canvas.draw_idle()


# 슬라이더 이벤트 연결
time_slider.on_changed(update_plot)

# 초기 플롯
update_plot(0)

plt.show()

# %%
# 아웃라이어 제거 및 CSV 저장
outliers = [0, 1, 8, 9]  # 제거할 마커 인덱스
keep_markers = [i for i in range(n_tracked_markers) if i not in outliers]

print(f"\nRemoving outliers: {outliers}")
print(f"Keeping markers: {keep_markers}")

# 아웃라이어를 제거한 데이터 생성
filtered_positions = tracked_positions[
    [3 * i + j for i in keep_markers for j in range(3)], :
]
filtered_valid_mask = valid_mask[keep_markers, :]

# CSV로 저장할 데이터프레임 생성
csv_data = {"time": times}

for idx, marker_idx in enumerate(keep_markers):
    # 각 마커의 x, y, z 좌표 추가
    csv_data[f"marker_{marker_idx}_x"] = filtered_positions[3 * idx + 0, :]
    csv_data[f"marker_{marker_idx}_y"] = filtered_positions[3 * idx + 1, :]
    csv_data[f"marker_{marker_idx}_z"] = filtered_positions[3 * idx + 2, :]
    csv_data[f"marker_{marker_idx}_valid"] = filtered_valid_mask[idx, :].astype(int)

df_output = pd.DataFrame(csv_data)

# CSV 파일로 저장
output_file = "/home/min/project_SORO/data/tracked_markers_cleaned.csv"
df_output.to_csv(output_file, index=False)

print(f"\nCleaned data saved to: {output_file}")
print(f"Shape: {df_output.shape}")
print(f"Columns: {list(df_output.columns)}")
