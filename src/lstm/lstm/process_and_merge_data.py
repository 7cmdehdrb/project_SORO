#!/usr/bin/env python3
"""
마커 데이터 처리 및 Arduino 데이터와 병합하는 통합 스크립트
- 마커 추적 (ID 재할당 문제 해결)
- 자동 아웃라이어 검출 (공간 클러스터링 기반)
- Arduino 데이터와 시간 기준 병합
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


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
    tracks = []

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
        print(f"짧은 트랙 필터링: {n_tracks} -> {len(valid_tracks)}")
        tracked_positions = tracked_positions[
            [3 * i + j for i in valid_tracks for j in range(3)], :
        ]
        valid_mask = valid_mask[valid_tracks, :]

    return tracked_positions, valid_mask


def detect_outliers_by_clustering(tracked_positions, valid_mask, n_expected=6):
    """
    공간 클러스터링을 사용하여 아웃라이어 자동 검출

    Parameters:
    - tracked_positions: (M*3, time_steps) 배열
    - valid_mask: (M, time_steps) 배열
    - n_expected: 예상되는 정상 마커 수 (기본값: 6)

    Returns:
    - keep_markers: 유지할 마커 인덱스 리스트
    - outlier_markers: 제거할 마커 인덱스 리스트
    """
    n_markers = tracked_positions.shape[0] // 3

    # 각 마커의 평균 위치 계산 (유효한 프레임만 사용)
    centroids = []
    marker_indices = []

    for i in range(n_markers):
        valid_frames = valid_mask[i, :]
        if valid_frames.sum() == 0:
            continue

        x = tracked_positions[3 * i + 0, valid_frames].mean()
        y = tracked_positions[3 * i + 1, valid_frames].mean()
        z = tracked_positions[3 * i + 2, valid_frames].mean()

        centroids.append([x, y, z])
        marker_indices.append(i)

    centroids = np.array(centroids)
    print(f"\n마커 중심 위치 계산 완료: {len(centroids)}개")

    # DBSCAN을 사용하여 클러스터링
    # eps는 거리 임계값, min_samples는 클러스터를 형성하는 최소 포인트 수
    clustering = DBSCAN(eps=0.3, min_samples=min(3, n_expected // 2)).fit(centroids)
    labels = clustering.labels_

    # 각 클러스터의 크기 확인
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    if len(unique_labels) == 0:
        print("경고: 클러스터를 찾을 수 없습니다. 모든 마커 유지")
        return list(range(n_markers)), []

    print(f"클러스터 검출: {len(unique_labels)}개")
    for label, count in zip(unique_labels, counts):
        print(f"  클러스터 {label}: {count}개 마커")

    # 가장 큰 클러스터를 정상 마커로 간주
    main_cluster_label = unique_labels[np.argmax(counts)]
    main_cluster_size = counts[np.argmax(counts)]

    print(f"\n메인 클러스터 (라벨 {main_cluster_label}): {main_cluster_size}개 마커")

    # 메인 클러스터에 속한 마커와 아웃라이어 분리
    keep_markers = [
        marker_indices[i]
        for i, label in enumerate(labels)
        if label == main_cluster_label
    ]
    outlier_markers = [
        marker_indices[i]
        for i, label in enumerate(labels)
        if label != main_cluster_label
    ]

    print(f"유지할 마커: {keep_markers}")
    print(f"제거할 마커 (아웃라이어): {outlier_markers}")

    # 예상 개수와 다르면 경고
    if len(keep_markers) != n_expected:
        print(
            f"경고: 예상 마커 수({n_expected})와 검출된 마커 수({len(keep_markers)})가 다릅니다."
        )

    return keep_markers, outlier_markers


def load_and_process_markers(marker_file, n_expected_markers=6):
    """
    마커 CSV 로드 및 처리

    Parameters:
    - marker_file: 마커 CSV 파일 경로
    - n_expected_markers: 예상되는 정상 마커 수

    Returns:
    - times: 시간 배열
    - filtered_positions: 아웃라이어 제거된 마커 위치 (M*3, time_steps)
    - filtered_valid_mask: 유효성 마스크 (M, time_steps)
    """
    print(f"\n마커 파일 로드 중: {marker_file}")
    df = pd.read_csv(marker_file, on_bad_lines="skip")

    # 마커 인덱스 추출
    column_names = df.columns.tolist()
    header_indices = sorted(
        {
            int(m.group(1))
            for col in column_names
            if (m := re.match(r"^markers\[(\d+)\]\.header(?:\.|$)", col))
        }
    )
    i_min, i_max = header_indices[0], header_indices[-1]

    # 데이터 추출
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

    print(f"원본 마커 수: {positions.shape[0]//3}")
    print(f"시간 스텝 수: {positions.shape[1]}")

    # 마커 추적
    print("\n마커 추적 시작...")
    tracked_positions, valid_mask = track_markers_with_id_recovery(positions, labels)

    print(f"추적된 마커 수: {tracked_positions.shape[0]//3}")
    print(f"\n마커별 유효성:")
    for i in range(tracked_positions.shape[0] // 3):
        valid_ratio = valid_mask[i].sum() / len(valid_mask[i]) * 100
        print(
            f"  마커 {i}: {valid_mask[i].sum()}/{len(valid_mask[i])} 프레임 ({valid_ratio:.1f}%)"
        )

    # 자동 아웃라이어 검출
    print("\n아웃라이어 자동 검출 시작...")
    keep_markers, outlier_markers = detect_outliers_by_clustering(
        tracked_positions, valid_mask, n_expected=n_expected_markers
    )

    # n_expected_markers보다 많은 경우, invalid 비율이 높은 순으로 제거
    if len(keep_markers) > n_expected_markers:
        print(
            f"\n검출된 마커 수({len(keep_markers)})가 예상 수({n_expected_markers})보다 많습니다."
        )
        print("Invalid 비율이 높은 마커부터 제거합니다...")

        # 각 마커의 invalid 비율 계산
        marker_invalid_ratios = []
        for marker_idx in keep_markers:
            total_frames = len(valid_mask[marker_idx])
            valid_frames = valid_mask[marker_idx].sum()
            invalid_ratio = 1.0 - (valid_frames / total_frames)
            marker_invalid_ratios.append(
                (marker_idx, invalid_ratio, valid_frames, total_frames)
            )
            print(
                f"  마커 {marker_idx}: {total_frames - valid_frames}/{total_frames} invalid ({invalid_ratio*100:.1f}%)"
            )

        # invalid 비율로 정렬 (낮은 것이 좋은 것)
        marker_invalid_ratios.sort(key=lambda x: x[1])

        # 상위 n_expected_markers개만 유지
        keep_markers = [
            idx for idx, _, _, _ in marker_invalid_ratios[:n_expected_markers]
        ]
        removed_markers = [
            (idx, ratio)
            for idx, ratio, _, _ in marker_invalid_ratios[n_expected_markers:]
        ]

        print(f"\n최종 유지할 마커: {sorted(keep_markers)}")
        print(f"제거된 마커 (invalid 비율 높음): {[idx for idx, _ in removed_markers]}")

    # 아웃라이어 제거
    filtered_positions = tracked_positions[
        [3 * i + j for i in keep_markers for j in range(3)], :
    ]
    filtered_valid_mask = valid_mask[keep_markers, :]

    print(f"\n최종 마커 수: {len(keep_markers)} (예상: {n_expected_markers})")

    # 마커 인덱스 재배치: 0번은 움직임이 가장 적은 원점, 그 다음은 체인 형태로 가까운 순서
    print("\n마커 인덱스 재배치 시작...")

    # 각 마커의 움직임 계산 (유효한 프레임에서의 위치 표준편차)
    n_final_markers = len(keep_markers)
    movements = []
    for i in range(n_final_markers):
        valid_frames = filtered_valid_mask[i, :]
        if valid_frames.sum() == 0:
            movements.append(float("inf"))
            continue

        x = filtered_positions[3 * i + 0, valid_frames]
        y = filtered_positions[3 * i + 1, valid_frames]
        z = filtered_positions[3 * i + 2, valid_frames]

        # 위치 변화량의 표준편차 합계 (움직임의 지표)
        movement = np.std(x) + np.std(y) + np.std(z)
        movements.append(movement)

    # 0번: 가장 움직임이 적은 마커 (원점)
    origin_idx = np.argmin(movements)
    print(
        f"원점 마커 (0번): 기존 인덱스 {origin_idx}, 움직임: {movements[origin_idx]:.6f}"
    )

    # 체인 형태로 인덱스 재배치
    new_order = [origin_idx]
    remaining = [i for i in range(n_final_markers) if i != origin_idx]

    while remaining:
        # 마지막으로 추가된 마커의 평균 위치
        last_idx = new_order[-1]
        last_valid = filtered_valid_mask[last_idx, :]
        last_pos = np.array(
            [
                filtered_positions[3 * last_idx + 0, last_valid].mean(),
                filtered_positions[3 * last_idx + 1, last_valid].mean(),
                filtered_positions[3 * last_idx + 2, last_valid].mean(),
            ]
        )

        # 남은 마커들 중 가장 가까운 마커 찾기
        min_dist = float("inf")
        closest_idx = None

        for idx in remaining:
            valid_frames = filtered_valid_mask[idx, :]
            if valid_frames.sum() == 0:
                continue

            pos = np.array(
                [
                    filtered_positions[3 * idx + 0, valid_frames].mean(),
                    filtered_positions[3 * idx + 1, valid_frames].mean(),
                    filtered_positions[3 * idx + 2, valid_frames].mean(),
                ]
            )

            dist = np.linalg.norm(pos - last_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        if closest_idx is not None:
            new_order.append(closest_idx)
            remaining.remove(closest_idx)
            print(
                f"마커 {len(new_order)-1}번: 기존 인덱스 {closest_idx}, 거리: {min_dist:.6f}m"
            )
        else:
            # 남은 마커가 있지만 유효한 프레임이 없는 경우
            print(
                f"경고: 남은 마커 {remaining}는 유효한 프레임이 없어 마지막에 추가됩니다."
            )
            new_order.extend(remaining)
            break

    # 새로운 순서로 재배치
    filtered_positions = filtered_positions[
        [3 * i + j for i in new_order for j in range(3)], :
    ]
    filtered_valid_mask = filtered_valid_mask[new_order, :]

    print(f"\n마커 인덱스 재배치 완료")
    print(f"재배치 순서 (기존 인덱스): {new_order}")

    return times, filtered_positions, filtered_valid_mask


def process_and_merge_data(
    marker_file, arduino_file, output_file, tolerance=0.1, n_expected_markers=6
):
    """
    마커 데이터 처리 및 Arduino 데이터와 병합

    Parameters:
    - marker_file: 마커 CSV 파일 경로
    - arduino_file: Arduino CSV 파일 경로
    - output_file: 출력 CSV 파일 경로
    - tolerance: 시간 매칭 허용 오차 (초)
    - n_expected_markers: 예상되는 정상 마커 수

    Returns:
    - merged_df: 병합된 데이터프레임
    """
    # 1. 마커 데이터 처리
    times, filtered_positions, filtered_valid_mask = load_and_process_markers(
        marker_file, n_expected_markers
    )

    # 마커 데이터프레임 생성
    n_markers = filtered_positions.shape[0] // 3
    marker_data = {"time": times}

    for i in range(n_markers):
        # 0부터 시작하는 인덱스로 컬럼 이름 지정
        marker_data[f"marker_{i}_x"] = filtered_positions[3 * i + 0, :]
        marker_data[f"marker_{i}_y"] = filtered_positions[3 * i + 1, :]
        marker_data[f"marker_{i}_z"] = filtered_positions[3 * i + 2, :]
        marker_data[f"marker_{i}_valid"] = filtered_valid_mask[i, :].astype(int)

    markers_df = pd.DataFrame(marker_data)
    print(f"\n마커 데이터프레임 생성: {markers_df.shape}")

    # 2. Arduino 데이터 로드
    print(f"\nArduino 파일 로드 중: {arduino_file}")
    arduino_df = pd.read_csv(arduino_file)
    print(f"Arduino 데이터 shape: {arduino_df.shape}")

    # Arduino 데이터 컬럼 확인 및 표준화
    print(f"Arduino 원본 컬럼: {arduino_df.columns.tolist()}")

    # 컬럼 이름 매핑 (다양한 형식 지원)
    column_mapping = {}
    required_columns = [
        "time",
        "target_pressure",
        "current_pressure",
        "filtered_pressure",
        "valve",
    ]

    for col in arduino_df.columns:
        col_lower = col.lower()
        if "time" in col_lower and "time" not in column_mapping:
            column_mapping[col] = "time"
        elif "target" in col_lower and "pressure" in col_lower:
            column_mapping[col] = "target_pressure"
        elif "current" in col_lower and "pressure" in col_lower:
            column_mapping[col] = "current_pressure"
        elif "filtered" in col_lower and "pressure" in col_lower:
            column_mapping[col] = "filtered_pressure"
        elif "valve" in col_lower:
            column_mapping[col] = "valve"
        # data[0], data[1] 형식 지원
        elif col == "data[0]":
            column_mapping[col] = "target_pressure"
        elif col == "data[1]":
            column_mapping[col] = "current_pressure"
        elif col == "data[2]":
            column_mapping[col] = "filtered_pressure"
        elif col == "data[3]":
            column_mapping[col] = "valve"

    # 컬럼 이름 변경
    if column_mapping:
        arduino_df = arduino_df.rename(columns=column_mapping)
        print(f"Arduino 컬럼 매핑: {column_mapping}")

    # 필요한 컬럼만 선택
    available_columns = ["time"] + [
        col for col in required_columns[1:] if col in arduino_df.columns
    ]
    arduino_df = arduino_df[available_columns]
    print(f"Arduino 최종 컬럼: {arduino_df.columns.tolist()}")

    # 3. 시간 기준으로 정렬
    arduino_df = arduino_df.sort_values("time").reset_index(drop=True)
    markers_df = markers_df.sort_values("time").reset_index(drop=True)

    # 4. 시간 기준 병합
    print("\n시간 기준으로 병합 중...")
    merged_df = pd.merge_asof(
        arduino_df, markers_df, on="time", direction="nearest", tolerance=tolerance
    )

    print(f"병합 후 shape: {merged_df.shape}")

    # 5. NaN 제거
    print("\nNaN이 있는 행 제거 중...")
    before_dropna = len(merged_df)
    merged_df = merged_df.dropna()
    after_dropna = len(merged_df)

    print(f"제거된 행 수: {before_dropna - after_dropna}")
    print(f"최종 데이터 shape: {merged_df.shape}")

    # 6. 저장
    merged_df.to_csv(output_file, index=False)
    print(f"\n병합된 데이터 저장 완료: {output_file}")

    return merged_df


if __name__ == "__main__":
    # 파일 경로 설정
    base_path = Path("/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02")
    marker_file = base_path / "natnet_client_node_unlabeled.csv"
    arduino_file = base_path / "arduino_data.csv"
    output_file = base_path / "new_merged_data.csv"

    # 처리 및 병합 실행
    merged_data = process_and_merge_data(
        marker_file=marker_file,
        arduino_file=arduino_file,
        output_file=output_file,
        tolerance=0.05,  # 0.05초 이내의 시간만 매칭
        n_expected_markers=6,  # 예상되는 정상 마커 수
    )

    # 결과 미리보기
    print("\n=== 병합 결과 미리보기 ===")
    print(merged_data.head(10))

    print("\n=== 최종 컬럼 목록 ===")
    print(merged_data.columns.tolist())

    print("\n=== 데이터 통계 ===")
    print(merged_data.describe())
