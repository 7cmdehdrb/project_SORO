#!/usr/bin/env python3
"""
CSV 파일을 시간 기준으로 병합하는 스크립트
시간이 완벽하게 일치하지 않을 경우 가장 가까운 시간을 매칭
NaN이 있는 행은 제거
"""

import pandas as pd
import numpy as np
from pathlib import Path


def merge_csv_by_time(
    arduino_csv_path, markers_csv_path, output_path=None, tolerance=None
):
    """
    두 CSV 파일을 시간 기준으로 병합

    Parameters:
    -----------
    arduino_csv_path : str
        Arduino 데이터 CSV 파일 경로
    markers_csv_path : str
        마커 데이터 CSV 파일 경로
    output_path : str, optional
        출력 CSV 파일 경로 (기본값: merged_data.csv)
    tolerance : float, optional
        시간 매칭 허용 오차 (초 단위)

    Returns:
    --------
    pandas.DataFrame
        병합된 데이터프레임
    """
    # CSV 파일 읽기
    print(f"읽는 중: {arduino_csv_path}")
    arduino_df = pd.read_csv(arduino_csv_path)

    print(f"읽는 중: {markers_csv_path}")
    markers_df = pd.read_csv(markers_csv_path)

    print(f"\nArduino 데이터 shape: {arduino_df.shape}")
    print(f"Markers 데이터 shape: {markers_df.shape}")

    # 시간 기준으로 정렬
    arduino_df = arduino_df.sort_values("time").reset_index(drop=True)
    markers_df = markers_df.sort_values("time").reset_index(drop=True)

    # merge_asof를 사용하여 가장 가까운 시간 매칭
    # direction='nearest'로 가장 가까운 시간을 찾음
    print("\n시간 기준으로 병합 중...")
    if tolerance is not None:
        merged_df = pd.merge_asof(
            arduino_df,
            markers_df,
            on="time",
            direction="nearest",
            tolerance=tolerance,
            suffixes=("_arduino", "_marker"),
        )
    else:
        merged_df = pd.merge_asof(
            arduino_df,
            markers_df,
            on="time",
            direction="nearest",
            suffixes=("_arduino", "_marker"),
        )

    print(f"병합 후 shape: {merged_df.shape}")

    # NaN이 있는 행 제거
    print("\nNaN이 있는 행 제거 중...")
    before_dropna = len(merged_df)
    merged_df = merged_df.dropna()
    after_dropna = len(merged_df)

    print(f"제거된 행 수: {before_dropna - after_dropna}")
    print(f"최종 데이터 shape: {merged_df.shape}")

    # 결과 저장
    if output_path is None:
        output_path = "merged_data.csv"

    merged_df.to_csv(output_path, index=False)
    print(f"\n병합된 데이터 저장 완료: {output_path}")

    return merged_df


if __name__ == "__main__":
    # 파일 경로 설정
    base_path = Path("/home/min/project_SORO/data/rosbag2_2026_03_03-16_24_27")
    arduino_csv = base_path / "arduino_data.csv"
    markers_csv = base_path / "tracked_markers_cleaned.csv"
    output_csv = base_path / "merged_data.csv"

    # 병합 실행
    merged_data = merge_csv_by_time(
        arduino_csv_path=arduino_csv,
        markers_csv_path=markers_csv,
        output_path=output_csv,
        tolerance=0.1,  # 0.1초 이내의 시간만 매칭 (필요시 조정 가능)
    )

    # 결과 미리보기
    print("\n=== 병합 결과 미리보기 ===")
    print(merged_data.head(10))

    print("\n=== 컬럼 목록 ===")
    print(merged_data.columns.tolist())
