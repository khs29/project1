"""입력 크기(프레임 수)별 순수 추론 시간 변화를 측정한다.

이 레코딩에서는 Manus GT가 매칭되는 프레임이 일부뿐이라(폴더의 프레임 수와
CSV 행 수가 일대일로 대응하지 않음), MAE 평가용 run()으로 입력 크기를
늘려도 실제 추론 횟수가 늘지 않는다. 따라서 "성능(추론 시간) 측정"과
"정확도(MAE) 평가"를 분리한다: MAE는 GT가 있는 프레임만으로 한 번 계산하고,
입력 크기별 시간 비교는 GT 매칭과 무관하게 처음 N장에 대해 순수 추론만
반복 실행해서 측정한다.
"""
from __future__ import annotations
from pathlib import Path
import time
import csv

import cv2
import numpy as np

from config import EvalConfig
from pose_estimator import RTMPoseEstimator
from evaluator import Evaluator


def benchmark_raw_inference(estimator, img_files, n, repeats=3):
    """img_files 처음 n장에 대해 추론만 repeats회 반복, 매 회 총 소요시간(초) 배열 반환."""
    subset = img_files[:n]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for img_path in subset:
            img = cv2.imread(str(img_path))
            estimator.predict(img)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def main() -> None:
    config = EvalConfig(
        img_dir=Path("/home/abc/hamer/inpainting_glove_frames2"),
        csv_path=Path("/home/abc/hamer/Untitled_2026-05-12_15-36-07_1_R.csv"),
        rtmpose_config="/home/abc/mmpose/checkpoints/rtmpose-m_8xb256-210e_hand5-256x256.py",
        rtmpose_checkpoint=(
            "/home/abc/mmpose/checkpoints/"
            "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
        ),
    )

    estimator = RTMPoseEstimator(config.rtmpose_config, config.rtmpose_checkpoint, config.device)

    img_files = sorted(config.img_dir.glob("*.png"))
    total_frames = len(img_files)
    print(f"전체 프레임 수: {total_frames}")

    # 정확도(MAE)는 GT가 매칭되는 프레임만으로 한 번 계산 (입력 크기 변수와 분리)
    evaluator = Evaluator(config, estimator)
    mae_result = evaluator.run()
    print(
        f"GT 매칭 프레임: {len(mae_result.frame_ids)}장, "
        f"전체 평균 MAE: {mae_result.mae_total():.2f}°\n"
    )

    sizes = sorted({max(1, total_frames // 4), max(1, total_frames // 2), total_frames})

    # 워밍업: cudnn autotune 등 첫 추론 비용을 본 측정에서 제외
    benchmark_raw_inference(estimator, img_files, min(2, total_frames), repeats=1)

    rows = []
    for n in sizes:
        times = benchmark_raw_inference(estimator, img_files, n, repeats=3)
        per_frame_ms = (times / n) * 1000
        print(
            f"n={n:>4} frames | total {times.mean():.3f}s ± {times.std():.3f}s | "
            f"per-frame {per_frame_ms.mean():.1f}ms ± {per_frame_ms.std():.1f}ms"
        )
        rows.append({
            "n_frames": n,
            "total_mean_sec": times.mean(),
            "total_std_sec": times.std(),
            "per_frame_mean_ms": per_frame_ms.mean(),
            "per_frame_std_ms": per_frame_ms.std(),
        })

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "benchmark_by_input_size.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n결과 저장: {out_dir / 'benchmark_by_input_size.csv'}")


if __name__ == "__main__":
    main()
