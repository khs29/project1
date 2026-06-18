"""실행 진입점.

- 1회 평가를 실행해 MAE/리포트를 출력
- 캐시를 비우고 N회 반복 실행해 전체 평가 시간의 평균/표준편차를 측정
  (보고서 6장: "평균 실행 시간", "표준편차" 요구사항 대응)
- per-joint MAE와 측정 환경을 results/ 에 CSV로 저장
"""
from __future__ import annotations
import platform
import sys
import time
from pathlib import Path

import numpy as np

from config import EvalConfig
from pose_estimator import RTMPoseEstimator
from evaluator import Evaluator


def record_environment() -> dict:
    """벤치마크 6장 요구사항: OS/Python/라이브러리 버전 등 측정 환경 기록."""
    import torch  # noqa: WPS433 (의도적 지연 임포트, torch 없는 환경 대비)

    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


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
    evaluator = Evaluator(config, estimator)

    result = evaluator.run()
    print(result.report())

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    result.mae_per_joint().to_csv(out_dir / "mae_per_joint.csv", header=["MAE_deg"])

    # 반복 실행 -> 전체 평가 시간 평균/표준편차 (캐시 초기화해서 매번 실제 추론하도록)
    repeat_times = []
    for _ in range(config.n_repeats):
        evaluator._kp_cache.clear()
        evaluator.timings.clear()
        t0 = time.perf_counter()
        evaluator.run()
        repeat_times.append(time.perf_counter() - t0)

    repeat_times = np.array(repeat_times)
    print(
        f"\n전체 평가 {config.n_repeats}회 반복: "
        f"{repeat_times.mean():.2f}s ± {repeat_times.std():.2f}s"
    )

    env = record_environment()
    with open(out_dir / "environment.txt", "w") as f:
        for k, v in env.items():
            f.write(f"{k}: {v}\n")
        f.write(f"n_frames: {len(result.frame_ids)}\n")
        f.write(f"n_repeats: {config.n_repeats}\n")
        f.write(f"repeat_mean_s: {repeat_times.mean():.4f}\n")
        f.write(f"repeat_std_s: {repeat_times.std():.4f}\n")


if __name__ == "__main__":
    main()
