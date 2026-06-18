"""Evaluator: PoseEstimator를 프레임 디렉토리에 대해 실행하고 Manus GT와
비교해 MAE를 계산/리포트한다.

원본 스크립트는 모델 로딩, GT 로딩, 각도 변환, 추론 루프, MAE 계산, 출력이
하나의 평평한 스크립트에 전부 섞여 있어서 재사용도, 모델 교체도, 단위
테스트도 불가능했다. 여기서는 그 역할들을 분리해서, Evaluator는 어떤
PoseEstimator를 넣어도 그대로 동작한다.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import EvalConfig
from pose_estimator import PoseEstimator
from angle_calc import keypoints_to_flex_angles, FINGER_JOINT_TRIPLES
from decorators import timed, cached_by_arg


class Evaluator:
    def __init__(self, config: EvalConfig, estimator: PoseEstimator):
        self.config = config
        self.estimator = estimator
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._kp_cache: dict[Path, "np.ndarray | None"] = {}
        self._gt_df = pd.read_csv(config.csv_path)

    def _ground_truth(self, frame_idx: int) -> "np.ndarray | None":
        row_idx = (frame_idx - 1) * self.config.csv_fps_multiplier
        if row_idx >= len(self._gt_df):
            return None
        row = self._gt_df.iloc[row_idx]
        return row[list(self.config.gt_cols)].to_numpy(dtype=float)

    @cached_by_arg("_kp_cache")
    @timed("inference")
    def _predict(self, img_path: Path) -> "np.ndarray | None":
        img = cv2.imread(str(img_path))
        return self.estimator.predict(img)

    def run(self, max_frames: "int | None" = None) -> "EvalResult":
        """max_frames를 지정하면 앞에서부터 N장만 평가한다.
        입력 크기별 실행 시간 변화를 측정할 때 사용 (보고서 6장)."""
        per_col_pred: dict[str, list[float]] = defaultdict(list)
        per_col_gt: dict[str, list[float]] = defaultdict(list)
        frame_ids: list[int] = []

        img_files = sorted(self.config.img_dir.glob("*.png"))
        if max_frames is not None:
            img_files = img_files[:max_frames]
        for img_path in img_files:
            frame_idx = int(img_path.stem)

            gt = self._ground_truth(frame_idx)
            if gt is None:
                continue

            kps = self._predict(img_path)
            if kps is None:
                continue

            pred_by_finger = keypoints_to_flex_angles(kps)
            flat_pred = np.concatenate([pred_by_finger[f] for f in FINGER_JOINT_TRIPLES])

            for col, p, g in zip(self.config.gt_cols, flat_pred, gt):
                per_col_pred[col].append(p)
                per_col_gt[col].append(g)
            frame_ids.append(frame_idx)

        return EvalResult(
            pred=pd.DataFrame(per_col_pred),
            gt=pd.DataFrame(per_col_gt),
            frame_ids=frame_ids,
            timings=dict(self.timings),
        )


class EvalResult:
    def __init__(self, pred: pd.DataFrame, gt: pd.DataFrame, frame_ids: list[int], timings: dict):
        self.pred = pred
        self.gt = gt
        self.frame_ids = frame_ids
        self.timings = timings

    def mae_per_joint(self) -> pd.Series:
        return (self.pred - self.gt).abs().mean()

    def mae_total(self) -> float:
        return float(self.mae_per_joint().mean())

    def report(self) -> str:
        lines = ["=== RTMPose vs Manus GT MAE ==="]
        for col, mae in self.mae_per_joint().items():
            lines.append(f"{col}: {mae:.2f}°")
        lines.append(f"\n전체 평균 MAE: {self.mae_total():.2f}°")

        inf_times = self.timings.get("inference", [])
        if inf_times:
            t = np.array(inf_times)
            lines.append(
                f"\n추론 시간 (cache miss만): {t.mean()*1000:.2f}ms ± {t.std()*1000:.2f}ms"
                f" / frame  ({len(t)}회, {1/t.mean():.1f} fps)"
            )
        return "\n".join(lines)
