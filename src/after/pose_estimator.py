"""PoseEstimator 인터페이스.

원본 코드는 mmpose 호출이 평가 루프 안에 그대로 박혀 있어서, HaMeR나
WiLoR로 같은 비교를 하려면 스크립트 전체를 복붙해야 했다
(compare_new.py, compare_new2.py, compare_26.py, full_compare.py ...
이 전부 그 결과물로 보인다).

`predict(image) -> keypoints` 하나만 구현하면 Evaluator는 어떤 모델이든
그대로 받아들인다. HaMeR/WiLoR를 추가하려면 이 파일에 클래스 하나만
더 만들면 된다.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class PoseEstimator(Protocol):
    def predict(self, image: np.ndarray) -> "np.ndarray | None":
        """(21, 2) 키포인트 배열을 반환. 손이 검출되지 않으면 None."""
        ...


class RTMPoseEstimator:
    """mmpose RTMPose top-down 추론을 PoseEstimator 인터페이스로 감싼 것."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        import sys
        sys.path.insert(0, "/home/abc/mmpose")
        from mmpose.apis import init_model, inference_topdown

        self._inference_topdown = inference_topdown
        self._model = init_model(config_path, checkpoint_path, device=device)

    def predict(self, image: np.ndarray) -> "np.ndarray | None":
        results = self._inference_topdown(self._model, image)
        if not results or len(results[0].pred_instances.keypoints) == 0:
            return None
        return results[0].pred_instances.keypoints[0]


# 향후 HaMeR/WiLoR 추가 시 예시:
#
# class HaMeRPoseEstimator:
#     def __init__(self, ...): ...
#     def predict(self, image: np.ndarray) -> "np.ndarray | None":
#         ...  # HaMeR 추론 + MANO -> 21 keypoints 변환
#
# Evaluator(config, HaMeRPoseEstimator(...)) 로 바로 교체 가능.
