"""키포인트 -> 관절 굴곡각 변환.

원본은 `calc_angle`을 관절 12개마다 한 번씩, 즉 프레임당 12번의 개별 python
함수 호출로 처리했고(각 호출마다 np.dot/np.linalg.norm/arccos를 따로 수행),
"어떤 관절이 어떤 손가락에 속하는지"는 호출 순서로만 암묵적으로 표현돼
있었다. 손가락 -> 관절 triple을 dict로 명시하고, 각 손가락의 3개 관절을
한 번에 벡터화해서 계산하면 같은 결과를 더 적은 python-level 호출로 얻고,
구조도 한눈에 보인다 (5.1 "같은 계산을 반복하는 부분" 대응).
"""
from __future__ import annotations
import numpy as np

# RTMPose hand21 레이아웃: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle,
# 13-16=ring, 17-20=pinky
FINGER_JOINT_TRIPLES: dict[str, list[tuple[int, int, int]]] = {
    "Index":  [(0, 5, 6),   (5, 6, 7),    (6, 7, 8)],
    "Middle": [(0, 9, 10),  (9, 10, 11),  (10, 11, 12)],
    "Ring":   [(0, 13, 14), (13, 14, 15), (14, 15, 16)],
    "Pinky":  [(0, 17, 18), (17, 18, 19), (18, 19, 20)],
}


def _batch_angles(kps: np.ndarray, triples: list[tuple[int, int, int]]) -> np.ndarray:
    """여러 (p1, p2, p3) triple의 각도를 한 번의 numpy 연산으로 계산."""
    idx1 = np.array([t[0] for t in triples])
    idx2 = np.array([t[1] for t in triples])
    idx3 = np.array([t[2] for t in triples])

    v1 = kps[idx1] - kps[idx2]
    v2 = kps[idx3] - kps[idx2]
    dot = np.einsum("ij,ij->i", v1, v2)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
    angles = np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))
    return 180.0 - angles


def keypoints_to_flex_angles(kps: np.ndarray) -> dict[str, np.ndarray]:
    """{손가락 이름: array([MCP, PIP, DIP])} 형태로 반환 (GT_COLUMNS 순서와 일치)."""
    return {
        finger: _batch_angles(kps, triples)
        for finger, triples in FINGER_JOINT_TRIPLES.items()
    }
