"""평가 설정.

원본 스크립트는 경로, 디바이스, GT 컬럼 목록이 모델 로딩/추론 로직과
한 파일에 뒤섞여 있어서, 다른 데이터셋이나 다른 체크포인트로 평가하려면
스크립트 본문을 직접 고쳐야 했다. 설정을 별도 객체로 분리하면 실행 조건
변경 시 이 파일만 건드리면 된다 (5.1 "실험 조건 변경이 어려운 부분" 대응).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# RTMPose hand21 GT 컬럼 (Manus CSV 기준)
GT_COLUMNS: tuple[str, ...] = (
    "Index_MCP_Flex", "Index_PIP_Flex", "Index_DIP_Flex",
    "Middle_MCP_Flex", "Middle_PIP_Flex", "Middle_DIP_Flex",
    "Ring_MCP_Flex", "Ring_PIP_Flex", "Ring_DIP_Flex",
    "Pinky_MCP_Flex", "Pinky_PIP_Flex", "Pinky_DIP_Flex",
)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    img_dir: Path
    csv_path: Path
    rtmpose_config: str
    rtmpose_checkpoint: str
    device: str = "cuda"
    csv_fps_multiplier: int = 4  # 영상 15fps, Manus CSV 60fps -> 4배
    gt_cols: tuple[str, ...] = field(default_factory=lambda: GT_COLUMNS)
    n_repeats: int = 5  # 벤치마크용 반복 횟수 (평균/표준편차 측정)
