# RTMPose vs Manus GT 평가 코드 리팩토링

## 실행 방법

```bash
conda activate mmpose
cd src/after
python main.py              # 평가 1회 실행 + MAE/추론시간 리포트
python benchmark_sizes.py   # 입력 크기별(30/61/123 frame) 순수 추론시간 비교
```

`src/before/rtmpose_manus_mapping.py` 는 원본(최적화 전) 코드, `src/after/` 는 리팩토링
결과입니다. 경로(`img_dir`, `csv_path`, 체크포인트 경로)는 `main.py`/`benchmark_sizes.py`
안의 `EvalConfig` 생성 부분에서 본인 환경에 맞게 수정하면 됩니다. 다른 데이터셋/체크포인트로
바꿀 때 이 한 곳만 고치면 됩니다.

실행 후 `src/after/results/` 에 `mae_per_joint.csv`, `environment.txt`,
`benchmark_by_input_size.csv` 가 생성됩니다.

## 파일 구조

| 파일 | 역할 | 적용 기법 |
|---|---|---|
| `src/after/config.py` | 경로/컬럼/디바이스 설정 (dataclass) | C: config/state 분리 |
| `src/after/pose_estimator.py` | RTMPose 추론 래핑, `PoseEstimator` Protocol | C: 단일책임원칙, Protocol |
| `src/after/decorators.py` | `@timed`, `@cached_by_arg` | D: timing/caching decorator |
| `src/after/angle_calc.py` | 키포인트 → 손가락별 굴곡각 (dict + 벡터화) | A: 자료구조(dict), 반복계산 제거 |
| `src/after/evaluator.py` | 평가 루프 + MAE 계산 + 리포트 | C: 단일책임원칙 |
| `src/after/main.py` | 엔트리포인트, 반복 측정, CSV/환경 저장 | 벤치마크(6장 요구사항) |
| `src/after/benchmark_sizes.py` | 입력 크기별 순수 추론시간 측정 | 6장: 입력 크기 변화 비교 |
| `report/report.pdf` | 최종 보고서 (7.1~7.6) | — |

## 과제 보고서 항목과의 매핑

- **5.1 (문제점 분석)**: 원본은 모델 로딩/GT 로딩/각도계산/추론루프/MAE/출력이 한 스크립트에
  혼재, `calc_angle` 12회 개별 호출, 계측 없음, 모델 교체 시 전체 복붙(`compare_new.py`,
  `compare_26.py` 등 다수 파일이 그 증거).
- **C (Class)**: `PoseEstimator` Protocol 도입으로 HaMeR/WiLoR 추가 시 클래스 하나만 구현하면
  됨 — "재사용/확장 어려움" 문제를 직접 해결.
- **D (Decorator)**: `@cached_by_arg` 와 `@timed` 를 겹쳐 적용해, 캐시 히트 시 추론 시간이
  거의 0이 되는 것을 실측값으로 보여줄 수 있음 (캐싱 효과의 정량적 증거).
- **A (자료구조)**: `FINGER_JOINT_TRIPLES` dict로 관절-손가락 매핑을 명시적 구조로 표현하고,
  `np.einsum` 벡터화로 손가락당 3개 관절을 한 번에 계산 (원본은 관절마다 개별 함수 호출).
- **6 (성능 측정)**: `main.py` 가 N회 반복 실행으로 평균/표준편차를 자동 측정하고
  `environment.txt` 에 OS/Python/CUDA/GPU 정보를 기록.

## 검증 결과

본인 환경(RTX 4070, conda env `mmpose`)에서 실행 검증 완료.

- 원본 vs 리팩토링 MAE 일치: 두 버전 모두 **34.82°**
- CPU 추론: 2833.06ms ± 929.96ms / frame (0.4 fps)
- GPU 추론: 85.8~86.8ms / frame (약 11.6 fps)
- 입력 크기별 (GPU, warm-up 후): 30 frame 2.605s / 61 frame 5.276s / 123 frame 10.558s
  — 프레임 수에 거의 정확히 비례 (선형 스케일링)

자세한 해석은 `report/report.pdf` 7.5~7.6 참조.
