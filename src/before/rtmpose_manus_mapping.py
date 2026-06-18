import numpy as np
import pandas as pd
import cv2
import sys
sys.path.insert(0, '/home/abc/mmpose')
from mmpose.apis import init_model, inference_topdown
from pathlib import Path

# 모델 로드
device = 'cuda'
config = '/home/abc/mmpose/checkpoints/rtmpose-m_8xb256-210e_hand5-256x256.py'
checkpoint = '/home/abc/mmpose/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
pose_model = init_model(config, checkpoint, device=device)

# 경로
img_dir = Path('/home/abc/hamer/inpainting_glove_frames2')
csv_path = '/home/abc/hamer/Untitled_2026-05-12_15-36-07_1_R.csv'

# CSV 로드
df = pd.read_csv(csv_path)

# Manus GT 컬럼
gt_cols = [
    'Index_MCP_Flex', 'Index_PIP_Flex', 'Index_DIP_Flex',
    'Middle_MCP_Flex', 'Middle_PIP_Flex', 'Middle_DIP_Flex',
    'Ring_MCP_Flex', 'Ring_PIP_Flex', 'Ring_DIP_Flex',
    'Pinky_MCP_Flex', 'Pinky_PIP_Flex', 'Pinky_DIP_Flex',
]


def calc_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


# RTMPose 키포인트 → 관절각도 매핑
# RTMPose: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
def kps_to_angles(kps):
    angles = np.array([
        calc_angle(kps[0], kps[5], kps[6]),
        calc_angle(kps[5], kps[6], kps[7]),
        calc_angle(kps[6], kps[7], kps[8]),
        calc_angle(kps[0], kps[9], kps[10]),
        calc_angle(kps[9], kps[10], kps[11]),
        calc_angle(kps[10], kps[11], kps[12]),
        calc_angle(kps[0], kps[13], kps[14]),
        calc_angle(kps[13], kps[14], kps[15]),
        calc_angle(kps[14], kps[15], kps[16]),
        calc_angle(kps[0], kps[17], kps[18]),
        calc_angle(kps[17], kps[18], kps[19]),
        calc_angle(kps[18], kps[19], kps[20]),
    ])
    return 180.0 - angles


rtm_angles_list = []
gt_angles_list = []
frame_ids = []

img_files = sorted(img_dir.glob('*.png'))
print(f"총 {len(img_files)}장 처리")

for img_path in img_files:
    frame_idx = int(img_path.stem)
    csv_row = (frame_idx - 1) * 4
    if csv_row >= len(df):
        continue

    img = cv2.imread(str(img_path))
    results = inference_topdown(pose_model, img)
    if not results or len(results[0].pred_instances.keypoints) == 0:
        print(f"[{frame_idx}] 미검출 스킵")
        continue

    kps = results[0].pred_instances.keypoints[0]
    rtm_angles = kps_to_angles(kps)

    gt_row = df.iloc[csv_row]
    gt_angles = gt_row[gt_cols].values.astype(float)

    rtm_angles_list.append(rtm_angles)
    gt_angles_list.append(gt_angles)
    frame_ids.append(frame_idx)

rtm_angles_arr = np.array(rtm_angles_list)
gt_angles_arr = np.array(gt_angles_list)

# MAE 계산
mae_per_joint = np.mean(np.abs(rtm_angles_arr - gt_angles_arr), axis=0)
mae_total = np.mean(mae_per_joint)

print(f"\n=== RTMPose vs Manus GT MAE ===")
for col, mae in zip(gt_cols, mae_per_joint):
    print(f"{col}: {mae:.2f}°")
print(f"\n전체 평균 MAE: {mae_total:.2f}°")
