"""
人体关键点数据预处理脚本（优化版）

新增优化：
1. 时序平滑：对连续帧关键点做移动平均，消除抖动
2. 异常值修正：过滤超出画面/不合理的关键点坐标
3. 置信度加权：低置信度关键点用邻帧均值替换
4. 特征标准化：对最终特征做Z-score标准化，提升LSTM训练稳定性
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
import sys
import glob
from datetime import datetime
from scipy.signal import medfilt  # 新增：中值滤波
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# import sql
from tqdm import tqdm
# from .. import PoseKeypoints

# 关键点索引定义（YOLO Pose 17个关键点）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 定义关键的三点组合（用于计算角度）
ANGLE_TRIPLETS = [
    (5, 7, 9),   # left_shoulder -> left_elbow -> left_wrist
    (6, 8, 10),  # right_shoulder -> right_elbow -> right_wrist
    (11, 13, 15),# left_hip -> left_knee -> left_ankle
    (12, 14, 16),# right_hip -> right_knee -> right_ankle
    (5, 11, 13), # left_shoulder -> left_hip -> left_knee
    (6, 12, 14), # right_shoulder -> right_hip -> right_knee
    (0, 5, 6),   # nose -> left_shoulder -> right_shoulder
]

# ===================== 新增：异常值修正 =====================
def correct_outliers(keypoints: List[Dict], img_width: float = 1.0, img_height: float = 1.0) -> List[Dict]:
    """
    修正异常关键点（超出画面/不合理坐标）
    :param keypoints: 关键点列表
    :param img_width: 画面宽度（归一化后为1.0）
    :param img_height: 画面高度（归一化后为1.0）
    :return: 修正后的关键点
    """
    corrected_kps = []
    for kp in keypoints:
        corrected_kp = kp.copy()

        # 1. 坐标限制在画面内（0~1）
        corrected_kp["x"] = np.clip(kp.get("x", 0.0), 0.0, img_width)
        corrected_kp["y"] = np.clip(kp.get("y", 0.0), 0.0, img_height)

        # 2. 修正明显不合理的关键点（比如离身体中心过远）
        if kp.get("confidence", 0) > 0.3:
            # 计算到身体中心的距离（髋部中心）
            left_hip = keypoints[11] if 11 < len(keypoints) else {"x":0.5, "y":0.5}
            right_hip = keypoints[12] if 12 < len(keypoints) else {"x":0.5, "y":0.5}
            center_x = (left_hip["x"] + right_hip["x"]) / 2
            center_y = (left_hip["y"] + right_hip["y"]) / 2

            distance = np.sqrt((kp["x"] - center_x)**2 + (kp["y"] - center_y)**2)
            # 如果距离超过画面一半，判定为异常点
            if distance > max(img_width, img_height) / 2:
                corrected_kp["confidence"] = 0.0  # 标记为低置信度

        corrected_kps.append(corrected_kp)

    return corrected_kps

# ===================== 新增：时序平滑 =====================
def smooth_keypoints_sequence(segment_keypoints: List[List[Dict]], window_size: int = 3) -> List[List[Dict]]:
    """
    对整个片段的关键点序列做时序平滑（移动平均+中值滤波）
    :param segment_keypoints: 片段所有帧的关键点列表 [[帧1关键点], [帧2关键点], ...]
    :param window_size: 平滑窗口大小（奇数）
    :return: 平滑后的关键点序列
    """
    if len(segment_keypoints) < window_size:
        return segment_keypoints

    # 1. 转换为数组格式便于处理 (帧数, 关键点数, 3) -> (x, y, confidence)
    seq_array = np.zeros((len(segment_keypoints), len(KEYPOINT_NAMES), 3))
    for i, frame_kps in enumerate(segment_keypoints):
        for j, kp in enumerate(frame_kps):
            if j < len(KEYPOINT_NAMES):
                seq_array[i, j, 0] = kp.get("x", 0.0)
                seq_array[i, j, 1] = kp.get("y", 0.0)
                seq_array[i, j, 2] = kp.get("confidence", 0.0)

    # 2. 对x/y坐标做中值滤波（抗噪）+ 移动平均（平滑）
    smoothed_array = seq_array.copy()
    for j in range(len(KEYPOINT_NAMES)):
        # 只对置信度>0.1的点做平滑
        valid_mask = seq_array[:, j, 2] > 0.1

        if np.sum(valid_mask) > window_size:
            # 中值滤波去毛刺
            smoothed_array[:, j, 0] = medfilt(seq_array[:, j, 0], kernel_size=window_size)
            smoothed_array[:, j, 1] = medfilt(seq_array[:, j, 1], kernel_size=window_size)

            # 移动平均平滑
            for coord in [0, 1]:  # x和y分别平滑
                padded = np.pad(smoothed_array[:, j, coord], (window_size//2, window_size//2), mode='edge')
                smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
                smoothed_array[:, j, coord] = smoothed

    # 3. 置信度加权：低置信度点用平滑值替换
    for i in range(len(segment_keypoints)):
        for j in range(len(KEYPOINT_NAMES)):
            original_conf = seq_array[i, j, 2]
            if original_conf < 0.3:
                # 低置信度点：用平滑值替换
                segment_keypoints[i][j]["x"] = smoothed_array[i, j, 0]
                segment_keypoints[i][j]["y"] = smoothed_array[i, j, 1]
                # 置信度用邻帧均值填充
                start = max(0, i - window_size//2)
                end = min(len(segment_keypoints), i + window_size//2 + 1)
                avg_conf = np.mean(seq_array[start:end, j, 2])
                segment_keypoints[i][j]["confidence"] = avg_conf
            else:
                # 高置信度点：保留原始值，叠加少量平滑值
                segment_keypoints[i][j]["x"] = 0.8 * segment_keypoints[i][j]["x"] + 0.2 * smoothed_array[i, j, 0]
                segment_keypoints[i][j]["y"] = 0.8 * segment_keypoints[i][j]["y"] + 0.2 * smoothed_array[i, j, 1]

    return segment_keypoints

# ===================== 原有函数：归一化（小幅修改） =====================
def normalize_coordinates(keypoints: List[Dict], method: str = "center") -> List[Dict]:
    """
    归一化关键点坐标（新增：先修正异常值）
    """
    if not keypoints or len(keypoints) == 0:
        return []

    # 新增：先修正异常值
    keypoints = correct_outliers(keypoints)

    # 提取有效关键点（置信度>0.3）
    valid_kps = [kp for kp in keypoints if kp.get("confidence", 0) > 0.3]

    if len(valid_kps) == 0:
        return keypoints

    # 提取坐标
    coords = np.array([[kp["x"], kp["y"]] for kp in valid_kps])

    if method == "center":
        # 使用髋部中心作为参考点
        left_hip_idx = 11
        right_hip_idx = 12

        if (left_hip_idx < len(keypoints) and keypoints[left_hip_idx].get("confidence", 0) > 0.3 and
            right_hip_idx < len(keypoints) and keypoints[right_hip_idx].get("confidence", 0) > 0.3):
            center_x = (keypoints[left_hip_idx]["x"] + keypoints[right_hip_idx]["x"]) / 2
            center_y = (keypoints[left_hip_idx]["y"] + keypoints[right_hip_idx]["y"]) / 2
        else:
            center_x = np.mean([kp["x"] for kp in valid_kps])
            center_y = np.mean([kp["y"] for kp in valid_kps])

        # 计算到中心的距离作为归一化因子
        distances = np.sqrt(np.sum((coords - np.array([center_x, center_y])) ** 2, axis=1))
        max_distance = np.max(distances) if len(distances) > 0 and np.max(distances) > 0 else 1.0

        # 归一化：相对于中心，使用最大距离作为尺度
        normalized_kps = []
        for kp in keypoints:
            if kp.get("confidence", 0) > 0.3:
                norm_kp = {
                    "name": kp["name"],
                    "x": (kp["x"] - center_x) / max_distance,
                    "y": (kp["y"] - center_y) / max_distance,
                    "confidence": kp["confidence"]
                }
            else:
                norm_kp = kp.copy()
                # 新增：低置信度点坐标归一化（避免数值过大）
                norm_kp["x"] = (kp.get("x", 0.0) - center_x) / max_distance
                norm_kp["y"] = (kp.get("y", 0.0) - center_y) / max_distance
            normalized_kps.append(norm_kp)

        return normalized_kps

    elif method == "shoulder":
        # 使用肩膀中心作为参考点
        left_shoulder_idx = 5
        right_shoulder_idx = 6

        if (left_shoulder_idx < len(keypoints) and keypoints[left_shoulder_idx].get("confidence", 0) > 0.3 and
            right_shoulder_idx < len(keypoints) and keypoints[right_shoulder_idx].get("confidence", 0) > 0.3):
            center_x = (keypoints[left_shoulder_idx]["x"] + keypoints[right_shoulder_idx]["x"]) / 2
            center_y = (keypoints[left_shoulder_idx]["y"] + keypoints[right_shoulder_idx]["y"]) / 2
            # 使用肩膀宽度作为尺度
            scale = abs(keypoints[right_shoulder_idx]["x"] - keypoints[left_shoulder_idx]["x"])
            if scale == 0:
                scale = 1.0
        else:
            center_x = np.mean([kp["x"] for kp in valid_kps])
            center_y = np.mean([kp["y"] for kp in valid_kps])
            scale = 1.0

        normalized_kps = []
        for kp in keypoints:
            if kp.get("confidence", 0) > 0.3:
                norm_kp = {
                    "name": kp["name"],
                    "x": (kp["x"] - center_x) / scale,
                    "y": (kp["y"] - center_y) / scale,
                    "confidence": kp["confidence"]
                }
            else:
                norm_kp = kp.copy()
                norm_kp["x"] = (kp.get("x", 0.0) - center_x) / scale
                norm_kp["y"] = (kp.get("y", 0.0) - center_y) / scale
            normalized_kps.append(norm_kp)

        return normalized_kps

    else:  # minmax
        # 最小-最大归一化
        all_x = [kp["x"] for kp in valid_kps]
        all_y = [kp["y"] for kp in valid_kps]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        range_x = max_x - min_x if max_x != min_x else 1.0
        range_y = max_y - min_y if max_y != min_y else 1.0

        normalized_kps = []
        for kp in keypoints:
            if kp.get("confidence", 0) > 0.3:
                norm_kp = {
                    "name": kp["name"],
                    "x": (kp["x"] - min_x) / range_x,
                    "y": (kp["y"] - min_y) / range_y,
                    "confidence": kp["confidence"]
                }
            else:
                norm_kp = kp.copy()
                norm_kp["x"] = (kp.get("x", 0.0) - min_x) / range_x
                norm_kp["y"] = (kp.get("y", 0.0) - min_y) / range_y
            normalized_kps.append(norm_kp)

        return normalized_kps

# ===================== 原有函数：角度计算（无修改） =====================
def calculate_angle_sin_cos(p1: Dict, p2: Dict, p3: Dict) -> Tuple[float, float]:
    if (p1.get("confidence", 0) < 0.3 or p2.get("confidence", 0) < 0.3 or
        p3.get("confidence", 0) < 0.3):
        return (0.0, 1.0)

    v1 = np.array([p2["x"] - p1["x"], p2["y"] - p1["y"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

    len_v1 = np.linalg.norm(v1)
    len_v2 = np.linalg.norm(v2)

    if len_v1 == 0 or len_v2 == 0:
        return (0.0, 1.0)

    v1_norm = v1 / len_v1
    v2_norm = v2 / len_v2

    cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
    dot = np.dot(v1_norm, v2_norm)

    angle = np.arctan2(cross, dot)

    return (np.sin(angle), np.cos(angle))

def extract_angle_features(keypoints: List[Dict]) -> Dict[str, Tuple[float, float]]:
    angle_features = {}

    angle_names = [
        "left_arm_angle", "right_arm_angle", "left_leg_angle",
        "right_leg_angle", "left_torso_angle", "right_torso_angle", "head_angle"
    ]

    for i, (idx1, idx2, idx3) in enumerate(ANGLE_TRIPLETS):
        if idx1 < len(keypoints) and idx2 < len(keypoints) and idx3 < len(keypoints):
            p1 = keypoints[idx1]
            p2 = keypoints[idx2]
            p3 = keypoints[idx3]

            sin_val, cos_val = calculate_angle_sin_cos(p1, p2, p3)
            angle_features[angle_names[i]] = (sin_val, cos_val)
        else:
            angle_features[angle_names[i]] = (0.0, 1.0)

    return angle_features

# ===================== 修改：process_segment（新增时序平滑） =====================
def process_segment(segment: Dict, normalize_method: str = "center") -> Dict:
    """
    处理单个视频片段（新增：先提取所有帧关键点→时序平滑→再归一化）
    """
    processed_segment = {
        "video_file": segment.get("video_file", ""),
        "start_frame": segment.get("start_frame", 0),
        "end_frame": segment.get("end_frame", 0),
        "num_frames": len(segment.get("frame_data", [])),
        "frame_data": []
    }

    # 第一步：提取整个片段的原始关键点序列
    segment_keypoints = []
    valid_frame_indices = []
    for i, frame_data in enumerate(segment.get("frame_data", [])):
        kps = frame_data.get("keypoints", [])
        if len(kps) > 0:
            segment_keypoints.append(kps)
            valid_frame_indices.append(i)

    # 第二步：对整个序列做时序平滑（核心优化）
    segment_keypoints = smooth_keypoints_sequence(segment_keypoints)

    # 第三步：逐帧处理（归一化+角度提取）
    for i, frame_idx in enumerate(valid_frame_indices):
        frame_data = segment["frame_data"][frame_idx]
        keypoints = segment_keypoints[i]

        # 1. 修正异常值（已整合到归一化中）
        # 2. 归一化坐标
        normalized_kps = normalize_coordinates(keypoints, method=normalize_method)

        # 3. 提取角度特征
        angle_features = extract_angle_features(normalized_kps)

        # 4. 构建处理后的帧数据
        processed_frame = {
            "frame_number": frame_data.get("frame_number", 0),
            "num_persons": frame_data.get("num_persons", 0),
            "normalized_keypoints": normalized_kps,
            "angle_features": {
                name: {"sin": sin_val, "cos": cos_val}
                for name, (sin_val, cos_val) in angle_features.items()
            }
        }

        processed_segment["frame_data"].append(processed_frame)

    return processed_segment

# ===================== 修改：convert_to_features（新增特征标准化） =====================
def convert_to_features(processed_json_path: str) -> np.ndarray:
    """
    将处理后的JSON数据转换为特征数组（新增：Z-score标准化）
    """
    print(f"正在转换特征: {os.path.basename(processed_json_path)}")

    with open(processed_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = []

    for segment in tqdm(data.get("segments", []), desc="转换特征"):
        segment_features = []

        for frame_data in segment.get("frame_data", []):
            frame_feature = []

            # 1. 添加归一化后的关键点坐标（x, y, confidence）
            normalized_kps = frame_data.get("normalized_keypoints", [])
            for kp in normalized_kps:
                frame_feature.append(kp.get("x", 0.0))
                frame_feature.append(kp.get("y", 0.0))
                frame_feature.append(kp.get("confidence", 0.0))

            # 2. 添加角度特征（sin和cos）
            angle_features = frame_data.get("angle_features", {})
            for angle_name in ["left_arm_angle", "right_arm_angle",
                              "left_leg_angle", "right_leg_angle",
                              "left_torso_angle", "right_torso_angle", "head_angle"]:
                angle_data = angle_features.get(angle_name, {"sin": 0.0, "cos": 1.0})
                frame_feature.append(angle_data.get("sin", 0.0))
                frame_feature.append(angle_data.get("cos", 0.0))

            segment_features.append(np.array(frame_feature, dtype=np.float32))

        # 确保每个片段有30帧
        if len(segment_features) == 30:
            features.append(np.array(segment_features))

    features_array = np.array(features, dtype=np.float32)

    # 新增：特征标准化（Z-score）→ 均值0，方差1，提升LSTM训练稳定性
    if len(features_array) > 0:
        mean = np.mean(features_array, axis=(0, 1), keepdims=True)
        std = np.std(features_array, axis=(0, 1), keepdims=True)
        std = np.maximum(std, 1e-6)  # 避免除0
        features_array = (features_array - mean) / std

    print(f"✓ 特征转换完成")
    print(f"  特征形状: {features_array.shape}")
    print(f"  特征维度: {features_array.shape[2]} (17关键点×3 + 7角度×2 = 65)")

    return features_array

# ===================== 主函数（小幅修改，适配测试） =====================
def main():
    print("="*60)
    print("人体关键点数据预处理脚本（优化版）")
    print("="*60)
    # "E:\最新的毕业设计\毕业设计\script\pose_data_20251202_161341.json"
    # 测试用：使用当前目录
    DATA_DIR = "E:/最新的毕业设计/毕业设计/script/try00/asset/JSONFOLDER"

    # 查找最新的JSON文件
    json_pattern = os.path.join(DATA_DIR, "short_video_pose_data_20260227_133703.json")
    json_files = glob.glob(json_pattern)


    if not json_files:
        print(f"✗ 未找到JSON数据文件，创建测试文件...")
        # 创建测试用JSON文件
        test_data = {
            "segments": [
                {
                    "video_file": "test_video.mp4",
                    "start_frame": 0,
                    "end_frame": 29,
                    "frame_data": [
                        {
                            "frame_number": i,
                            "num_persons": 1,
                            "keypoints": [
                                {"name": name, "x": np.random.uniform(0, 1), "y": np.random.uniform(0, 1), "confidence": np.random.uniform(0.2, 0.9)}
                                for name in KEYPOINT_NAMES
                            ]
                        } for i in range(30)
                    ]
                }
            ]
        }
        test_json_path = os.path.join(DATA_DIR, "pose_data_test.json")
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        json_files = [test_json_path]

    latest_json = max(json_files, key=os.path.getctime)
    print(f"\n找到JSON文件: {os.path.basename(latest_json)}")

    # 输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = os.path.join(DATA_DIR, f"pose_data_processed_{timestamp}.json")
    output_npy_path = os.path.join(DATA_DIR, f"pose_features_processed_{timestamp}.npy")

    # 归一化方法选择
    NORMALIZE_METHOD = "center"

    print(f"\n配置信息:")
    print(f"  输入文件: {os.path.basename(latest_json)}")
    print(f"  输出JSON: {os.path.basename(output_json_path)}")
    print(f"  输出NPY: {os.path.basename(output_npy_path)}")
    print(f"  归一化方法: {NORMALIZE_METHOD}")

    # 1. 处理JSON数据
    print(f"\n{'='*60}")
    print("步骤1: 处理JSON数据（归一化 + 角度特征 + 时序平滑）")
    print(f"{'='*60}")
    process_json_file(latest_json, output_json_path, NORMALIZE_METHOD)

    # 2. 转换为特征数组
    print(f"\n{'='*60}")
    print("步骤2: 转换为特征数组（标准化）")
    print(f"{'='*60}")
    features = convert_to_features(output_json_path)

    # 3. 保存特征数组
    print(f"\n{'='*60}")
    print("步骤3: 保存特征数组")
    print(f"{'='*60}")
    np.save(output_npy_path, features)
    print(f"✓ 特征数组已保存: {os.path.basename(output_npy_path)}")

    # 4. 保存元数据
    metadata_path = os.path.join(DATA_DIR, f"pose_metadata_processed_{timestamp}.json")
    metadata = {
        "features_shape": list(features.shape),
        "num_segments": len(features),
        "feature_dim": features.shape[2],
        "json_file": os.path.basename(output_json_path),
        "npy_file": os.path.basename(output_npy_path),
        "normalize_method": NORMALIZE_METHOD,
        "timestamp": timestamp,
        "optimizations": ["时序平滑", "异常值修正", "置信度加权", "特征标准化"]
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✓ 元数据已保存: {os.path.basename(metadata_path)}")

    print(f"\n{'='*60}")
    print("✅ 预处理完成！")
    print(f"{'='*60}")
    print(f"生成的文件:")
    print(f"  处理后的JSON: {os.path.basename(output_json_path)}")
    print(f"  特征数组: {os.path.basename(output_npy_path)}")
    print(f"  元数据: {os.path.basename(metadata_path)}")
    print(f"\n所有文件已保存到: {os.path.abspath(DATA_DIR)}")

# 保持原有process_json_file函数不变
def process_json_file(input_json_path: str, output_json_path: str,
                     normalize_method: str = "center"):
    print(f"正在处理JSON文件: {os.path.basename(input_json_path)}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  原始数据包含 {len(data.get('segments', []))} 个片段")

    processed_segments = []
    for segment in tqdm(data.get("segments", []), desc="处理片段"):
        processed_segment = process_segment(segment, normalize_method)
        processed_segments.append(processed_segment)

    output_data = {
        "metadata": {
            "source_file": os.path.basename(input_json_path),
            "normalize_method": normalize_method,
            "total_segments": len(processed_segments),
            "segment_length": 30,
            "keypoint_names": KEYPOINT_NAMES,
            "angle_features": [
                "left_arm_angle", "right_arm_angle",
                "left_leg_angle", "right_leg_angle",
                "left_torso_angle", "right_torso_angle", "head_angle"
            ],
            "processing_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "optimizations": ["时序平滑", "异常值修正", "置信度加权"]
        },
        "segments": processed_segments
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 处理完成，已保存到: {os.path.basename(output_json_path)}")
    print(f"  处理后的片段数: {len(processed_segments)}")

if __name__ == '__main__':
    main()