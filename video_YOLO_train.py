"""
视频YOLO骨骼信息提取脚本
适配场景：批量处理已截取的30帧短视频（如 gesture_1_frames_15-45.mp4）
功能：
1. 每个短视频直接提取为1个30帧片段（无需间隔）
2. 不足30帧自动补0，不丢弃数据
3. 保存JSON/NPY格式，适配LSTM训练
"""

import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from tqdm import tqdm
from datetime import datetime


class YOLOPoseExtractor:
    """YOLO姿态提取器 - 适配30帧短视频批量处理"""

    def __init__(self, model_name: str = "yolov8l-pose.pt", device: str = "auto"):
        print(f"正在加载YOLO模型: {model_name}")
        self.model = YOLO(model_name)

        # 自动检测设备
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            self.model.to(device)
            print(f"✓ 使用设备: CUDA")
        else:
            print(f"✓ 使用设备: CPU")

        self.device = device

        # YOLO Pose 17个关键点定义
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        # 关键点分组（适配你的hands+legs需求）
        self.hand_keypoints = ["left_wrist", "right_wrist"]
        self.leg_keypoints = ["left_hip", "right_hip", "left_knee", "right_knee",
                             "left_ankle", "right_ankle"]

    def extract_single_short_video(self, video_path: str, target_frames: int = 30) -> Dict:
        """
        处理单个30帧短视频，返回1个片段数据
        :param video_path: 短视频路径（如 gesture_1_frames_15-45.mp4）
        :param target_frames: 目标帧数（固定30）
        :return: 单个片段的完整数据
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ 无法打开视频: {os.path.basename(video_path)}")
            return None

        # 获取视频基本信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  处理: {os.path.basename(video_path)} | 尺寸: {width}x{height} | 实际帧数: {total_frames} | 目标帧数: {target_frames}")

        # 初始化片段数据（单个短视频对应1个片段）
        segment_data = {
            "video_file": os.path.basename(video_path),
            "video_path": video_path,
            "start_frame": 0,
            "end_frame": total_frames - 1,
            "original_frames": total_frames,
            "frames": []
        }

        # 逐帧读取并提取骨骼信息（核心修改：不跳帧，完整读取）
        frame_idx = 0
        while frame_idx < target_frames:
            # 读取当前帧（超过视频总帧数则停止）
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                # 读取失败：填充空数据（后续补0）
                segment_data["frames"].append({
                    "frame_number": frame_idx,
                    "poses": [],
                    "num_persons": 0
                })
                frame_idx += 1
                continue

            # YOLO Pose检测
            results = self.model(frame, conf=0.5, verbose=False)

            # 提取关键点
            frame_poses = []
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # 只取第一个人（适配你的max_persons=1需求）
                keypoints = results[0].keypoints.data[0]  # 取第一个人的关键点
                person_pose = {
                    "person_id": 0,
                    "keypoints": []
                }

                # 解析17个关键点
                for kp_idx, (x, y, conf) in enumerate(keypoints.cpu().numpy()):
                    person_pose["keypoints"].append({
                        "name": self.keypoint_names[kp_idx],
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(conf)
                    })
                frame_poses.append(person_pose)

            # 保存当前帧数据
            segment_data["frames"].append({
                "frame_number": frame_idx,
                "poses": frame_poses,
                "num_persons": len(frame_poses)
            })

            frame_idx += 1

        cap.release()
        return segment_data

    def extract_from_short_video_folder(self, folder_path: str, target_frames: int = 30) -> List[Dict]:
        """
        批量处理文件夹下的所有30帧短视频
        :param folder_path: 短视频文件夹路径
        :param target_frames: 目标帧数（30）
        :return: 所有片段数据列表（1个视频=1个片段）
        """
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []

        # 遍历文件夹找视频
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))

        if len(video_files) == 0:
            print(f"✗ 未在 {folder_path} 找到任何视频文件")
            return []

        print(f"\n共找到 {len(video_files)} 个短视频文件，开始批量处理...")

        all_segments = []
        # 加进度条，方便查看处理进度
        for video_file in tqdm(video_files, desc="处理短视频"):
            segment = self.extract_single_short_video(video_file, target_frames)
            if segment is not None:
                all_segments.append(segment)

        print(f"\n✓ 批量处理完成 | 成功提取 {len(all_segments)} 个片段（1个视频=1个片段）")
        return all_segments


def convert_poses_to_features(segments: List[Dict], max_persons: int = 1,
                              focus_areas: List[str] = None, target_frames: int = 30) -> np.ndarray:
    """
    转换为特征向量，不足30帧自动补0
    """
    # 关键点索引定义（适配hands+legs）
    hand_indices = [9, 10]    # 左手腕、右手腕
    leg_indices = [11, 12, 13, 14, 15, 16]  # 髋、膝、踝
    upper_body_indices = [5, 6, 7, 8, 9, 10]
    lower_body_indices = [11, 12, 13, 14, 15, 16]

    # 确定要提取的关键点
    if focus_areas is None or 'all' in focus_areas:
        selected_indices = list(range(17))
    else:
        selected_indices = []
        if 'hands' in focus_areas:
            selected_indices.extend(hand_indices)
        if 'legs' in focus_areas:
            selected_indices.extend(leg_indices)
        if 'upper_body' in focus_areas:
            selected_indices.extend(upper_body_indices)
        if 'lower_body' in focus_areas:
            selected_indices.extend(lower_body_indices)
        selected_indices = sorted(list(set(selected_indices)))

    # 计算特征维度
    num_keypoints = len(selected_indices)
    keypoints_per_person = num_keypoints * 3  # x,y,confidence
    feature_dim = keypoints_per_person * max_persons

    print(f"\n特征转换配置:")
    print(f"  提取关键点数量: {num_keypoints} | 索引: {selected_indices}")
    print(f"  特征维度: {feature_dim} (每帧) | 目标帧数: {target_frames}")

    features = []
    for segment in segments:
        segment_features = []

        # 处理每帧数据
        for frame_data in segment["frames"]:
            frame_feature = np.zeros(feature_dim, dtype=np.float32)

            # 只处理第一个人
            if frame_data["poses"] and len(frame_data["poses"]) > 0:
                person_pose = frame_data["poses"][0]
                for feat_idx, kp_idx in enumerate(selected_indices):
                    if kp_idx < len(person_pose["keypoints"]):
                        kp = person_pose["keypoints"][kp_idx]
                        idx = feat_idx * 3
                        frame_feature[idx] = kp["x"]
                        frame_feature[idx + 1] = kp["y"]
                        frame_feature[idx + 2] = kp["confidence"]

            segment_features.append(frame_feature)

        # 确保是30帧（不足补0）
        segment_features = np.array(segment_features, dtype=np.float32)
        if len(segment_features) < target_frames:
            pad_len = target_frames - len(segment_features)
            pad = np.zeros((pad_len, feature_dim), dtype=np.float32)
            segment_features = np.vstack([segment_features, pad])

        features.append(segment_features[:target_frames])  # 防止超过30帧

    return np.array(features, dtype=np.float32)


def save_pose_data(segments: List[Dict], features: np.ndarray, output_dir: str = "."):
    """保存数据（JSON+NPY）"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存JSON（包含视频命名信息）
    json_path = os.path.join(output_dir, f"short_video_pose_data_{timestamp}.json")
    print(f"\n保存JSON数据到: {json_path}")

    json_data = {
        "metadata": {
            "total_segments": len(segments),
            "target_frames": 30,
            "feature_dim": features.shape[2] if len(features.shape) > 2 else features.shape[1],
            "focus_areas": "hands+legs",
            "extraction_time": timestamp,
            "note": "适配30帧短视频，1个视频=1个片段，不足30帧补0"
        },
        "segments": []
    }

    for segment in segments:
        simplified_segment = {
            "video_file": segment["video_file"],
            "video_path": segment["video_path"],
            "original_frames": segment["original_frames"],
            "processed_frames": len(segment["frames"]),
            "frame_data": []
        }

        # 只保存第一个人的关键点（简化数据）
        for frame_data in segment["frames"]:
            simplified_frame = {
                "frame_number": frame_data["frame_number"],
                "num_persons": frame_data["num_persons"],
                "keypoints": []
            }

            if frame_data["poses"]:
                first_person = frame_data["poses"][0]
                for kp in first_person["keypoints"]:
                    simplified_frame["keypoints"].append({
                        "name": kp["name"],
                        "x": kp["x"],
                        "y": kp["y"],
                        "confidence": kp["confidence"]
                    })

            simplified_segment["frame_data"].append(simplified_frame)

        json_data["segments"].append(simplified_segment)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON数据已保存: {json_path}")

    # 2. 保存NPY特征（LSTM训练用）
    npy_path = os.path.join(output_dir, f"short_video_pose_features_{timestamp}.npy")
    np.save(npy_path, features)
    print(f"✓ NPY特征数据已保存: {npy_path}")
    print(f"  特征形状: {features.shape} (片段数, 30帧, 特征维度)")

    # 3. 保存元数据
    metadata_path = os.path.join(output_dir, f"short_video_pose_metadata_{timestamp}.json")
    metadata = {
        "features_shape": list(features.shape),
        "num_segments": len(segments),
        "json_file": os.path.basename(json_path),
        "npy_file": os.path.basename(npy_path),
        "timestamp": timestamp
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return json_path, npy_path, metadata_path


def main():
    """主函数 - 适配30帧短视频处理"""
    print("="*70)
    print("YOLO Pose 30帧短视频骨骼提取脚本（适配gesture_1_frames_15-45.mp4格式）")
    print("="*70)

    # 配置（修改为你的视频文件夹路径）
    VIDEO_FOLDER = r"E:\最新的毕业设计\毕业设计\accset\cut_videos\all_continuous_values1"
    OUTPUT_DIR = r"E:\最新的毕业设计\毕业设计\script\try00\asset\JSONFOLDER"  # JSON/NPY保存到当前目录
    MODEL_NAME = "yolov8l-pose.pt"  # 高精度模型
    FOCUS_AREAS = ['hands', 'legs']  # 重点提取手+腿
    TARGET_FRAMES = 30  # 固定30帧

    # 打印配置信息
    print(f"\n📌 配置信息:")
    print(f"  视频文件夹: {os.path.abspath(VIDEO_FOLDER)}")
    print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  YOLO模型: {MODEL_NAME}")
    print(f"  重点提取: {FOCUS_AREAS}")
    print(f"  目标帧数: {TARGET_FRAMES}")

    # 1. 初始化提取器
    print(f"\n{'='*70}")
    print("步骤1: 加载YOLOv8-Pose模型")
    print(f"{'='*70}")
    extractor = YOLOPoseExtractor(model_name=MODEL_NAME)

    # 2. 批量处理短视频
    print(f"\n{'='*70}")
    print("步骤2: 批量提取短视频骨骼信息")
    print(f"{'='*70}")
    segments = extractor.extract_from_short_video_folder(VIDEO_FOLDER, TARGET_FRAMES)

    if len(segments) == 0:
        print("✗ 未提取到任何片段，程序退出")
        return

    # 3. 转换为特征向量（补0到30帧）
    print(f"\n{'='*70}")
    print("步骤3: 转换为LSTM训练用特征向量")
    print(f"{'='*70}")
    features = convert_poses_to_features(segments, max_persons=1, focus_areas=FOCUS_AREAS, target_frames=TARGET_FRAMES)

    # 4. 保存数据
    print(f"\n{'='*70}")
    print("步骤4: 保存JSON和NPY文件")
    print(f"{'='*70}")
    json_path, npy_path, metadata_path = save_pose_data(segments, features, OUTPUT_DIR)

    # 最终提示
    print(f"\n✅ 全部处理完成！")
    print(f"  📄 JSON文件: {os.path.abspath(json_path)}")
    print(f"  📊 NPY特征: {os.path.abspath(npy_path)}")
    print(f"  📋 元数据: {os.path.abspath(metadata_path)}")
    print(f"\n  特征形状: {features.shape} → 可直接用于LSTM训练！")


if __name__ == '__main__':
    main()