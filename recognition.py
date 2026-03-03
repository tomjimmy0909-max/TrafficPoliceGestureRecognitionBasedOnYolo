import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import json
from sklearn.preprocessing import StandardScaler
import os
import re
from ultralytics import YOLO
from collections import Counter  # 新增：用于统计类别


# ====================== 1. 复用训练时的模型定义 ======================
class FrameLevelMLP(nn.Module):
    """帧级全连接网络（和训练时完全一致）"""

    def __init__(self, input_dim=24, hidden_dims=[512, 256, 128], num_classes=8, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ====================== 2. YOLO Pose特征提取 ======================
class YOLOPoseInference:
    """YOLO Pose推理类 - 适配单帧特征提取"""

    def __init__(self, model_name="yolov8l-pose.pt", device="auto"):
        self.model = YOLO(model_name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

        self.hand_indices = [9, 10]
        self.leg_indices = [11, 12, 13, 14, 15, 16]
        self.selected_indices = sorted(list(set(self.hand_indices + self.leg_indices)))

    def extract_single_frame_feature(self, frame):
        results = self.model(frame, conf=0.5, verbose=False)
        feature_dim = len(self.selected_indices) * 3
        frame_feature = np.zeros(feature_dim, dtype=np.float32)

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            for feat_idx, kp_idx in enumerate(self.selected_indices):
                if kp_idx < len(keypoints):
                    x, y, conf = keypoints[kp_idx]
                    idx = feat_idx * 3
                    frame_feature[idx] = x
                    frame_feature[idx + 1] = y
                    frame_feature[idx + 2] = conf

        return frame_feature


# ====================== 3. 工具函数 ======================
def load_scaler_from_train_data(train_npy_path):
    train_features = np.load(train_npy_path)
    train_features = train_features.reshape(-1, train_features.shape[-1])
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def predict_gesture(model, scaler, frame_feature, label_to_idx, device):
    frame_feature = scaler.transform(frame_feature.reshape(1, -1))
    feature_tensor = torch.tensor(frame_feature, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        output = model(feature_tensor)
        prob = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    pred_label = idx_to_label[pred_idx]
    pred_conf = prob[0][pred_idx].item()

    return pred_label, pred_conf, prob  # 新增返回原始概率


# ====================== 4. 主函数：视频手势识别（新增视频级别统计） ======================
def video_gesture_recognition(
        video_path,
        model_path,
        train_npy_path,
        yolo_model_name="yolov8l-pose.pt",
        save_output_video=True
):
    # 1. 基础配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    label_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    input_dim = 24
    num_classes = 8

    # 2. 加载模型
    print(f"📌 加载YOLO Pose模型: {yolo_model_name}")
    yolo_infer = YOLOPoseInference(model_name=yolo_model_name, device=DEVICE)

    print(f"📌 加载训练好的MLP模型: {model_path}")
    model = FrameLevelMLP(input_dim=input_dim, hidden_dims=[512, 256, 128], num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    print(f"📌 加载训练数据的标准化参数: {train_npy_path}")
    scaler = load_scaler_from_train_data(train_npy_path)

    # 3. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ 无法打开视频：{video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 视频信息：FPS={fps}, 分辨率={width}x{height}, 总帧数={total_frames}")

    # 4. 准备保存视频
    output_video_path = None
    out = None
    if save_output_video and video_path != 0:
        output_video_path = f"{os.path.splitext(video_path)[0]}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"📽️ 识别结果将保存到：{output_video_path}")

    # 5. 新增：统计所有帧的预测结果
    frame_predictions = []  # 保存每帧的预测类别
    frame_confidences = []  # 保存每帧的置信度
    frame_probs = []  # 保存每帧的概率分布

    # 6. 逐帧处理
    frame_count = 0
    print("\n🚀 开始识别（按Q退出）...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # a. 提取特征
        frame_feature = yolo_infer.extract_single_frame_feature(frame)

        # b. 预测（新增返回概率）
        pred_label, pred_conf, pred_prob = predict_gesture(model, scaler, frame_feature, label_to_idx, DEVICE)

        # c. 保存帧预测结果（用于后续统计）
        frame_predictions.append(pred_label)
        frame_confidences.append(pred_conf)
        frame_probs.append(pred_prob.cpu().numpy()[0])

        # d. 可视化
        text_gesture = f"Gesture: {pred_label} (Conf: {pred_conf:.2f})"
        cv2.putText(frame, text_gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        text_frame = f"Frame: {frame_count} / {total_frames}" if video_path != 0 else f"Real-time Frame: {frame_count}"
        cv2.putText(frame, text_frame, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # e. 显示/保存
        cv2.imshow("YOLO + MLP 手势识别", frame)
        if out is not None:
            out.write(frame)

        # f. 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0 and video_path != 0:
            print(f"进度：{frame_count}/{total_frames} 帧")

    # 7. 释放资源
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # 8. 新增：视频级别分类统计（核心！）
    print(f"\n{'=' * 60}")
    print("📊 视频级别分类结果")
    print(f"{'=' * 60}")
    if len(frame_predictions) == 0:
        print("❌ 无有效预测结果")
    else:
        # 方式1：投票法（最直观，取出现次数最多的类别）
        pred_counter = Counter(frame_predictions)
        video_final_label = pred_counter.most_common(1)[0][0]  # 最终类别
        video_label_count = pred_counter.most_common(1)[0][1]  # 该类别帧数
        video_label_ratio = video_label_count / len(frame_predictions)  # 占比

        # 方式2：概率平均法（更精准，所有帧概率平均后取最大）
        avg_probs = np.mean(frame_probs, axis=0)
        avg_prob_label_idx = np.argmax(avg_probs)
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        avg_prob_label = idx_to_label[avg_prob_label_idx]
        avg_prob_conf = avg_probs[avg_prob_label_idx]

        # 输出统计结果
        print(f"📽️ 视频文件：{os.path.basename(video_path)}")
        print(f"🔢 总有效帧数：{len(frame_predictions)}")
        print(f"\n【投票法结果】")
        print(f"  最终类别：手势 {video_final_label}")
        print(f"  该类别帧数：{video_label_count} 帧")
        print(f"  该类别占比：{video_label_ratio:.2%}")
        print(f"\n【概率平均法结果】")
        print(f"  最终类别：手势 {avg_prob_label}")
        print(f"  平均置信度：{avg_prob_conf:.4f}")
        print(f"\n📋 所有类别分布：")
        for label, count in pred_counter.most_common():
            ratio = count / len(frame_predictions)
            print(f"  手势 {label}：{count} 帧 ({ratio:.2%})")

    print(f"\n✅ 识别完成！")
    if output_video_path:
        print(f"📁 结果视频：{output_video_path}")


# ====================== 5. 运行入口 ======================
if __name__ == "__main__":
    # 替换为你的实际路径！
    VIDEO_PATH = r"E:\最新的毕业设计\毕业设计\accset\cut_videos\all_continuous_values1\gesture_2_frames_2110-2145.mp4"
    MODEL_PATH = r"./models/frame_level_mlp_best.pth"
    TRAIN_NPY_PATH = r"E:\最新的毕业设计\毕业设计\script\try00\asset\JSONFOLDER\short_video_pose_features_20260227_133703.npy"
    YOLO_MODEL_NAME = "yolov8l-pose.pt"

    # 执行视频识别
    video_gesture_recognition(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        train_npy_path=TRAIN_NPY_PATH,
        yolo_model_name=YOLO_MODEL_NAME,
        save_output_video=True
    )