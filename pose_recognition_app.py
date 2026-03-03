"""
人体关键点识别网页应用

功能：
1. 使用YOLO从视频/摄像头提取人体关键点
2. 对关键点进行预处理（归一化+角度特征）
3. 使用训练好的LSTM模型进行value值识别（value_1到value_8）
4. 实时显示识别结果
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import json
import os
import sys
import glob
from typing import List, Dict, Optional
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# import sql

# 导入预处理函数（如果模块存在）
try:
    from pose_data_preprocessing import normalize_coordinates, extract_angle_features
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    print("⚠ 警告: pose_data_preprocessing模块未找到，将使用简化处理")
    
    # 简化的归一化和角度提取函数
    def normalize_coordinates(keypoints, method="center"):
        """简化版归一化"""
        if not keypoints:
            return []
        # 简单归一化：使用第一个关键点作为参考
        if len(keypoints) > 0:
            ref_x = keypoints[0].get("x", 0)
            ref_y = keypoints[0].get("y", 0)
            normalized = []
            for kp in keypoints:
                normalized.append({
                    "name": kp.get("name", ""),
                    "x": kp.get("x", 0) - ref_x,
                    "y": kp.get("y", 0) - ref_y,
                    "confidence": kp.get("confidence", 0)
                })
            return normalized
        return keypoints
    
    def extract_angle_features(keypoints):
        """简化版角度提取"""
        # 返回默认角度特征
        return {
            "left_arm_angle": (0.0, 1.0),
            "right_arm_angle": (0.0, 1.0),
            "left_leg_angle": (0.0, 1.0),
            "right_leg_angle": (0.0, 1.0),
            "left_torso_angle": (0.0, 1.0),
            "right_torso_angle": (0.0, 1.0),
            "head_angle": (0.0, 1.0)
        }

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 全局变量
yolo_model = None
lstm_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
is_processing = False
current_frame = None
frame_lock = None
import threading
frame_lock = threading.Lock()

# 关键点名称
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

VALUE_NAME_MAP = {
    1: 'Stop',  # 停止
    2: 'Forward',         # 直行
    3: 'Left Turn',          # 左转弯
    4: 'Left Turn Waiting',               # 左转弯待转
    5: 'Right Turn',          # 右转弯
    6: 'Lane Change',           # 变道
    7: 'Slow-down',      # 减速慢行
    8: 'Pull-over'             # 车辆靠边停车
}

# 导入LSTM模型类
from video_lstm_train import PoseLSTM


def init_models():
    """初始化YOLO和LSTM模型"""
    global yolo_model, lstm_model, device
    
    print("正在初始化模型...")
    
    # 1. 初始化YOLO模型
    try:
        yolo_model = YOLO("yolov8n-pose.pt")
        if device == 'cuda':
            yolo_model.to(device)
        print(f"✓ YOLO模型加载成功，使用设备: {device}")
    except Exception as e:
        print(f"✗ YOLO模型加载失败: {str(e)}")
        return False
    
    # 2. 加载LSTM模型
    try:
        # 从数据库获取模型路径
        # MODEL_PATH = sql.name_at_address(table_name="user", list_col="address", target_name="MODEL_SAVE_PATH")
        MODEL_PATH = "./models/pose_lstm_final.pth"
        if not MODEL_PATH or MODEL_PATH == "":
            MODEL_PATH = "../models/pose_lstm_best.pth"
        else:
            MODEL_PATH = MODEL_PATH + ".pth"
        
        if not os.path.exists(MODEL_PATH):
            print(f"✗ 模型文件不存在: {MODEL_PATH}")
            return False
        
        # 从模型文件读取参数
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        input_dim = checkpoint.get('input_dim', 65)  # 默认65（处理后数据）
        num_classes = checkpoint.get('num_classes', 8)
        hidden_dim = 128  # 默认值，可以从checkpoint读取
        num_layers = 2    # 默认值
        
        # 创建LSTM模型
        lstm_model = PoseLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # 加载权重
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.to(device)
        lstm_model.eval()
        
        print(f"✓ LSTM模型加载成功")
        print(f"  输入维度: {input_dim}")
        print(f"  类别数: {num_classes}")
        print(f"  模型路径: {MODEL_PATH}")
        
        return True
        
    except Exception as e:
        print(f"✗ LSTM模型加载失败: {str(e)}")
        return False


def extract_pose_features(frame, normalize_method="center"):
    """
    从单帧图像中提取姿态特征
    :param frame: 视频帧
    :param normalize_method: 归一化方法
    :return: 特征向量（65维：51关键点 + 14角度）
    """
    global yolo_model
    
    if yolo_model is None:
        return None
    
    # 使用YOLO检测姿态
    results = yolo_model(frame, conf=0.5, verbose=False)
    
    # 提取关键点
    keypoints_list = []
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # 只处理第一个检测到的人
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        
        for kp_idx, (x, y, conf) in enumerate(keypoints):
            if kp_idx < len(KEYPOINT_NAMES):
                keypoints_list.append({
                    "name": KEYPOINT_NAMES[kp_idx],
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf)
                })
    
    if len(keypoints_list) == 0:
        return None
    
    # 归一化坐标
    normalized_kps = normalize_coordinates(keypoints_list, method=normalize_method)
    
    # 提取角度特征
    angle_features = extract_angle_features(normalized_kps)
    
    # 构建特征向量
    feature_vector = []
    
    # 1. 添加归一化后的关键点坐标（x, y, confidence）
    for kp in normalized_kps:
        feature_vector.append(kp.get("x", 0.0))
        feature_vector.append(kp.get("y", 0.0))
        feature_vector.append(kp.get("confidence", 0.0))
    
    # 2. 添加角度特征（sin和cos）
    angle_names = ["left_arm_angle", "right_arm_angle", 
                   "left_leg_angle", "right_leg_angle",
                   "left_torso_angle", "right_torso_angle", "head_angle"]
    for angle_name in angle_names:
        angle_data = angle_features.get(angle_name, (0.0, 1.0))
        feature_vector.append(angle_data[0])  # sin
        feature_vector.append(angle_data[1])  # cos
    
    return np.array(feature_vector, dtype=np.float32)


def predict_value(frame_buffer: List[np.ndarray]) -> Optional[Dict]:
    """
    使用LSTM模型预测value值
    :param frame_buffer: 30帧的特征向量列表
    :return: 预测结果字典
    """
    global lstm_model, device
    
    if lstm_model is None or len(frame_buffer) != 30:
        return None
    
    try:
        # 转换为张量
        features = torch.FloatTensor(np.array(frame_buffer)).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = lstm_model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # 获取所有类别的概率
            probs = probabilities[0].cpu().numpy()
            
            predicted_class = predicted.item()
            confidence = float(probs[predicted_class])
            
            # 构建结果
            result = {
                "predicted_value": predicted_class + 1,  # 映射回value_1到value_8
                "confidence": confidence,
                "all_probabilities": {f"value_{i+1}": float(probs[i]) for i in range(len(probs))}
            }
            
            return result
            
    except Exception as e:
        print(f"预测错误: {str(e)}")
        return None


def generate_frames():
    """生成视频流（带识别结果）"""
    global is_processing, current_frame, frame_lock, yolo_model
    
    # 帧缓冲区（存储30帧的特征）
    frame_buffer = []
    prediction_result = None
    frame_count = 0
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("无法打开摄像头")
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while is_processing:
        ret, frame = camera.read()
        if not ret:
            break
        
        # 水平翻转
        frame = cv2.flip(frame, 1)
        
        # 提取特征
        features = extract_pose_features(frame)
        
        if features is not None:
            # 添加到缓冲区
            frame_buffer.append(features)
            
            # 保持缓冲区大小为30帧
            if len(frame_buffer) > 30:
                frame_buffer.pop(0)
            
            # 当缓冲区有30帧时，进行预测
            if len(frame_buffer) == 30:
                prediction_result = predict_value(frame_buffer)
                frame_count = 0  # 重置计数器
        
        # 绘制YOLO检测结果
        if yolo_model:
            results = yolo_model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()
        
        # 显示预测结果
        if prediction_result:
            value = prediction_result["predicted_value"]
            confidence = prediction_result["confidence"]
            
            # 显示结果
            # text = f"Value: {value} (confidence: {confidence*100:.1f}%)"
            value_text = VALUE_NAME_MAP.get(value, f"Value {value}")
            text = f"{value_text} (accuray: {confidence*100:.1f}%)"


            cv2.putText(annotated_frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示所有类别的概率（前3个）
            sorted_probs = sorted(prediction_result["all_probabilities"].items(), 
                                 key=lambda x: x[1], reverse=True)
            y_offset = 70
            for i, (value_name, prob) in enumerate(sorted_probs[:3]):
                prob_text = f"{value_name}: {prob*100:.1f}%"
                cv2.putText(annotated_frame, prob_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30
        
        # 显示帧数信息
        frame_count += 1
        info_text = f"fps: {frame_count} | buffer: {len(frame_buffer)}/30"
        cv2.putText(annotated_frame, info_text, (10, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()


@app.route('/')
def index():
    """主页面"""
    return render_template('pose_recognition.html')


@app.route('/start', methods=['POST'])
def start_recognition():
    """开始识别"""
    global is_processing
    
    if is_processing:
        return jsonify({'status': 'error', 'message': '识别已在运行中'})
    
    if yolo_model is None or lstm_model is None:
        return jsonify({'status': 'error', 'message': '模型未加载，请检查模型文件'})
    
    try:
        is_processing = True
        return jsonify({
            'status': 'success',
            'message': '识别已开始',
            'device': device
        })
    except Exception as e:
        is_processing = False
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/stop', methods=['POST'])
def stop_recognition():
    """停止识别"""
    global is_processing
    
    is_processing = False
    return jsonify({'status': 'success', 'message': '识别已停止'})


@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """对单帧图像进行预测（用于上传图片）"""
    global lstm_model
    
    if lstm_model is None:
        return jsonify({'status': 'error', 'message': '模型未加载'})
    
    try:
        # 这里可以处理上传的图片
        # 暂时返回示例
        return jsonify({
            'status': 'success',
            'message': '单帧预测功能待实现'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/status', methods=['GET'])
def get_status():
    """获取当前状态"""
    global is_processing, yolo_model, lstm_model, device
    
    return jsonify({
        'is_processing': is_processing,
        'yolo_loaded': yolo_model is not None,
        'lstm_loaded': lstm_model is not None,
        'device': device,
        'cuda_available': torch.cuda.is_available()
    })


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'message': '服务运行正常',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # 初始化模型
    if not init_models():
        print("⚠ 模型初始化失败，请检查模型文件")
        print("  程序将继续运行，但识别功能可能无法使用")
    
    # 启动Flask应用
    print("\n" + "="*60)
    print("🚀 Flask应用启动中...")
    print("="*60)
    print(f"📡 本地访问: http://localhost:5000")
    print(f"📡 网络访问: http://0.0.0.0:5000")
    print(f"📡 局域网访问: http://[你的IP地址]:5000")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"✗ 启动失败: {str(e)}")
        print("  请检查端口5000是否被占用")
        print("  可以尝试修改端口号或关闭占用端口的程序")

