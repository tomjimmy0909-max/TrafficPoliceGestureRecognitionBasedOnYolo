# TrafficPoliceGestureRecognitionBasedOnYolo

视频姿态提取：基于 YOLO 目标检测与姿态估计算法，从视频中提取人体关键点序列
数据预处理：支持对原始姿态数据进行清洗、归一化、时序对齐等操作
多模型训练：提供 YOLO、LSTM、MLP 等多种模型的训练脚本，适配不同任务需求
实时识别应用：可部署为独立应用，实现实时姿态识别与动作预测
视频标注切割：支持根据 JSON 标注文件对视频进行自动化切割与数据生成
🛠️ 技术栈
深度学习框架：PyTorch
计算机视觉：YOLO, OpenCV, MediaPipe
时序建模：LSTM, MLP
数据处理：NumPy, Pandas
部署与应用：Flask , Python


🚀 快速开始
1. 克隆仓库
bash
运行
git clone https://github.com/tomjimmy0909-max/your-repo-name.git
cd your-repo-name
2. 安装依赖
bash
运行
pip install -r requirements.txt
3. 运行示例
姿态识别应用
bash
运行
python pose_recognition_app.py
模型训练
bash
运行
# 训练YOLO模型
python video_YOLO_train.py

# 训练LSTM姿态预测模型
python video_lstm_train.py

# 训练MLP模型
python video_MLP_train.py
数据预处理
bash
运行
# 通用数据预处理
python dataPreprocessing.py

# 姿态数据预处理
python pose_data_preprocessing.py
视频切割
bash
运行
python cut_video_by_json.py --input your_video.mp4 --annotations annotations.json
📁 项目结构
plaintext
├── templates/              # 前端模板或配置文件（如Flask应用模板）
├── __init__.py             # Python包初始化文件
├── cut_video_by_json.py    # 基于JSON标注切割视频，生成训练数据
├── dataPreprocessing.py    # 通用数据预处理脚本（清洗、归一化等）
├── pose_data_preprocessing.py # 姿态数据专用预处理（时序对齐、特征提取）
├── pose_lstm_predict.py    # LSTM模型推理脚本，实现姿态序列预测
├── pose_recognition_app.py # 姿态识别应用主入口，支持实时/离线识别
├── process_csv_continuous_values.py # CSV连续值处理脚本
├── recognition.py          # 核心识别逻辑封装（姿态检测、动作分类）
├── requirements.txt        # 项目依赖清单
├── video_MLP_train.py      # MLP模型训练脚本（适用于动作分类等任务）
├── video_YOLO_train.py     # YOLO模型训练脚本（目标检测与姿态提取）
├── video_lstm_train.py     # LSTM模型训练脚本（时序动作预测）
└── README.md               # 项目说明文档
📊 预期效果
姿态提取准确率：在标准数据集上达到 95%+ 的关键点检测精度
动作识别准确率：基于 LSTM 的时序模型在自定义动作数据集
实时性能：在普通消费级 GPU 上可实现 30+ FPS 的实时姿态识别
