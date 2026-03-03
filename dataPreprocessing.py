from ultralytics import YOLO
import cv2

# 1. 加载预训练的 YOLOv8-Pose 模型（n/s/m/l/x 代表模型大小，n最快，x最准）
model = YOLO("yolov8n-pose.pt")  # 自动下载预训练权重，也可换 yolov8s-pose.pt

# 2. 配置视频输入（本地视频文件/摄像头（0）/网络流）
video_path = "input_video.mp4"  # 替换为你的视频路径，或改为 0 调用摄像头
cap = cv2.VideoCapture(video_path)

# 3. 获取视频参数（用于保存输出视频）
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 输出视频编码
out = cv2.VideoWriter("output_pose.mp4", fourcc, fps, (width, height))

# 4. 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕
    
    # 5. 对单帧执行姿态检测（conf=置信度阈值，iou=重叠框阈值）
    results = model(frame, conf=0.5, iou=0.7)
    
    # 6. 绘制姿态关键点/骨架到帧上（results[0].plot() 自动绘制）
    annotated_frame = results[0].plot()
    
    # 7. 保存标注后的帧到输出视频
    out.write(annotated_frame)
    
    # 8. 实时展示结果（可选，按 q 退出）
    cv2.imshow("YOLOv8 Pose Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 9. 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()