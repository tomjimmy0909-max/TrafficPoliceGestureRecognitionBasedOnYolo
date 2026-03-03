import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 新增：数据标准化
from collections import Counter
import os
import re


def reshape_data_to_frame_level(npy_path: str, json_path: str):
    """
    将时序数据重塑为帧级独立样本
    :param npy_path: 预处理后的NPY文件路径
    :param json_path: 预处理后的JSON文件路径
    :return: (帧级特征, 帧级标签)
    """
    # 1. 加载时序特征
    features = np.load(npy_path)  # (560, 30, 65)
    if len(features.shape) == 3:
        num_segments, num_frames, feature_dim = features.shape
    else:
        raise ValueError(f"特征形状异常: {features.shape}，应为 (片段数, 帧数, 特征维度)")

    print(f"原始时序特征形状: {features.shape}")
    print(f"自动识别特征维度： {feature_dim}")
    # 2. 重塑为帧级特征：(560,30,65) → (560×30,65) = (16800,65)
    frame_features = features.reshape(-1, feature_dim)
    print(f"重塑后帧级特征形状: {frame_features.shape}")

    # 3. 加载并扩展标签
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data.get("segments", [])

    # 提取片段级标签
    segment_labels = []
    for seg in segments:
        video_file = seg.get("video_file", "")
        match = re.search(r'gesture[_ ]*(\d+)', video_file, re.IGNORECASE)
        if match:
            segment_labels.append(int(match.group(1)))
        else:
            segment_labels.append(1)

    # 只保留特征对应的片段标签（前560个）
    segment_labels = segment_labels[:num_segments]

    # 扩展为帧级标签：每个片段的标签复制30次
    frame_labels = []
    for label in segment_labels:
        frame_labels.extend([label] * num_frames)
    frame_labels = np.array(frame_labels)

    print(f"片段级标签数: {len(segment_labels)}")
    print(f"扩展后帧级标签数: {len(frame_labels)}")
    print(f"帧级标签分布: {Counter(frame_labels)}")

    return frame_features, frame_labels


class FrameLevelMLP(nn.Module):
    """帧级全连接网络（优化模型结构）"""

    def __init__(self, input_dim=24, hidden_dims=[512, 256, 128], num_classes=8, dropout=0.1):  # 修改：增加隐藏层、降低dropout
        super().__init__()
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))  # 新增：批归一化
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))  # 新增：批归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FrameDataset(Dataset):
    """帧级数据集"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_frame_level_model():
    # 配置
    NPY_PATH = r"E:\最新的毕业设计\毕业设计\script\try00\asset\JSONFOLDER\short_video_pose_features_20260227_133703.npy"
    JSON_PATH = r"E:\最新的毕业设计\毕业设计\script\try00\asset\JSONFOLDER\short_video_pose_data_20260227_133703.json"
    BATCH_SIZE = 128  # 修改：增大批次大小
    EPOCHS = 50  # 修改：增加训练轮数（配合早停）
    LEARNING_RATE = 0.0005  # 修改：降低初始学习率
    WEIGHT_DECAY = 1e-4  # 修改：增大权重衰减
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 检查文件是否存在
    if not os.path.exists(NPY_PATH):
        raise FileNotFoundError(f"❌ NPY文件不存在: {NPY_PATH}")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"❌ JSON文件不存在: {JSON_PATH}")

    # 创建模型保存目录
    model_dir = "./models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"使用设备: {DEVICE}")

    # 1. 数据重塑
    print("\n" + "=" * 60)
    print("步骤1: 数据重塑（帧级独立样本）")
    print("=" * 60)
    frame_features, frame_labels = reshape_data_to_frame_level(NPY_PATH, JSON_PATH)

    # 2. 标签映射（value_X→0-7）
    valid_classes = [1, 2, 3, 4, 5, 6, 7, 8]
    label_to_idx = {l: i for i, l in enumerate(valid_classes)}
    # 过滤无效标签（防止key不存在报错）
    frame_labels = np.array([label_to_idx[l] if l in label_to_idx else 0 for l in frame_labels])
    num_classes = len(valid_classes)
    print(f"\n标签映射: {label_to_idx}")
    print(f"类别数: {num_classes}")

    # 3. 拆分训练/验证集（修改：分层抽样，保证标签分布一致）
    X_train, X_val, y_train, y_val = train_test_split(
        frame_features, frame_labels,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=frame_labels  # 新增：分层抽样
    )
    print(f"\n训练集: {len(X_train)} 个样本")
    print(f"验证集: {len(X_val)} 个样本")

    # 4. 数据标准化（新增：关键优化步骤）
    print("\n步骤1.5: 数据标准化")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # 训练集拟合+转换
    X_val = scaler.transform(X_val)  # 验证集仅转换（避免数据泄露）
    print("✅ 数据标准化完成")

    # 5. 创建数据加载器
    train_dataset = FrameDataset(X_train, y_train)
    val_dataset = FrameDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 创建模型
    print("\n" + "=" * 60)
    print("步骤2: 创建帧级MLP模型")
    print("=" * 60)
    model = FrameLevelMLP(
        input_dim=frame_features.shape[1],  # 动态获取特征维度
        hidden_dims=[512, 256, 128],  # 修改：更深的网络
        num_classes=num_classes,
        dropout=0.1  # 修改：降低dropout
    )
    model.to(DEVICE)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 7. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(  # 修改：使用AdamW优化器（效果更好）
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # 新增：学习率调度器（训练后期降低学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,

    )

    # 8. 训练循环（增加早停策略）
    print("\n" + "=" * 60)
    print("步骤3: 开始训练")
    print("=" * 60)
    best_val_acc = 0.0
    patience = 8  # 早停耐心值
    patience_counter = 0  # 早停计数器

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = 100. * train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        # 学习率调度
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率：{current_lr:.6f}")

        # 打印
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "frame_level_mlp_best.pth"))
            print(f"  ✓ 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            print(f"  ⚠️  验证准确率未提升，计数器: {patience_counter}/{patience}")

        # 早停判断
        if patience_counter >= patience:
            print(f"\n❌ 早停触发！已连续{patience}轮无提升，停止训练")
            break

    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    train_frame_level_model()