"""
视频LSTM训练脚本 - 使用YOLO提取的人体关键点数据

功能：
1. 从保存的骨骼信息数据（NPY文件）中加载特征
2. 使用LSTM模型对人体关键点序列进行训练
3. 生成智能模型用于人体姿势识别/分类

数据来源：
- 由video_YOLO_train.py生成的pose_features_*.npy文件
- 特征包含：手部、腿部等关键点的坐标和置信度
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import sys
import glob
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# import sql
from tqdm import tqdm


class PoseDataLoader:
    """骨骼信息数据加载器 - 从NPY文件加载YOLO提取的骨骼特征"""
    
    def __init__(self, data_dir: str = "."):
        """
        初始化数据加载器
        :param data_dir: 数据文件所在目录（默认当前目录）
        """
        self.data_dir = r"E:\最新的毕业设计\毕业设计\script\try00\asset\JSONFOLDER"
    
    def find_latest_npy_file(self, prefer_processed: bool = True) -> Optional[str]:
        """
        查找最新的NPY特征文件
        :param prefer_processed: 是否优先查找处理后的数据
        :return: 文件路径，如果未找到返回None
        """
        # ===== 新增调试信息 =====
        print(f"\n【调试】当前查找目录: {os.path.abspath(self.data_dir)}")
        print(f"【调试】优先查找处理后文件: {prefer_processed}")
        if prefer_processed:
            # 优先查找处理后的数据
            processed_pattern = os.path.join(self.data_dir, "pose_features_processed_*.npy")
            processed_files = glob.glob(processed_pattern)
            if processed_files:
                latest_file = max(processed_files, key=os.path.getctime)
                print(f"  找到处理后的数据文件: {os.path.basename(latest_file)}")
                return latest_file
        
        # 如果找不到处理后的数据，查找原始数据
        pattern = os.path.join(self.data_dir, "pose_features_*.npy")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # 返回最新的文件
        latest_file = max(files, key=os.path.getctime)
        if not prefer_processed or "processed" not in latest_file:
            print(f"  找到原始数据文件: {os.path.basename(latest_file)}")
        return latest_file
    
    def find_latest_json_file(self, prefer_processed: bool = True) -> Optional[str]:
        """
        查找最新的JSON数据文件
        :param prefer_processed: 是否优先查找处理后的数据
        :return: 文件路径，如果未找到返回None
        """
        if prefer_processed:
            # 优先查找处理后的数据
            processed_pattern = os.path.join(self.data_dir, "pose_data_processed_*.json")
            processed_files = glob.glob(processed_pattern)
            if processed_files:
                latest_file = max(processed_files, key=os.path.getctime)
                print(f"  找到处理后的JSON文件: {os.path.basename(latest_file)}")
                return latest_file
        
        # 如果找不到处理后的数据，查找原始数据
        pattern = os.path.join(self.data_dir, "pose_data_*.json")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # 返回最新的文件
        latest_file = max(files, key=os.path.getctime)
        if not prefer_processed or "processed" not in latest_file:
            print(f"  找到原始JSON文件: {os.path.basename(latest_file)}")
        return latest_file
    
    def extract_labels_from_json(self, json_path: str) -> List[int]:
        """
        从JSON文件中提取类别标签
        :param json_path: JSON文件路径
        :return: 标签列表，每个标签对应一个segment
        """
        import re
        
        print(f"从JSON文件提取标签: {os.path.basename(json_path)}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labels = []
            segments = data.get("segments", [])
            
            for segment in segments:
                video_file = segment.get("video_file", "")
                
                # 从video_file中提取value_X，例如: "002_value_0.mp4" -> 0
                match = re.search(r'gesture[_ ]*(\d+)', video_file, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
                    labels.append(value)
                else:
                    # 如果无法提取，默认设为0（无效类别）
                    print(f"  ⚠ 警告: 无法从 '{video_file}' 提取类别，设为0（无效）")
                    labels.append(0)
            
            print(f"✓ 成功提取 {len(labels)} 个标签")
            
            # 统计各类别数量
            from collections import Counter
            label_counts = Counter(labels)
            print(f"  类别分布:")
            for label in sorted(label_counts.keys()):
                print(f"    value_{label}: {label_counts[label]} 个样本")
            
            return labels
            
        except Exception as e:
            print(f"✗ 加载JSON文件失败: {str(e)}")
            return []
    
    def load_features(self, npy_path: Optional[str] = None, prefer_processed: bool = True) -> Tuple[np.ndarray, dict, List[int]]:
        """
        加载特征数据和标签
        :param npy_path: NPY文件路径，如果为None则自动查找最新文件
        :param prefer_processed: 是否优先加载处理后的数据
        :return: (特征数组, 元数据字典, 标签列表)
        """
        if npy_path is None:
            npy_path = self.find_latest_npy_file(prefer_processed=prefer_processed)
        
        if npy_path is None or not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"未找到骨骼特征文件。\n"
                f"请先运行 video_YOLO_train.py 提取骨骼信息，或运行 pose_data_preprocessing.py 处理数据。\n"
                f"查找路径: {os.path.abspath(self.data_dir)}"
            )
        
        is_processed = "processed" in npy_path
        if is_processed:
            print(f"✓ 加载处理后的特征文件: {os.path.basename(npy_path)}")
        else:
            print(f"✓ 加载原始特征文件: {os.path.basename(npy_path)}")
        
        features = np.load(npy_path)
        print(f"✓ 特征加载成功，形状: {features.shape}")
        
        # 尝试加载元数据
        metadata = {}
        if is_processed:
            metadata_pattern = os.path.join(self.data_dir, "pose_metadata_processed_*.json")
        else:
            metadata_pattern = os.path.join(self.data_dir, "pose_metadata_*.json")
        
        metadata_files = glob.glob(metadata_pattern)
        if metadata_files:
            latest_metadata = max(metadata_files, key=os.path.getctime)
            try:
                with open(latest_metadata, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"✓ 元数据加载成功: {os.path.basename(latest_metadata)}")
                if "normalize_method" in metadata:
                    print(f"  归一化方法: {metadata['normalize_method']}")
            except Exception as e:
                print(f"⚠ 加载元数据失败: {str(e)}")
        
        # 从JSON文件提取标签
        json_path = self.find_latest_json_file(prefer_processed=prefer_processed)
        if json_path:
            labels = self.extract_labels_from_json(json_path)
            
            # 确保标签数量与特征数量匹配
            if len(labels) != len(features):
                print(f"⚠ 警告: 标签数量({len(labels)})与特征数量({len(features)})不匹配")
                # 如果标签数量少于特征数量，用0填充
                if len(labels) < len(features):
                    labels.extend([0] * (len(features) - len(labels)))
                    print(f"  已用0填充标签到 {len(labels)} 个")
                else:
                    labels = labels[:len(features)]
                    print(f"  已截断标签到 {len(labels)} 个")
        else:
            print(f"⚠ 警告: 未找到JSON数据文件，将使用伪标签")
            labels = [i % 9 for i in range(len(features))]  # 伪标签（0-8）
        
        return features, metadata, labels


class PoseLSTMDataset(Dataset):
    """LSTM训练数据集 - 用于人体关键点序列"""
    
    def __init__(self, features: List[np.ndarray], labels: Optional[List[int]] = None):
        """
        初始化数据集
        :param features: 特征列表，每个元素形状为 (30, feature_dim)
        :param labels: 标签列表（可选，用于监督学习）
        """
        self.features = features
        self.labels = labels if labels is not None else [0] * len(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label


class PoseLSTM(nn.Module):
    """人体关键点LSTM模型 - 用于姿势识别/分类"""
    
    def __init__(self, input_dim: int = 24, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        """
        初始化LSTM模型
        :param input_dim: 输入特征维度（关键点数量 × 3，例如手部+腿部=8×3=24）
        :param hidden_dim: LSTM隐藏层维度
        :param num_layers: LSTM层数
        :param num_classes: 分类类别数
        :param dropout: Dropout比率
        """
        super(PoseLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制（可选，提高模型性能）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim*2,
            num_heads=4,
            batch_first=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状 (batch_size, seq_length, input_dim)
        :return: 输出张量，形状 (batch_size, num_classes)
        """
        x = self.norm(x)
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用最后一个时间步的输出
        # last_output = attn_out[:, -1, :]
        h_n = torch.cat((h_n[-2,:,:],h_n[-1,:,:]),dim=1)
        last_output = h_n
        # 全连接层
        output = self.dropout(last_output)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output


def get_unique_save_path(base_path: str) -> str:
    """
    获取唯一的保存路径，如果文件已存在则添加序号后缀
    自动处理带.pth后缀的输入路径，避免生成重复后缀
    :param base_path: 基础保存路径（可带/不带.pth后缀）
    :return: 唯一的保存路径（不含.pth后缀）
    """
    # 第一步：剥离已有的.pth后缀（如果存在）
    if base_path.endswith('.pth'):
        base_path = base_path[:-4]  # 去掉最后4个字符（.pth）

    # 第二步：检查基础路径是否存在，不存在则直接返回
    if not os.path.exists(f"{base_path}.pth"):
        return base_path

    # 第三步：如果存在，查找最大序号并+1
    idx = 1
    while os.path.exists(f"{base_path}_{idx}.pth"):
        idx += 1

    return f"{base_path}_{idx}"


def train_lstm_model(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model: nn.Module,
    num_epochs: int = 50,
    learning_rate: float = 0.0005,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: Optional[str] = None
):
    """
    训练LSTM模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器（可选）
    :param model: LSTM模型
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :param device: 训练设备
    :param save_path: 模型保存路径
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    print(f"\n开始训练，使用设备: {device}")
    print(f"训练轮数: {num_epochs}, 学习率: {learning_rate}")
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} [Train] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} [Val] - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
            
            scheduler.step(avg_val_loss)
        else:
            scheduler.step(avg_loss)
        
    # 只保存最终模型（不保存检查点和最佳模型）
    if save_path:
        unique_save_path = get_unique_save_path(save_path)
        save_dict = {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1],
            'accuracy': train_accuracies[-1],
            'num_classes': model.fc2.out_features,  # 保存类别数
            'input_dim': model.lstm.input_size,  # 保存输入维度
            'train_losses': train_losses,  # 保存训练损失历史
            'train_accuracies': train_accuracies,  # 保存训练准确率历史
        }
        
        # 如果有验证数据，也保存验证历史
        if val_losses and len(val_losses) > 0:
            save_dict['val_losses'] = val_losses
            save_dict['val_accuracies'] = val_accuracies
        
        torch.save(save_dict, f"{unique_save_path}.pth")
        print(f"\n✓ 模型已保存: {unique_save_path}.pth")
        print(f"  训练历史已包含在模型中")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def filter_invalid_data(features: List[np.ndarray], labels: List[int], invalid_class: int = 0):
    """
    过滤掉无效数据（类别为0的数据）
    :param features: 特征列表
    :param labels: 标签列表
    :param invalid_class: 无效类别标签（默认为0）
    :return: (过滤后的特征列表, 过滤后的标签列表)
    """
    valid_features = []
    valid_labels = []
    
    for i, label in enumerate(labels):
        if label != invalid_class:
            valid_features.append(features[i])
            valid_labels.append(label)
    
    print(f"  原始数据: {len(features)} 个样本")
    print(f"  过滤后数据: {len(valid_features)} 个样本")
    print(f"  剔除无效数据: {len(features) - len(valid_features)} 个样本（类别={invalid_class}）")
    
    return valid_features, valid_labels


def split_data(features: List[np.ndarray], labels: List[int], train_ratio: float = 0.8):
    """
    划分训练集和验证集
    :param features: 特征列表
    :param labels: 标签列表
    :param train_ratio: 训练集比例
    :return: (train_features, train_labels, val_features, val_labels)
    """
    # 转换为numpy数组以便打乱
    features_array = np.array([f for f in features], dtype=object)
    labels_array = np.array(labels)
    
    indices = np.random.permutation(len(features))
    split_idx = int(len(features) * train_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_features = [features[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_features, train_labels, val_features, val_labels


def main():
    """主函数"""
    print("="*60)
    print("人体关键点LSTM智能模型训练")
    print("="*60)
    
    # 数据目录（当前目录，与video_YOLO_train.py输出位置一致）
    DATA_DIR = r"E:\最新的毕业设计\毕业设计\script\try00\asset\JSONFOLDER"
    
    # 从数据库获取配置
    try:
        # MODEL_SAVE_PATH = sql.name_at_address(table_name="user", list_col="address", target_name="MODEL_SAVE_PATH")
        MODEL_SAVE_PATH = "./models/pose_lstm_final.pth"
        if not MODEL_SAVE_PATH or MODEL_SAVE_PATH == "":
            MODEL_SAVE_PATH = "./models/pose_lstm"
    except:
        MODEL_SAVE_PATH = "./models/pose_lstm"
    
    # 确保模型保存目录存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) if os.path.dirname(MODEL_SAVE_PATH) else ".", exist_ok=True)
    
    print(f"\n配置信息:")
    print(f"  数据目录: {os.path.abspath(DATA_DIR)}")
    print(f"  模型保存路径: {os.path.abspath(MODEL_SAVE_PATH)}")
    
    # 参数配置
    HIDDEN_DIM = 256     # LSTM隐藏层维度
    NUM_LAYERS = 2       # LSTM层数
    NUM_CLASSES = 8      # 分类类别数（value_1到value_8，共8个类别，value_0会被过滤）
    BATCH_SIZE = 8      # 批次大小
    NUM_EPOCHS = 50      # 训练轮数
    LEARNING_RATE = 0.001 # 学习率
    
    print(f"\n训练参数:")
    print(f"  LSTM隐藏层: {HIDDEN_DIM}")
    print(f"  LSTM层数: {NUM_LAYERS}")
    print(f"  分类类别数: {NUM_CLASSES} (value_1到value_8，value_0将被过滤)")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    
    # 1. 加载骨骼特征数据
    print(f"\n{'='*60}")
    print("步骤1: 加载骨骼特征数据")
    print(f"{'='*60}")
    data_loader = PoseDataLoader(data_dir=DATA_DIR)
    
    try:
        features, metadata, labels = data_loader.load_features()
        
        # 从元数据或特征形状中获取特征维度
        if "feature_dim" in metadata:
            FEATURE_DIM = metadata["feature_dim"]
        else:
            # 从特征数组形状推断
            if len(features.shape) == 3:
                FEATURE_DIM = features.shape[2]  # (segments, frames, features)
            else:
                FEATURE_DIM = features.shape[1]  # (segments, features)
        
        print(f"✓ 特征维度: {FEATURE_DIM}")
        print(f"  特征数组形状: {features.shape}")
        print(f"  说明: (片段数, 帧数, 特征维度)")
        print(f"✓ 标签数量: {len(labels)}")
        
    except FileNotFoundError as e:
        print(f"✗ {str(e)}")
        print(f"\n请先运行 video_YOLO_train.py 提取骨骼信息数据")
        return
    except Exception as e:
        print(f"✗ 加载数据失败: {str(e)}")
        return
    
    # 2. 准备数据集
    print(f"\n{'='*60}")
    print("步骤2: 准备训练数据")
    print(f"{'='*60}")
    
    # 将numpy数组转换为列表（每个元素是一个30帧的序列）
    features_list = [features[i] for i in range(len(features))]
    
    print(f"✓ 原始数据集大小: {len(features_list)}")
    print(f"  每个样本形状: {features_list[0].shape}")
    print(f"  标签数量: {len(labels)}")
    print(f"  ✓ 标签已从JSON文件的video_file字段提取")
    
    # 过滤掉类别为0（value_0）的无效数据
    print(f"\n过滤无效数据（value_0，类别=0）...")
    valid_features, valid_labels = filter_invalid_data(features_list, labels, invalid_class=0)
    
    if len(valid_features) == 0:
        print("✗ 错误: 过滤后没有有效数据，请检查标签数据")
        return
    
    # 重新映射标签：value_1到value_8映射为类别0到7
    # 原始标签: [1, 2, 3, 4, 5, 6, 7, 8] -> 新标签: [0, 1, 2, 3, 4, 5, 6, 7]
    unique_labels = sorted(list(set(valid_labels)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped_labels = [label_mapping[label] for label in valid_labels]
    
    print(f"✓ 标签映射（value_X -> 类别索引）:")
    for old_label in sorted(label_mapping.keys()):
        new_label = label_mapping[old_label]
        print(f"  value_{old_label} -> 类别 {new_label}")
    
    # 更新类别数（排除类别0后，实际是value_1到value_8，共8个类别）
    actual_num_classes = len(unique_labels)
    if actual_num_classes != NUM_CLASSES:
        print(f"✓ 类别数已更新: {NUM_CLASSES} -> {actual_num_classes} (value_1到value_{max(unique_labels)})")
        NUM_CLASSES = actual_num_classes
    
    # 使用所有数据训练（不划分验证集）
    print(f"\n使用所有数据训练（不划分验证集）...")
    print(f"✓ 训练数据集大小: {len(valid_features)} 个样本")
    print(f"  所有数据将用于训练，不进行验证集划分")
    
    # 创建数据集和数据加载器（使用所有数据）
    train_dataset = PoseLSTMDataset(valid_features, remapped_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"✓ 数据加载器创建完成")
    print(f"  训练批次数量: {len(train_loader)}")
    
    # 3. 创建模型
    print(f"\n{'='*60}")
    print("步骤3: 创建LSTM模型")
    print(f"{'='*60}")
    print(f"  输入特征维度: {FEATURE_DIM}")
    print(f"  输出类别数: {NUM_CLASSES} (已排除无效类别0)")
    
    model = PoseLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型创建完成")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型结构:")
    print(model)
    
    # 4. 训练模型
    print(f"\n{'='*60}")
    print("步骤4: 开始训练（使用所有数据）")
    print(f"{'='*60}")
    history = train_lstm_model(
        train_loader=train_loader,
        val_loader=None,  # 不使用验证集，所有数据用于训练
        model=model,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_path=MODEL_SAVE_PATH
    )
    
    print(f"\n{'='*60}")
    print("✅ 训练完成！")
    print(f"{'='*60}")
    print(f"最终训练损失: {history['train_losses'][-1]:.4f}")
    print(f"最终训练准确率: {history['train_accuracies'][-1]:.2f}%")
    if history['val_accuracies']:
        print(f"最终验证准确率: {history['val_accuracies'][-1]:.2f}%")
    print(f"\n模型文件:")



if __name__ == '__main__':
    main()

