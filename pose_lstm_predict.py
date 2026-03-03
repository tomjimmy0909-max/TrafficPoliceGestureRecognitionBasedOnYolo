"""
人体关键点LSTM模型预测和评估脚本

功能：
1. 加载训练好的LSTM模型
2. 对数据进行预测
3. 生成混淆矩阵
4. 可视化准确率变化、损失曲线等
5. 生成详细的评估报告
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import sys
import glob
import json
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# import sql
from tqdm import tqdm
from datetime import datetime

# 设置中文字体（用于图表）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入LSTM模型类
from video_lstm_train import PoseLSTM, PoseLSTMDataset, PoseDataLoader


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化评估器
        :param model: 训练好的LSTM模型
        :param device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        对数据进行预测
        :param data_loader: 数据加载器
        :return: (预测标签, 真实标签)
        """
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(data_loader, desc="预测中"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 class_names: Optional[List[str]] = None) -> Dict:
        """
        评估模型性能
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :param class_names: 类别名称列表
        :return: 评估结果字典
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 计算宏平均和微平均
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                          save_path: str, title: str = "混淆矩阵"):
    """
    绘制混淆矩阵
    :param cm: 混淆矩阵
    :param class_names: 类别名称列表
    :param save_path: 保存路径
    :param title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 绘制热力图
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '百分比 (%)'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 混淆矩阵已保存: {save_path}")


def plot_accuracy_curve(train_accuracies: List[float], save_path: str,val_accuracies: Optional[List[float]] = None,
                        title: str = "准确率变化曲线"):
    """
    绘制准确率变化曲线
    :param train_accuracies: 训练准确率列表
    :param val_accuracies: 验证准确率列表（可选）
    :param save_path: 保存路径
    :param title: 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'b-', label='训练准确率', linewidth=2, marker='o', markersize=4)
    
    if val_accuracies and len(val_accuracies) > 0:
        plt.plot(epochs[:len(val_accuracies)], val_accuracies, 'r-', 
                label='验证准确率', linewidth=2, marker='s', markersize=4)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 准确率曲线已保存: {save_path}")


def plot_loss_curve(train_losses: List[float], save_path: str, val_losses: Optional[List[float]] = None,
                    title: str = "损失变化曲线"):
    """
    绘制损失变化曲线
    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表（可选）
    :param save_path: 保存路径
    :param title: 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
    
    if val_losses and len(val_losses) > 0:
        plt.plot(epochs[:len(val_losses)], val_losses, 'r-', 
                label='验证损失', linewidth=2, marker='s', markersize=4)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 损失曲线已保存: {save_path}")


def plot_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str], save_path: str):
    """
    绘制分类报告（精确率、召回率、F1分数）
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param class_names: 类别名称列表
    :param save_path: 保存路径
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('精确率 (Precision)', precision),
        ('召回率 (Recall)', recall),
        ('F1分数 (F1-Score)', f1)
    ]
    
    for idx, (metric_name, values) in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(class_names, values, color='steelblue', alpha=0.7)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('分数', fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 分类报告已保存: {save_path}")


def load_model(model_path: str, input_dim: int, hidden_dim: int, 
              num_layers: int, num_classes: int, device: str) -> nn.Module:
    """
    加载训练好的模型
    :param model_path: 模型文件路径
    :param input_dim: 输入特征维度
    :param hidden_dim: LSTM隐藏层维度
    :param num_layers: LSTM层数
    :param num_classes: 类别数
    :param device: 设备
    :return: 加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型
    model = PoseLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功: {os.path.basename(model_path)}")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'accuracy' in checkpoint:
        print(f"  训练准确率: {checkpoint['accuracy']:.2f}%")
    
    return model


def main():
    """主函数"""
    print("="*60)
    print("人体关键点LSTM模型预测和评估")
    print("="*60)
    
    # 数据目录
    DATA_DIR = "."
    
    # 从数据库获取配置
    try:
        MODEL_PATH = sql.name_at_address(table_name="user", list_col="address", target_name="MODEL_SAVE_PATH")
        if not MODEL_PATH or MODEL_PATH == "":
            MODEL_PATH = "./models/pose_lstm.pth"
        else:
            MODEL_PATH = MODEL_PATH + ".pth"
    except:
        MODEL_PATH = "./models/pose_lstm.pth"
    
    # 输出目录
    OUTPUT_DIR = "./evaluation_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n配置信息:")
    print(f"  数据目录: {os.path.abspath(DATA_DIR)}")
    print(f"  模型路径: {os.path.abspath(MODEL_PATH)}")
    print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}")
    
    # 模型参数（将从模型文件中读取，如果不存在则使用默认值）
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    BATCH_SIZE = 16
    
    # 尝试从模型文件中读取参数
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            if 'input_dim' in checkpoint:
                # 参数将从模型文件中读取
                print(f"  将从模型文件中读取参数")
        except:
            pass
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    # 1. 加载数据
    print(f"\n{'='*60}")
    print("步骤1: 加载数据")
    print(f"{'='*60}")
    data_loader = PoseDataLoader(data_dir=DATA_DIR)
    
    try:
        features, metadata, labels = data_loader.load_features(prefer_processed=True)
        
        # 获取特征维度
        if "feature_dim" in metadata:
            FEATURE_DIM = metadata["feature_dim"]
        else:
            if len(features.shape) == 3:
                FEATURE_DIM = features.shape[2]
            else:
                FEATURE_DIM = features.shape[1]
        
        print(f"✓ 特征维度: {FEATURE_DIM}")
        print(f"  特征数组形状: {features.shape}")
        print(f"✓ 标签数量: {len(labels)}")
        
    except FileNotFoundError as e:
        print(f"✗ {str(e)}")
        return
    except Exception as e:
        print(f"✗ 加载数据失败: {str(e)}")
        return
    
    # 2. 过滤无效数据并重新映射标签
    print(f"\n{'='*60}")
    print("步骤2: 准备数据")
    print(f"{'='*60}")
    
    features_list = [features[i] for i in range(len(features))]
    
    # 过滤类别0
    from video_lstm_train import filter_invalid_data
    valid_features, valid_labels = filter_invalid_data(features_list, labels, invalid_class=0)
    
    if len(valid_features) == 0:
        print("✗ 错误: 过滤后没有有效数据")
        return
    
    # 重新映射标签
    unique_labels = sorted(list(set(valid_labels)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped_labels = [label_mapping[label] for label in valid_labels]
    
    NUM_CLASSES = len(unique_labels)
    print(f"✓ 有效数据: {len(valid_features)} 个样本")
    print(f"✓ 类别数: {NUM_CLASSES}")
    
    # 创建类别名称
    class_names = [f"value_{old_label}" for old_label in sorted(label_mapping.keys())]
    
    # 创建数据集（使用所有数据进行预测）
    dataset = PoseLSTMDataset(valid_features, remapped_labels)
    data_loader_obj = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. 加载模型
    print(f"\n{'='*60}")
    print("步骤3: 加载模型")
    print(f"{'='*60}")
    
    # 尝试从模型文件中读取参数
    model_params = {}
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            if 'input_dim' in checkpoint:
                # 使用模型文件中的参数
                model_params['input_dim'] = checkpoint['input_dim']
                model_params['num_classes'] = checkpoint.get('num_classes', NUM_CLASSES)
                print(f"  从模型文件读取: input_dim={model_params['input_dim']}, num_classes={model_params['num_classes']}")
        except:
            pass
    
    # 如果模型文件中没有参数，使用当前数据推断的参数
    if 'input_dim' not in model_params:
        model_params['input_dim'] = FEATURE_DIM
        model_params['num_classes'] = NUM_CLASSES
    
    try:
        model = load_model(MODEL_PATH, model_params['input_dim'], HIDDEN_DIM, NUM_LAYERS, 
                          model_params['num_classes'], device)
        
        # 检查模型参数是否匹配
        if model_params['input_dim'] != FEATURE_DIM:
            print(f"  ⚠ 警告: 模型输入维度({model_params['input_dim']})与数据特征维度({FEATURE_DIM})不匹配")
        if model_params['num_classes'] != NUM_CLASSES:
            print(f"  ⚠ 警告: 模型类别数({model_params['num_classes']})与数据类别数({NUM_CLASSES})不匹配")
            NUM_CLASSES = model_params['num_classes']  # 使用模型中的类别数
            
    except Exception as e:
        print(f"✗ 加载模型失败: {str(e)}")
        return
    
    # 4. 进行预测
    print(f"\n{'='*60}")
    print("步骤4: 进行预测")
    print(f"{'='*60}")
    
    evaluator = ModelEvaluator(model, device)
    y_pred, y_true = evaluator.predict(data_loader_obj)
    
    print(f"✓ 预测完成")
    print(f"  预测样本数: {len(y_pred)}")
    
    # 5. 评估模型
    print(f"\n{'='*60}")
    print("步骤5: 评估模型性能")
    print(f"{'='*60}")
    
    results = evaluator.evaluate(y_true, y_pred, class_names)
    
    print(f"✓ 总体准确率: {results['accuracy']*100:.2f}%")
    print(f"✓ 宏平均精确率: {results['macro_precision']*100:.2f}%")
    print(f"✓ 宏平均召回率: {results['macro_recall']*100:.2f}%")
    print(f"✓ 宏平均F1分数: {results['macro_f1']*100:.2f}%")
    
    # 打印每个类别的详细指标
    print(f"\n各类别详细指标:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    精确率: {results['precision'][i]*100:.2f}%")
        print(f"    召回率: {results['recall'][i]*100:.2f}%")
        print(f"    F1分数: {results['f1_score'][i]*100:.2f}%")
        print(f"    样本数: {results['support'][i]}")
    
    # 6. 生成可视化图表
    print(f"\n{'='*60}")
    print("步骤6: 生成可视化图表")
    print(f"{'='*60}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 6.1 混淆矩阵
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
    # 6.2 分类报告
    report_path = os.path.join(OUTPUT_DIR, f"classification_report_{timestamp}.png")
    plot_classification_report(y_true, y_pred, class_names, report_path)
    
    # 6.3 加载训练历史并绘制曲线
    print(f"\n加载训练历史...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # 如果模型文件中有训练历史，绘制曲线
        if 'train_accuracies' in checkpoint:
            acc_path = os.path.join(OUTPUT_DIR, f"accuracy_curve_{timestamp}.png")
            val_acc = checkpoint.get('val_accuracies', None)
            plot_accuracy_curve(checkpoint['train_accuracies'], val_acc, acc_path)
        else:
            print(f"  ⚠ 模型文件中未找到训练准确率历史")
        
        if 'train_losses' in checkpoint:
            loss_path = os.path.join(OUTPUT_DIR, f"loss_curve_{timestamp}.png")
            val_loss = checkpoint.get('val_losses', None)
            plot_loss_curve(checkpoint['train_losses'], val_loss, loss_path)
        else:
            print(f"  ⚠ 模型文件中未找到训练损失历史")
            
    except Exception as e:
        print(f"  ⚠ 加载训练历史失败: {str(e)}")
    
    # 7. 保存评估报告
    print(f"\n{'='*60}")
    print("步骤7: 保存评估报告")
    print(f"{'='*60}")
    
    report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("模型评估报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {MODEL_PATH}\n")
        f.write(f"数据样本数: {len(y_true)}\n")
        f.write(f"类别数: {NUM_CLASSES}\n\n")
        
        f.write("总体指标:\n")
        f.write(f"  准确率: {results['accuracy']*100:.2f}%\n")
        f.write(f"  宏平均精确率: {results['macro_precision']*100:.2f}%\n")
        f.write(f"  宏平均召回率: {results['macro_recall']*100:.2f}%\n")
        f.write(f"  宏平均F1分数: {results['macro_f1']*100:.2f}%\n\n")
        
        f.write("各类别详细指标:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  {class_name}:\n")
            f.write(f"    精确率: {results['precision'][i]*100:.2f}%\n")
            f.write(f"    召回率: {results['recall'][i]*100:.2f}%\n")
            f.write(f"    F1分数: {results['f1_score'][i]*100:.2f}%\n")
            f.write(f"    样本数: {results['support'][i]}\n\n")
        
        f.write("混淆矩阵:\n")
        f.write(str(results['confusion_matrix']))
        f.write("\n\n")
        
        f.write("分类报告:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"✓ 评估报告已保存: {os.path.basename(report_path)}")
    
    # 8. 保存JSON格式的评估结果
    json_report_path = os.path.join(OUTPUT_DIR, f"evaluation_results_{timestamp}.json")
    json_results = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "num_samples": len(y_true),
        "num_classes": NUM_CLASSES,
        "overall_accuracy": float(results['accuracy']),
        "macro_precision": float(results['macro_precision']),
        "macro_recall": float(results['macro_recall']),
        "macro_f1": float(results['macro_f1']),
        "class_names": class_names,
        "per_class_metrics": {
            class_names[i]: {
                "precision": float(results['precision'][i]),
                "recall": float(results['recall'][i]),
                "f1_score": float(results['f1_score'][i]),
                "support": int(results['support'][i])
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": results['confusion_matrix'].tolist()
    }
    
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON评估结果已保存: {os.path.basename(json_report_path)}")
    
    print(f"\n{'='*60}")
    print("✅ 评估完成！")
    print(f"{'='*60}")
    print(f"生成的文件:")
    print(f"  混淆矩阵: confusion_matrix_{timestamp}.png")
    print(f"  分类报告图: classification_report_{timestamp}.png")
    print(f"  评估报告: evaluation_report_{timestamp}.txt")
    print(f"  JSON结果: evaluation_results_{timestamp}.json")
    print(f"\n所有文件已保存到: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()

