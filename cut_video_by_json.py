"""
视频裁切工具 - 根据JSON文件裁切视频

文件查找说明：
1. JSON文件查找位置：
   - 从数据库读取 JSON_FOLDER 配置
   - 如果数据库中没有配置，则使用 OUTPUT_FOLDER
   - 如果都没有，使用默认路径 "./output_dicts"
   - 脚本会在该文件夹中查找所有 .json 文件

2. 视频文件查找位置：
   - 从数据库读取 FOLDER_PATH 配置（视频文件夹路径）
   - 如果数据库中没有配置，使用默认路径 "./videos"
   - 脚本会根据JSON中CSV文件名在该文件夹中查找对应的视频文件
   - 支持的视频格式：.mp4, .avi, .mov, .mkv, .flv

3. 输出文件位置：
   - 从数据库读取 CUT_OUTPUT_FOLDER 配置
   - 如果数据库中没有配置，默认输出到视频文件夹的父目录下的 "cut_videos" 文件夹
   - 每个JSON文件会创建一个对应的子文件夹
   - 裁切后的视频文件命名格式：{视频名}_value_{值}.mp4

示例：
  JSON文件位置: ./accest/output_dicts/all_continuous_values.json
  视频文件位置: ./videos/video1.mp4
  输出位置: ./cut_videos/all_continuous_values/video1_value_1.mp4
"""

import os
import json
import cv2
from typing import Dict, List
import sys
from .. import PoseKeypoints
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# import sql

def load_json_file(json_path: str) -> Dict:
    """
    加载JSON文件
    :param json_path: JSON文件路径
    :return: 解析后的字典数据
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 成功加载JSON文件: {json_path}")
        return data
    except Exception as e:
        print(f"✗ 加载JSON文件失败: {str(e)}")
        return {}

def scan_folder_json(folder_path: str) -> List[str]:
    """
    扫描文件夹下所有.json文件
    :param folder_path: 文件夹路径（JSON文件查找位置）
    :return: JSON文件路径列表
    """
    print(f"\n📁 正在扫描JSON文件夹: {os.path.abspath(folder_path)}")
    if not os.path.exists(folder_path):
        print(f"  ⚠ 警告: 文件夹不存在，将尝试创建")
        try:
            os.makedirs(folder_path)
            print(f"  ✓ 已创建文件夹: {folder_path}")
        except Exception as e:
            print(f"  ✗ 创建文件夹失败: {str(e)}")
            return []
    
    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.json'):
                full_path = os.path.join(root, file)
                json_files.append(full_path)
                print(f"  ✓ 找到JSON文件: {file}")
    
    return json_files


def cut_video_by_interval(video_path: str, start_frame: int, end_frame: int,
                          output_path: str) -> bool:
    """
    裁切单个区间的视频（修复后：单个区间对应单个视频文件）
    :param video_path: 原视频路径
    :param start_frame: 起始帧
    :param end_frame: 结束帧
    :param output_path: 输出视频路径
    :return: 是否成功
    """
    video_path = os.path.abspath(video_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(video_path):
        print(f"✗ 视频文件不存在: {video_path}")
        return False

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ 无法打开视频文件: {video_path}")
        return False

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = int(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 校验帧号有效性
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    if start_frame >= end_frame:
        print(f"✗ 无效区间: 起始帧 {start_frame} >= 结束帧 {end_frame}")
        cap.release()
        return False

    # 设置视频编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"✗ 无法创建输出视频文件: {output_path}")
            cap.release()
            return False

    # 定位到起始帧并写入区间内的帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written_frames = 0
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ 提前结束：帧 {frame_idx} 读取失败")
            break
        out.write(frame)
        written_frames += 1

    # 释放资源
    cap.release()
    out.release()

    # 验证文件有效性
    if written_frames == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"✗ 区间 {start_frame}~{end_frame} 无有效帧写入")
        return False

    file_size = os.path.getsize(output_path)
    if file_size > 0:
        print(f"✓ 单个区间裁切完成: {output_path}")
        print(f"  帧范围: {start_frame}~{end_frame} (共{written_frames}帧)")
        print(f"  文件大小: {file_size / (1024 * 1024):.2f} MB")
        return True
    else:
        os.remove(output_path)
        print(f"✗ 输出文件为空: {output_path}")
        return False



def process_json_and_cut_videos(json_path: str, video_folder: str, output_folder: str,
                                target_values: List[str] = None) -> Dict[str, int]:
    """
    处理JSON文件并裁切视频
    :param json_path: JSON文件路径（输入文件）
    :param video_folder: 视频文件夹路径（视频文件查找位置）
    :param output_folder: 输出文件夹路径（裁切后视频的保存位置）
    :param target_values: 要处理的value值列表（None表示处理所有）
    :return: 处理统计信息 {成功数, 失败数}
    """
    print(f"\n📂 视频文件查找位置: {os.path.abspath(video_folder)}")
    print(f"💾 输出文件夹位置: {os.path.abspath(output_folder)}")

    # 加载JSON数据
    json_data = load_json_file(json_path)
    if not json_data:
        return {"success": 0, "failed": 0}

    # 规范化输出文件夹路径
    output_folder = os.path.abspath(output_folder)

    # 创建输出文件夹（确保父目录也存在）
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
            print(f"✓ 已创建输出文件夹: {output_folder}")
        except Exception as e:
            print(f"✗ 创建输出文件夹失败: {output_folder}")
            print(f"  错误信息: {str(e)}")
            return {"success": 0, "failed": 0}
    else:
        print(f"✓ 输出文件夹已存在: {output_folder}")

    # 验证文件夹是否可写
    if not os.access(output_folder, os.W_OK):
        print(f"✗ 输出文件夹不可写: {output_folder}")
        return {"success": 0, "failed": 0}

    # 获取JSON文件名（不含扩展名）
    json_basename = os.path.splitext(os.path.basename(json_path))[0]

    stats = {"success": 0, "failed": 0}

    # 遍历JSON中的每个CSV文件（对应一个视频）
    for csv_filename, csv_data in json_data.items():
        print(f"\n处理文件: {csv_filename}")

        # 查找对应的视频文件
        video_name = os.path.splitext(csv_filename)[0]  # 去掉.csv扩展名
        video_path = None

        print(f"  🔍 正在查找视频文件: {video_name}")
        print(f"     查找位置: {os.path.abspath(video_folder)}")

        # 尝试多种视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        for ext in video_extensions:
            potential_path = os.path.join(video_folder, video_name + ext)
            if os.path.exists(potential_path):
                video_path = potential_path
                print(f"     尝试格式 {ext}: ✓ 找到")
                break
            else:
                print(f"     尝试格式 {ext}: ✗ 不存在")

        if not video_path:
            print(f"  ✗ 未找到对应的视频文件: {video_name}")
            print(f"     已尝试的格式: {', '.join(video_extensions)}")
            print(f"     请确认视频文件是否存在于: {os.path.abspath(video_folder)}")
            stats["failed"] += 1
            continue

        print(f"  ✓ 找到视频: {os.path.abspath(video_path)}")

        # 处理每个value值
        for value, intervals in csv_data.items():
            if target_values is not None and value not in target_values:
                continue
            if value == "-1" or value == -1:
                continue

            print(f"\n  处理 value={value} (共 {len(intervals)} 个区间)")

            # 遍历每个区间，生成独立文件【核心修改：每个区间一个文件】
            for interval_idx, (start_col, end_col) in enumerate(intervals):
                try:
                    start_frame = int(start_col)
                    end_frame = int(end_col)

                    # 生成带区间序号的输出文件名
                    # 格式：[视频名]_value_[值]_interval_[区间序号].mp4
                    output_filename = f"{video_name}_value_{value}_interval_{interval_idx + 1}.mp4"
                    output_path = os.path.join(output_folder, output_filename)

                    print(f"\n  处理区间 {interval_idx + 1}: 帧 {start_frame} ~ {end_frame}")
                    print(f"  📤 输出文件: {os.path.abspath(output_path)}")

                    # 调用单区间裁切函数
                    if cut_video_by_interval(video_path, start_frame, end_frame, output_path):
                        stats["success"] += 1
                        print(f"  ✅ 成功保存区间 {interval_idx + 1}")
                    else:
                        stats["failed"] += 1
                        print(f"  ❌ 区间 {interval_idx + 1} 保存失败")
                except ValueError as e:
                    print(f"  ✗ 区间 {interval_idx + 1} 格式错误: {start_col} ~ {end_col}, 错误: {str(e)}")
                    stats["failed"] += 1
                    continue

        return stats




def batch_process_json_folder(json_folder: str, video_folder: str, output_base_folder: str,
                             target_values: List[str] = None) -> Dict[str, int]:
    """
    批量处理JSON文件夹中的所有JSON文件
    :param json_folder: JSON文件夹路径（JSON文件查找位置）
    :param video_folder: 视频文件夹路径（视频文件查找位置）
    :param output_base_folder: 输出基础文件夹路径（裁切后视频的保存位置）
    :param target_values: 要处理的value值列表（None表示处理所有）
    :return: 总处理统计信息
    """
    print(f"\n{'='*60}")
    print("🔍 开始扫描和查找文件")
    print(f"{'='*60}")
    
    json_files = scan_folder_json(json_folder)
    
    if not json_files:
        print(f"\n⚠ 在文件夹 {os.path.abspath(json_folder)} 中未找到JSON文件")
        print(f"   请确认JSON文件是否存在于该位置")
        return {"success": 0, "failed": 0}
    
    print(f"\n✓ 共找到 {len(json_files)} 个JSON文件")
    
    # 确保输出文件夹存在
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
        print(f"\n✓ 已创建输出基础文件夹: {os.path.abspath(output_base_folder)}")
    else:
        print(f"\n✓ 输出基础文件夹已存在: {os.path.abspath(output_base_folder)}")
    
    total_stats = {"success": 0, "failed": 0}
    
    for json_file in json_files:
        print(f"\n{'='*60}")
        print(f"📄 处理JSON文件: {os.path.basename(json_file)}")
        print(f"   完整路径: {os.path.abspath(json_file)}")
        print(f"{'='*60}")
        
        # 为每个JSON文件创建单独的输出文件夹
        json_basename = os.path.splitext(os.path.basename(json_file))[0]
        output_folder = os.path.join(output_base_folder, json_basename)
        print(f"📂 该JSON文件的输出文件夹: {os.path.abspath(output_folder)}")
        
        stats = process_json_and_cut_videos(json_file, video_folder, output_folder, target_values)
        
        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
    
    return total_stats

if __name__ == '__main__':
    print("="*60)
    print("视频裁切工具 - 根据JSON文件裁切视频")
    print("="*60)
    
    # 从数据库获取配置
    try:
        # JSON_FOLDER = sql.name_at_address(table_name="user", list_col="address", target_name="JSON_FOLDER")
        JSON_FOLDER = PoseKeypoints.JSON_FOLDER
        if not JSON_FOLDER or JSON_FOLDER == "":
            # 如果数据库中没有配置，尝试从OUTPUT_FOLDER获取
            try:
                # OUTPUT_FOLDER = sql.name_at_address(table_name="user", list_col="address", target_name="OUTPUT_FOLDER")
                OUTPUT_FOLDER = PoseKeypoints.JSON_FOLDER
                # JSON_FOLDER = OUTPUT_FOLDER if OUTPUT_FOLDER else "./output_dicts"
            except:
                JSON_FOLDER = "./output_dicts"
    except:
        JSON_FOLDER = "../../output_dicts"
    
    try:
        # VIDEO_FOLDER = sql.name_at_address(table_name="user", list_col="address", target_name="FOLDER_PATH")
        VIDEO_FOLDER = PoseKeypoints.VIDEO_FOLDER
        if not VIDEO_FOLDER or VIDEO_FOLDER == "":
            VIDEO_FOLDER = "./videos"
    except:
        VIDEO_FOLDER = "../../videos"
    
    try:
        # CUT_OUTPUT_FOLDER = sql.name_at_address(table_name="user", list_col="address", target_name="CUT_OUTPUT_FOLDER")
        CUT_OUTPUT_FOLDER = PoseKeypoints.CUT_OUTPUT_FOLDER
        if not CUT_OUTPUT_FOLDER or CUT_OUTPUT_FOLDER == "":
            CUT_OUTPUT_FOLDER = os.path.join(os.path.dirname(VIDEO_FOLDER), "cut_videos")
    except:
        # 截取后的短视频
        CUT_OUTPUT_FOLDER = os.path.join(os.path.dirname(VIDEO_FOLDER) if VIDEO_FOLDER else ".", "cut_videos")
    
    print(f"\n{'='*60}")
    print("📋 配置信息")
    print(f"{'='*60}")
    print(f"📁 JSON文件查找位置:")
    print(f"   {os.path.abspath(JSON_FOLDER)}")
    print(f"   (从数据库 JSON_FOLDER 或 OUTPUT_FOLDER 读取)")
    print(f"\n🎬 视频文件查找位置:")
    print(f"   {os.path.abspath(VIDEO_FOLDER)}")
    print(f"   (从数据库 FOLDER_PATH 读取)")
    print(f"\n💾 裁切视频输出位置:")
    print(f"   {os.path.abspath(CUT_OUTPUT_FOLDER)}")
    print(f"   (从数据库 CUT_OUTPUT_FOLDER 读取，如果不存在则自动创建)")
    print(f"{'='*60}")
    
    # 可选：指定要处理的value值（None表示处理所有）
    TARGET_VALUES = None  # 例如: ["1", "2"] 只处理value为1和2的区间
    
    # 批量处理
    print(f"\n开始批量处理...")
    total_stats = batch_process_json_folder(JSON_FOLDER, VIDEO_FOLDER, CUT_OUTPUT_FOLDER, TARGET_VALUES)
    
    print(f"\n{'='*60}")
    print("✅ 处理完成！")
    print(f"{'='*60}")
    print(f"📊 处理统计:")
    print(f"   成功: {total_stats['success']} 个视频")
    print(f"   失败: {total_stats['failed']} 个视频")
    print(f"\n💾 所有裁切后的视频已保存到:")
    print(f"   {os.path.abspath(CUT_OUTPUT_FOLDER)}")
    print(f"\n📁 输出文件结构示例:")
    print(f"   {CUT_OUTPUT_FOLDER}/")
    print(f"   ├── [JSON文件名1]/")
    print(f"   │   ├── video1_value_1.mp4")
    print(f"   │   └── video1_value_2.mp4")
    print(f"   ├── [JSON文件名2]/")
    print(f"   │   └── video2_value_1.mp4")
    print(f"   └── ...")
    print(f"\n💡 说明:")
    print(f"   - 每个JSON文件会创建一个对应的子文件夹")
    print(f"   - 裁切后的视频文件命名格式: [视频名]_value_[值].mp4")
    print(f"{'='*60}")

