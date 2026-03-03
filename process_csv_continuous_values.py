import os
import pandas as pd
from typing import Dict, List, Tuple
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
# import sql

def process_single_row_horizontal_continuous(csv_path: str) -> Dict[int, List[Tuple[str, str]]]:
    """
    处理仅1行的CSV，识别该行内「列的value连续相同」的横向区间
    :param csv_path: CSV文件路径
    :return: 结果词典，格式：{value值: [(区间起始列名, 区间结束列名), ...]}
    """
    # 1. 读取CSV：仅读取1行，保留列名（Excel中的列名，比如DKZ、DKA）
    df = pd.read_csv(
        csv_path,
        skip_blank_lines=True,
        header=None,
        nrows=1  # 强制仅读取1行
    ).fillna(-1)
    
    # 2. 统一转为整数（非数字值转-1）
    df = df.astype(int)
    
    # 3. 获取该行数据（仅1行）和对应的列名
    row_data = df.iloc[0].tolist()  # 该行的所有value
    col_names = df.columns.tolist()  # 列名（Excel中的列名，比如DKZ、DKA）
    
    # 4. 初始化结果：{value: [(起始列名, 结束列名), ...]}
    result = {}
    # 初始化横向连续值跟踪
    if not row_data:
        return result
    current_value = row_data[0]
    start_col = col_names[0]

    # 5. 遍历行内的每一列（横向判断连续值）
    for idx in range(1, len(row_data)):
        val = row_data[idx]
        col = col_names[idx]
        
        # 触发条件：当前列value ≠ 连续value（横向不连续）
        if val != current_value:
            # 记录当前连续区间
            if current_value not in result:
                result[current_value] = []
            result[current_value].append((start_col, col_names[idx-1]))
            
            # 更新跟踪变量
            current_value = val
            start_col = col

    # 6. 处理最后一段横向连续区间
    if current_value not in result:
        result[current_value] = []
    result[current_value].append((start_col, col_names[-1]))

    # 打印结果（匹配你的数据）
    print("=== 行内列的横向连续value区间 ===")
    for val, intervals in result.items():
        print(f"value={val} 的连续列区间：")
        for (start, end) in intervals:
            print(f"  {start} ~ {end}")

    return result


def scan_folder_csv(folder_path: str) -> List[str]:
    """
    扫描文件夹下所有.csv文件
    :param folder_path: 文件夹路径
    :return: CSV文件路径列表
    """
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def csv_to_continuous_dict(folder_path: str) -> Dict[str, Dict]:
    """
    主函数：处理文件夹下所有CSV，生成总词典
    :param folder_path: 文件夹路径
    :return: 总词典，格式：{CSV文件路径: 该文件的列-区间词典, ...}
    """
    csv_files = scan_folder_csv(folder_path)
    print(f"\n=== 路径 {folder_path} 下找到的CSV文件 ===")
    print(f"CSV文件数量：{len(csv_files)}")
    print(f"CSV文件列表：{csv_files}")  # 若为空，说明路径下无.csv文件
    total_result = {}
    for csv_file in csv_files:
        print(f"正在处理文件：{csv_file}")
        try:
            file_result = process_single_row_horizontal_continuous(csv_file)
            total_result[csv_file] = file_result
        except Exception as e:
            print(f"处理文件 {csv_file} 失败：{str(e)}")
            continue
    return total_result

# ------------------- 视频切割扩展功能（可选） -------------------
def cut_video_by_csv_interval(video_path: str, csv_result: Dict, col_name: str, target_num: int, output_path: str):
    """
    根据CSV的连续区间切割视频（需确保CSV行号与视频帧/时间对应）
    :param video_path: 原视频路径
    :param csv_result: 单个CSV的处理结果词典
    :param col_name: 参考的CSV列名
    :param target_num: 目标数字（切割该数字对应的区间）
    :param output_path: 输出视频路径
    """
    import cv2
    # 1. 获取目标数字的连续区间
    if col_name not in csv_result or target_num not in csv_result[col_name]:
        print(f"无目标数字 {target_num} 的区间")
        return
    intervals = csv_result[col_name][target_num]
    
    # 2. 打开视频，获取基础参数
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 3. 按区间切割视频（假设CSV行号对应视频帧号）
    for (start_row, end_row) in intervals:
        # 设置视频读取起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_row - 1)
        # 读取区间内的帧并写入
        for frame_idx in range(start_row, end_row + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    
    # 4. 释放资源
    cap.release()
    out.release()
    print(f"视频切割完成，输出路径：{output_path}")

# ------------------- 测试示例 -------------------
# ------------------- 测试示例 -------------------
if __name__ == '__main__':
    # 替换为你的CSV文件夹路径
    FOLDER_PATH = sql.name_at_address(table_name="user",list_col="address",target_name="FOLDER_PATH")
    print("获取到的FOLDER_PATH：", FOLDER_PATH)
    # 注释掉视频相关代码（若未实现/不需要）
    # VIDEO_PATH = sql.name_at_address(table_name="user",list_col="address",target_name="FOLDER_PATH")
    # OUTPUT_VIDEO_PATH = sql.name_at_address(table_name="user",list_col="address",target_name="OUTPUT_VIDEO_PATH")
    
    # 1. 处理所有CSV，生成连续值词典
    total_dict = csv_to_continuous_dict(FOLDER_PATH)
    
    # 2. 打印结果（修正核心：匹配csv_result的实际格式）
    for csv_file, csv_result in total_dict.items():
        print(f"\n=== {csv_file} 处理结果 ===")
        # csv_result格式：{value: [(起始列, 结束列), ...]}
        for value, intervals in csv_result.items():
            if value == -1:  # 跳过空值
                continue
            print(f"  value={value} 的连续列区间：")
            # 遍历该value对应的所有连续区间
            for interval_idx, (start_col, end_col) in enumerate(intervals):
                print(f"    第{interval_idx+1}个区间：列{start_col} ~ 列{end_col}")
    
    # 3. 若需要视频切割功能，先修正cut_video_by_csv_interval函数（可选）
    # if os.path.exists(VIDEO_PATH) and total_dict:
    #     first_csv = list(total_dict.keys())[0]
    #     first_result = total_dict[first_csv]
    #     # 切割value=1的区间（示例）
    #     cut_video_by_csv_interval(
    #         video_path=VIDEO_PATH,
    #         csv_result=first_result,
    #         target_num=1,  # 直接传target_num，无需col_name
    #         output_path=OUTPUT_VIDEO_PATH
    #     )