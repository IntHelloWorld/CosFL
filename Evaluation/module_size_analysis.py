import csv
import os
from collections import defaultdict


def analyze_cluster_sizes(root_dir, output_file):
    # 创建区间边界 (0,5), (5,10), ..., (170,175), (175,inf)
    intervals = [(i, i+5) for i in range(0, 175, 5)]
    intervals.append((175, float('inf')))  # 添加最后一个区间 >175
    counts = defaultdict(int)
    max_size = 0
    
    # 递归遍历目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "cluster_result.txt" in filenames:
            file_path = os.path.join(dirpath, "cluster_result.txt")
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content:  # 确保文件不为空
                    # 将字符串转换为数字列表
                    numbers = [int(x) for x in content.split(',')]
                    max_size = max(max_size, max(numbers))
                    
                    # 统计每个数字落入的区间
                    for num in numbers:
                        for start, end in intervals:
                            if start <= num < end:
                                counts[(start, end)] += 1
                                break
    
    # 将结果写入CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'count'])  # 写入表头
        
        # 按区间起始值排序并写入数据
        for (start, end), count in sorted(counts.items()):
            # 对于最后一个区间，用">175"表示
            if end == float('inf'):
                writer.writerow([start, f">{start}", count])
            else:
                writer.writerow([start, end, count])
            
    print(f"结果已保存到 {output_file}, 最大值为 {max_size}")


def analyze_cluster_sizes_opt(root_dir, output_file):
    # 创建区间边界
    intervals = [(i, i+2) for i in range(5, 25, 2)]
    intervals.append((25, float('inf')))  # 添加最后一个区间 >175
    counts = defaultdict(int)
    max_size = 0
    
    # 递归遍历目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "cluster_result.txt" in filenames:
            file_path = os.path.join(dirpath, "cluster_result.txt")
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content:  # 确保文件不为空
                    # 将字符串转换为数字列表
                    numbers = [int(x) for x in content.split(',')]
                    max_size = max(max_size, max(numbers))
                    
                    # 统计每个数字落入的区间
                    for num in numbers:
                        for start, end in intervals:
                            if start <= num < end:
                                counts[(start, end)] += 1
                                break
    
    # 将结果写入CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'count'])  # 写入表头
        
        # 按区间起始值排序并写入数据
        for (start, end), count in sorted(counts.items()):
            # 最后一个区间
            if end == float('inf'):
                writer.writerow([start, f">{start}", count])
            else:
                writer.writerow([start, end, count])
            
    print(f"结果已保存到 {output_file}, 最大值为 {max_size}")


if __name__ == "__main__":
    analyze_cluster_sizes("/home/qyh/projects/GarFL/DebugResult/module_analysis", "Evaluation/cluster_size.csv")
    analyze_cluster_sizes_opt("/home/qyh/projects/GarFL/DebugResult/module_analysis_optimized", "Evaluation/cluster_size_optimized.csv")
