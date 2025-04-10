import os
import re
from datetime import datetime


def parse_time(line):
    """Parse time from a log line"""
    time_str = line.split(' - ')[1]
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

def calculate_time_diff(time_points, start_point, end_point):
    """Calculate time difference in seconds"""
    try:
        time_diff = (time_points[end_point] - time_points[start_point]).total_seconds()
    except KeyError:
        time_diff = -1
    return time_diff

def analyze_cost(log_content):
    """Analyze time cost for different stages using regex"""
    lines = log_content.split('\n')
    time_points = {}
    
    patterns = {
        'context_start': r'summarizing \d+ contexts\.\.\.',
        'context_end': r'found \d+ methods already summarized',
        'node_start': r'summarizing \d+ methods\.\.\.',
        'node_end': r'get \d+ description nodes',
        'embedding_start': r'Generating Embedding for \d+ nodes',
        'embedding_end': r'Diagnosing faulty functionality\.\.\.',
        'diagnose_start': r'Diagnosing faulty functionality\.\.\.',
        'diagnose_end': r'Retrieving and reranking method nodes\.\.\.',
        'retrieval_start': r'Retrieving and reranking method nodes\.\.\.',
        'retrieval_end': r'Chat rerank \d+ method nodes'
    }
    
    for line in lines:
        if not re.search(r'^INFO', line):
            continue
            
        for point, pattern in patterns.items():
            if re.search(pattern, line):
                time_points[point] = parse_time(line)

    results = {
        'context_time': calculate_time_diff(time_points, 'context_start', 'context_end'),
        'node_time': calculate_time_diff(time_points, 'node_start', 'node_end'),
        'embedding_time': calculate_time_diff(time_points, 'embedding_start', 'embedding_end'),
        'diagnose_time': calculate_time_diff(time_points, 'diagnose_start', 'diagnose_end'),
        'retrieval_time': calculate_time_diff(time_points, 'retrieval_start', 'retrieval_end')
    }

    return results

if __name__ == '__main__':
    root_dir = "/home/qyh/projects/GarFL/DebugResult/default"
    all_res = {
        'context_time': [],
        'node_time': [],
        'embedding_time': [],
        'diagnose_time': [],
        'retrieval_time': []
    }
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.log'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    log_content = f.read()
                    if not log_content:
                        continue
                    time_costs = analyze_cost(log_content)
                    for stage, time in time_costs.items():
                        if time > 180:
                            continue
                        if time > 0:
                            all_res[stage].append(time)
    
    
    with open("Evaluation/time_cost.csv", "w") as f:
        f.write("context_time,node_time,embedding_time,diagnose_time,retrieval_time\n")
        for i in range(max(len(v) for v in all_res.values())):
            row = [all_res[stage][i] if i < len(all_res[stage]) else '' for stage in all_res]
            f.write(','.join(map(str, row)) + '\n')
    
    
    print("Time costs analysis (in seconds):")
    for stage in all_res:
        times = all_res[stage]
        if not times:
            continue
        avg_time = sum(times) / len(times)
        print(f"{stage}: {avg_time:.2f} (avg), {min(times):.2f} (min), {max(times):.2f} (max)")
