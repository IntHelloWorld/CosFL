import os
import re


def extract_cost(line, pattern):
    """Extract cost from a log line using regex pattern"""
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    return -1

def analyze_cost(log_content):
    """Analyze money cost for different stages using regex"""
    lines = log_content.split('\n')
    costs = {
        'context_cost': -1,
        'description_cost': -1,
        'embedding_cost': -1,
        'diagnose_cost': -1,
        'rerank_cost': -1
    }
    
    patterns = {
        'context_cost': r'get context nodes with \d+ tokens and ([\d\.]+) cost',
        'description_cost': r'get description nodes with \d+ tokens and ([\d\.]+) cost',
        'embedding_cost': r'mimic cost: \d+ tokens, (.+) money',
        'diagnose_cost': r'Diagnose finished, total cost: \d+ tokens, ([\d\.]+) USD',
        'rerank_cost': r'Chat rerank cost: \d+ tokens, ([\d\.]+) money'
    }
    
    for line in lines:
        for cost_type, pattern in patterns.items():
            cost = extract_cost(line, pattern)
            if cost > 0:
                costs[cost_type] = cost
                
    return costs

if __name__ == '__main__':
    root_dir = "/home/qyh/projects/GarFL/DebugResult/mimic"
    all_costs = {
        'context_cost': [],
        'description_cost': [],
        'embedding_cost': [],
        'diagnose_cost': [],
        'rerank_cost': []
    }
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.log'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    log_content = f.read()
                    if not log_content:
                        continue
                    costs = analyze_cost(log_content)
                    for stage, cost in costs.items():
                        if cost > 0:
                            all_costs[stage].append(cost)
    
    with open("Evaluation/money_cost.csv", "w") as f:
        f.write("context_cost,description_cost,embedding_cost,diagnose_cost,rerank_cost\n")
        for i in range(max(map(len, all_costs.values()))):
            f.write(",".join(str(all_costs[stage][i]) if i < len(all_costs[stage]) else "" for stage in all_costs) + "\n")
    
    print("Money costs analysis (in USD):")
    for stage in all_costs:
        costs = all_costs[stage]
        if not costs:
            continue
        avg_cost = sum(costs) / len(costs)
        print(f"{stage}: {avg_cost:.6f} (avg), {min(costs):.6f} (min), {max(costs):.6f} (max)")
