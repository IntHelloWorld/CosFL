import os
import re
from typing import Dict, List

from Utils.path_manager import PathManager


def parse_sbfl(sbfl_file) -> Dict[str, List[int]]:
    """
    Parse the SBFL result from line level to method level.
    e.g.:
        org.jfree.chart.plot$CategoryPlot#CategoryPlot():567;0.2581988897471611
        org.jfree.chart.plot$CategoryPlot#CategoryPlot():568;0.2581988897471611
        
        ==>
        
        {
            "CategoryPlot": [567, 568]
        }
    """
    res = {}
    with open(sbfl_file, "r") as f:
        line = f.readline() # skip the first line
        line = f.readline().strip("\n")
        while line:
            full_name, method_name, line_num, score = re.split(r"[#:;]", line)
            class_name = full_name.split("$")[1]
            
            if score == "0.0":
                break

            try:
                res[class_name].append(int(line_num))
            except:
                res[class_name] = [int(line_num)]
            line = f.readline().strip("\n")
    return res

def get_all_sbfl_res(path_manager: PathManager):
    sbfl_names = ["tarantula","ochiai","jaccard","ample","ochiai2","dstar"]
    sbfl_files = []
    for name in sbfl_names:
        sbfl_files.append(os.path.join(
            path_manager.root_path,
            "SBFL",
            "results",
            path_manager.project,
            path_manager.bug_id,
            f"{name}.ranking.csv"
        ))
    
    sbfl_reses = []
    for sbfl_file in sbfl_files:
        sbfl_reses.append(parse_sbfl(sbfl_file))
    return sbfl_reses