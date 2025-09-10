import os
import sys
import re
import glob
from pathlib import Path
from hq_job.job_engine_local import JobEngineLocal, JobDescription

def get_next_version_dir(base_dir):
    """
    找到最大版本号+1的目录路径
    """
    pattern = os.path.join(base_dir, "version_*")
    version_dirs = glob.glob(pattern)
    
    max_version = -1
    
    for dir_path in version_dirs:
        dir_name = os.path.basename(dir_path)
        match = re.search(r'version_(\d+)', dir_name)
        
        if match:
            version_num = int(match.group(1))
            if version_num > max_version:
                max_version = version_num
    
    next_version = max_version + 1
    next_version_dir = os.path.join(base_dir,f"version_{next_version}")
    
    return next_version_dir, next_version
    
engine = JobEngineLocal("/root/autodl-tmp/seat_task/jobs")
out_dir_root = "/root/detect_classify/lightning_logs"
out_dir, _ = get_next_version_dir(out_dir_root)

print(out_dir)

engine.run(
    JobDescription(
        command="python trainer.py",
        args=["-c", "configs/default.yaml"],
        working_dir="/root/detect_classify",
        output_dir=out_dir,
        description=sys.argv[1]
    )
)

