import os
import sys
import shutil
from datetime import datetime
from hq_job.job_engine_local import JobEngineLocal, JobDescription

checkpoints = sys.argv[1]
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/root/autodl-tmp/eval_out/{current_time}/"
os.makedirs(out_dir, exist_ok=True)
shutil.copy2("/root/detect_classify/utils/evaluate.py", f"{out_dir}/evaluate.py")

engine = JobEngineLocal("/root/autodl-tmp/seat_task/eval_jobs")

engine.run(
    JobDescription(
        command="python utils/evaluate.py",
        args=[
            "/root/autodl-tmp/seat_model/seat_dino_baseline.pth", 
            "/root/autodl-tmp/seat_dataset/chengdu_valid",
            checkpoints,
            "dinov3_vitb16",
            "--classification_confidence_threshold",   # 一阶段模型置信度
            "0.2",
            "--classification_score_threshold",     # 二阶段模型置信度
            "0.6"
        ],
        working_dir="/root/detect_classify",
        output_dir=out_dir,
        description=sys.argv[2]
    )
)

