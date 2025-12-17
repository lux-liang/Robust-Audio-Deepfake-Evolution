import subprocess
import os

os.chdir("/root/aasist-main")
with open("train_phase6_final.log", "w") as out:
    p = subprocess.Popen(
        ["python", "-u", "main.py", 
         "--config", "config/Phase6_Run.conf", 
         "--output_dir", "./exp_result", 
         "--comment", "Phase6_ResumeWithEvalDiag", 
         "--start_epoch", "3"],
        stdout=out,
        stderr=out,
        preexec_fn=os.setpgrp
    )
    
with open("training_pid_phase6.txt", "w") as f:
    f.write(str(p.pid))
    
print(f"Started training with PID {p.pid}")





