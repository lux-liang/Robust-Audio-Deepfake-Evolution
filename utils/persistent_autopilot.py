import time
import os
import subprocess
import re

# Configuration
LOG_FILE = "train_phase5.log"
CHECK_INTERVAL = 300  # Check every 5 minutes to reduce log spam
VALIDATION_EPOCHS = [1, 3, 5, 10, 15, 20, 25, 29] # More frequent validation early on

def get_last_epoch(log_file):
    if not os.path.exists(log_file):
        return -1
    try:
        result = subprocess.check_output(['tail', '-n', '100', log_file]).decode('utf-8')
        matches = re.findall(r"Start training epoch(\d+)", result)
        if matches:
            return int(matches[-1])
        return -1
    except Exception:
        return -1

def run_validation(epoch):
    print(f"🧪 [Auto-Pilot] Triggering Validation for Epoch {epoch}...")
    
    # 1. Find the model weights
    model_dir = "exp_result/cascade_mamba_phase5_ultimate"
    # Search recursively for the weight file
    cmd = f"find {model_dir} -name 'epoch_{epoch}_*.pth' | head -n 1"
    model_path = subprocess.getoutput(cmd).strip()
    
    if not model_path:
        print(f"❌ [Auto-Pilot] Model file for Epoch {epoch} not found. Retrying later.")
        return False

    print(f"   Model found: {model_path}")
    
    # 2. Run 2021 DF Evaluation
    log_file = f"auto_val_epoch{epoch}.log"
    eval_cmd = f"python evaluate_2021DF_patched.py --model_path '{model_path}' > {log_file} 2>&1"
    subprocess.run(eval_cmd, shell=True)
    
    # 3. Parse and Report EER
    eer_cmd = f"python calc_eer_21df.py" # Note: this script needs to know which score file to use
    # Currently calc_eer_21df.py hardcodes the score file path. 
    # Let's assume evaluate_2021DF_patched.py overwrites the default score file.
    # Based on previous code, it writes to exp_result/cascade_mamba_2021DF_eval/eval_scores_2021DF.txt
    
    eer_output = subprocess.getoutput(eer_cmd)
    print(f"📊 [Auto-Pilot] Validation Result (Epoch {epoch}):")
    print(eer_output)
    
    # 4. Save result to a summary file
    with open("phase4_progress_report.txt", "a") as f:
        f.write(f"\n=== Epoch {epoch} Validation ===\n")
        f.write(f"Model: {model_path}\n")
        f.write(eer_output + "\n")
        
    return True

def main():
    print("🤖 Persistent Auto-Pilot Activated.")
    print("   - Monitoring: train_phase4.log")
    print("   - Validation Checkpoints: " + str(VALIDATION_EPOCHS))
    
    processed_epochs = set()
    
    while True:
        current_epoch = get_last_epoch(LOG_FILE)
        
        if current_epoch in VALIDATION_EPOCHS and current_epoch not in processed_epochs:
            # Wait a bit to ensure the model file is fully saved
            print(f"⏳ [Auto-Pilot] Epoch {current_epoch} detected. Waiting 60s for file save...")
            time.sleep(60) 
            
            if run_validation(current_epoch):
                processed_epochs.add(current_epoch)
                
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()


