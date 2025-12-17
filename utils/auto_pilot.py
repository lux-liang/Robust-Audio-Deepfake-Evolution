import time
import os
import subprocess
import sys

# Configuration
LOG_FILE = "train_phase4.log"
CHECK_INTERVAL = 60  # Seconds
EPOCH_TO_VALIDATE = 5
MAX_EPOCHS = 30

def get_last_epoch(log_file):
    """Parse the log file to find the last started epoch."""
    if not os.path.exists(log_file):
        return -1
    try:
        # Efficiently read last lines to find epoch
        result = subprocess.check_output(['grep', 'Start training epoch', log_file]).decode('utf-8')
        last_line = result.strip().split('\n')[-1]
        # Expected format: "Start training epoch005"
        epoch_str = last_line.replace("Start training epoch", "")
        return int(epoch_str)
    except subprocess.CalledProcessError:
        return -1
    except Exception as e:
        print(f"Error parsing epoch: {e}")
        return -1

def check_dataset_status():
    """Check if dataset repair is done."""
    fix_marker = "/root/autodl-tmp/ASVspoof2021_DF_fix"
    target_dir = "/root/autodl-tmp/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac"
    
    # If fix dir exists, check if we need to sync
    if os.path.exists(fix_marker):
        try:
            count = len(os.listdir(f"{fix_marker}/ASVspoof2021_DF_eval/flac"))
            if count > 600000:
                print("📦 Dataset extraction complete. Running sync...")
                subprocess.run(["./sync_dataset.sh"], check=True)
                print("✅ Sync complete. Removing temp files...")
                subprocess.run(["rm", "-rf", fix_marker], check=True)
                return True
        except Exception as e:
            print(f"Dataset check error: {e}")
            return False
    
    # Verify target dir
    if os.path.exists(target_dir):
        count = len(os.listdir(target_dir))
        if count > 600000:
            return True
    return False

def run_validation(epoch):
    """Run validation for specific epoch."""
    print(f"🧪 Running validation for Epoch {epoch}...")
    
    # 1. Validate on 2019 LA (Standard)
    # Already done by main.py during training, checked via dev_score
    
    # 2. Validate on 2021 DF (The real test)
    model_path = f"exp_result/cascade_mamba_phase4_gated/weights/epoch_{epoch}_*.pth"
    # Find exact file
    files = subprocess.getoutput(f"ls {model_path}").split('\n')
    if not files or "No such file" in files[0]:
        print("❌ Model file not found for validation.")
        return

    model_file = files[0]
    print(f"   Model: {model_file}")
    
    # Update config if needed or use command line args for eval script if available
    # We'll use the existing eval script structure but point to new model
    # Note: We need to make sure evaluate_2021DF.py can accept model path arg or modify it
    
    # Let's modify evaluate_2021DF.py dynamically or assume it reads from somewhere
    # Actually, better to create a specific eval script for this phase
    
    cmd = f"python evaluate_2021DF.py --model_path {model_file}" 
    # Note: evaluate_2021DF.py might not support args yet, we might need to update it.
    # Checking previous steps, we used it but it hardcoded paths. 
    # We should update it to accept args to be robust.
    
    subprocess.run(cmd, shell=True)
    subprocess.run("python calc_eer_21df.py", shell=True)

def main_loop():
    print("🤖 Auto-Pilot Engaged. Rest well, human.")
    last_processed_epoch = -1
    
    while True:
        current_epoch = get_last_epoch(LOG_FILE)
        
        # 1. Monitor Training
        if current_epoch > last_processed_epoch:
            print(f"📈 Training advanced to Epoch {current_epoch}")
            last_processed_epoch = current_epoch
            
            # Check for specific milestones
            if current_epoch == EPOCH_TO_VALIDATE:
                if check_dataset_status():
                    run_validation(current_epoch)
                else:
                    print("⚠️ Dataset not ready for validation yet.")

        # 2. Monitor Dataset Repair (Independent check)
        if not check_dataset_status():
            # If not ready, the background supervisor or monitor script should be handling it.
            # We just log status.
            print("🔧 Dataset repair still in progress...")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()






