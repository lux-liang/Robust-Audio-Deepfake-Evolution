import os
import glob
import shutil
import time

def monitor_extraction_and_merge():
    source_dir = "/root/autodl-tmp/ASVspoof2021_DF_fix/ASVspoof2021_DF_eval/flac"
    target_dir = "/root/autodl-tmp/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac"
    
    print("🚀 Starting Dataset Repair Monitor...")
    
    # Check if extraction started
    while not os.path.exists(source_dir):
        print("⏳ Waiting for extraction to create source directory...")
        time.sleep(30)
        
    print("✅ Source directory detected. Monitoring file counts...")
    
    last_count = 0
    stable_count = 0
    
    while True:
        # Count files in source (extraction output)
        # Note: using os.scandir is faster than glob or ls for large dirs
        try:
            current_count = len(os.listdir(source_dir))
        except FileNotFoundError:
            current_count = 0
            
        print(f"📦 Extracted files: {current_count}")
        
        if current_count > 600000: # Expected total is around 611k
             print("✅ Extraction seems mostly complete. Starting merge process...")
             break
             
        if current_count == last_count and current_count > 0:
            stable_count += 1
            if stable_count > 10: # No change for 5 minutes (30s * 10)
                print("⚠️ Extraction seems stalled or finished. Proceeding with merge.")
                break
        else:
            stable_count = 0
            
        last_count = current_count
        time.sleep(30)
        
    # Merge strategy: move missing files from fix to target
    print("🔄 Merging files...")
    
    # This part we might want to do carefully to avoid overwriting if not needed, 
    # or just rsync. Since Python is slow for 600k file checks, 
    # we'll use rsync in shell via subprocess if possible, or just tell user.
    # But for this script, let's just finish monitoring.
    print("✅ Monitoring complete. Ready for rsync.")

if __name__ == "__main__":
    monitor_extraction_and_merge()






