
import os
import random
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def check_flac_file(file_path):
    try:
        data, sr = sf.read(str(file_path))
        if len(data) == 0:
            return False, "Empty file"
        return True, None
    except Exception as e:
        return False, str(e)

def quick_verify(dataset_path, sample_size=2000):
    dataset_path = Path(dataset_path)
    flac_dir = dataset_path / "ASVspoof2021_DF_eval" / "flac"
    
    if not flac_dir.exists():
        print(f"Error: {flac_dir} does not exist")
        return

    all_files = list(flac_dir.glob("*.flac"))
    total_files = len(all_files)
    print(f"Found {total_files} total files.")
    
    if total_files == 0:
        print("No files found to check.")
        return

    # Sample files
    sample_files = random.sample(all_files, min(sample_size, total_files))
    print(f"Checking a random sample of {len(sample_files)} files...")

    corrupted = []
    for f in tqdm(sample_files):
        is_valid, error = check_flac_file(f)
        if not is_valid:
            corrupted.append((f.name, error))

    if corrupted:
        print(f"\n❌ Found {len(corrupted)} corrupted files in sample:")
        for name, err in corrupted[:10]:
            print(f"{name}: {err}")
    else:
        print(f"\n✅ Successfully verified {len(sample_files)} random files. No corruption found.")

if __name__ == "__main__":
    quick_verify("/root/autodl-tmp/ASVspoof2021_DF")

