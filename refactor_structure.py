#!/usr/bin/env python3
"""
Script to refactor the project structure for open-source release.
Moves Phase 6 code to src/, archives previous phases, and organizes files.
"""
import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = Path("/root/aasist-main")

# Phase 6 files (latest version - Codec Augmentation & BiMamba)
PHASE6_FILES = [
    "main.py",
    "evaluation.py",
    "loss.py",
    "sam.py",
    "data_utils.py",
    "utils.py",
    "rawboost.py",
    "rawboost_official.py",
    "filter_dirty_data.py",
    "run_phase6_pipeline.sh",
    "phase6_watch.py",
    "phase6_attack_watch.py",
    "phase6_a18a19_watch_pid.txt",
    "phase6_watch_pid.txt",
    "training_pid_phase6.txt",
]

PHASE6_CONFIGS = [
    "config/Phase6_Proposed.conf",
    "config/Phase6_Run.conf",
]

PHASE6_MODELS = [
    "models/DualStreamSEMamba.py",
    "models/DualStreamSEMamba/",
]

# Phase 3 files (MoE-Mamba)
PHASE3_FILES = [
    "check_moe_model.py",
]

PHASE3_CONFIGS = [
    "config/MoEMambaASV.conf",
]

PHASE3_MODELS = [
    "models/MoEMambaASV.py",
]

# Phase 4 files (Dual-Stream)
PHASE4_CONFIGS = [
    "config/DualStreamSEMamba.conf",
]

# Phase 5 files (Finetune)
PHASE5_CONFIGS = [
    "config/Phase5_Finetune.conf",
]

# Shared utilities (to be moved to utils/)
SHARED_UTILS = [
    "download_dataset.py",
    "download_keys.py",
    "check_dataset.py",
    "check_model.py",
    "compare_models.py",
    "quick_verify_dataset.py",
    "visualize.py",
    "visualize_tsne.py",
    "inspect_model_gate.py",
    "auto_pilot.py",
    "persistent_autopilot.py",
]

# Documentation files (to be moved to docs/)
DOCS_FILES = [
    "MODEL_EVOLUTION_DETAILED_REPORT.md",
    "Model_Evolution_Report.md",
    "PROJECT_STRUCTURE.md",
    "PROJECT_PROGRESS_REPORT_2025_11_24.md",
    "ALL_PHASES_PERFORMANCE_SUMMARY.md",
    "ALL_PHASES_2021DF_PERFORMANCE.md",
    "CODE_REORGANIZATION_SUMMARY.md",
    "DISK_CLEANUP_REPORT.md",
    "EVALUATION_READY.md",
    "TRAINING_SAFETY_CHECKLIST.md",
    "UPLOAD_GUIDE.md",
    "README_TRAINING.md",
    "模型性能与瓶颈分析报告.md",
    "analysis_report_v2.md",
    "AASIST_Reproduction_Report/",
]

# Analysis scripts (Phase 6 related)
PHASE6_ANALYSIS = [
    "analyze_breakdown.py",
    "analyze_2021df_breakdown.py",
    "report_2021df_codec_breakdown.py",
]

def move_files(file_list, dest_dir, description):
    """Move files to destination directory."""
    dest_path = BASE_DIR / dest_dir
    dest_path.mkdir(parents=True, exist_ok=True)
    
    moved = []
    for file_path in file_list:
        src = BASE_DIR / file_path
        if src.exists():
            dest = dest_path / Path(file_path).name
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
                shutil.rmtree(src)
            else:
                shutil.move(str(src), str(dest))
            moved.append(file_path)
            print(f"  ✓ Moved {file_path} -> {dest_dir}/")
    
    if moved:
        print(f"\n✓ {description}: Moved {len(moved)} items")
    else:
        print(f"\n⚠ {description}: No files found")
    
    return moved

def main():
    print("=" * 60)
    print("Refactoring Project Structure")
    print("=" * 60)
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    dirs = [
        "src",
        "legacy_archives/phase3_moe",
        "legacy_archives/phase4_dual_stream",
        "legacy_archives/phase5_clean_specialist",
        "docs",
        "utils",
    ]
    for dir_path in dirs:
        (BASE_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {dir_path}/")
    
    # Move Phase 6 files to src/
    print("\n2. Moving Phase 6 (latest) code to src/...")
    move_files(PHASE6_FILES, "src", "Phase 6 main files")
    move_files(PHASE6_CONFIGS, "src/config", "Phase 6 configs")
    move_files(PHASE6_MODELS, "src/models", "Phase 6 models")
    move_files(PHASE6_ANALYSIS, "src", "Phase 6 analysis scripts")
    
    # Copy config and models directories structure
    print("\n3. Copying config and models directories to src/...")
    config_src = BASE_DIR / "config"
    config_dest = BASE_DIR / "src/config"
    if config_src.exists():
        # Copy all configs first, then we'll organize them
        shutil.copytree(config_src, config_dest, dirs_exist_ok=True)
        print("  ✓ Copied config/ to src/config/")
    
    models_src = BASE_DIR / "models"
    models_dest = BASE_DIR / "src/models"
    if models_src.exists():
        # Copy modules and official directories
        if (models_src / "modules").exists():
            shutil.copytree(models_src / "modules", models_dest / "modules", dirs_exist_ok=True)
        if (models_src / "official").exists():
            shutil.copytree(models_src / "official", models_dest / "official", dirs_exist_ok=True)
        print("  ✓ Copied models/modules and models/official to src/models/")
    
    # Move Phase 3 to legacy
    print("\n4. Moving Phase 3 (MoE-Mamba) to legacy_archives/phase3_moe/...")
    move_files(PHASE3_FILES, "legacy_archives/phase3_moe", "Phase 3 files")
    move_files(PHASE3_CONFIGS, "legacy_archives/phase3_moe/config", "Phase 3 configs")
    move_files(PHASE3_MODELS, "legacy_archives/phase3_moe/models", "Phase 3 models")
    
    # Move Phase 4 to legacy
    print("\n5. Moving Phase 4 (Dual-Stream) to legacy_archives/phase4_dual_stream/...")
    move_files(PHASE4_CONFIGS, "legacy_archives/phase4_dual_stream/config", "Phase 4 configs")
    
    # Move Phase 5 to legacy
    print("\n6. Moving Phase 5 (Finetune) to legacy_archives/phase5_clean_specialist/...")
    move_files(PHASE5_CONFIGS, "legacy_archives/phase5_clean_specialist/config", "Phase 5 configs")
    
    # Move shared utilities
    print("\n7. Moving shared utilities to utils/...")
    move_files(SHARED_UTILS, "utils", "Shared utilities")
    
    # Move documentation
    print("\n8. Moving documentation to docs/...")
    move_files(DOCS_FILES, "docs", "Documentation files")
    
    print("\n" + "=" * 60)
    print("Refactoring Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the moved files")
    print("2. Create .gitignore")
    print("3. Update requirements.txt")
    print("4. Update import paths in src/ files if needed")

if __name__ == "__main__":
    main()

