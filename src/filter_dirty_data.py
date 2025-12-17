import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from importlib import import_module
from pathlib import Path
from tqdm import tqdm
from data_utils import Dataset_ASVspoof2019_train, genSpoof_list

def get_model(model_config, device):
    """Create model exactly like main.py (Model(args, device))."""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")

    class Args:
        def __init__(self, d):
            self.__dict__.update(d)

    args = Args(model_config)
    model = _model(args, device).to(device)
    return model


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix if checkpoint was saved with DataParallel."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main(args):
    # 1. Load Configuration
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    print(f"Using device: {device}")
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "Data cleaning with WavLM-Large is extremely heavy on CPU and may get OOM-killed. "
            "Please enable GPU and run with --device cuda (recommended), or pass --allow_cpu to force CPU."
        )

    # 2. Load Model
    model_config = config["model_config"]
    model = get_model(model_config, device)

    # Apply LoRA adapters if Phase 5 config enabled it (must match training-time architecture)
    training_config = config.get("training_config", {})
    try:
        from main import apply_lora_to_wavlm
        model = apply_lora_to_wavlm(model, training_config)
    except Exception as e:
        print(f"⚠️  [DataCleaning] LoRA injection skipped/failed: {e}")
    
    # Load weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle both full checkpoint (with optimizer states) and state_dict only
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint

    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
        
    model.eval()

    # 3. Load Training Data
    # We use the training set to identify dirty samples
    database_path = Path(config["database_path"])
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)
    
    # Define paths (adapted from main.py)
    train_trial_path = (database_path /
                        "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                            track, prefix_2019))

    # Preserve original protocol lines so cleaned protocol is valid
    protocol_lines = {}
    with open(train_trial_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                protocol_lines[parts[1]] = line.strip()
    
    # Load metadata
    d_label_trn, file_trn = genSpoof_list(dir_meta=train_trial_path,
                                          is_train=True,
                                          is_eval=False)
    
    print(f"Total training samples: {len(file_trn)}")

    # Create Dataset and DataLoader
    # IMPORTANT: We disable augmentation (algo=0) to find intrinsically hard/noisy samples
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_trn,
                                           labels=d_label_trn,
                                           base_dir=database_path / f"ASVspoof2019_{track}_train",
                                           algo=0, # No RawBoost
                                           use_codec=False) # No codec aug in cleaning
    
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=False, # Important: No shuffle to keep order
                              drop_last=False,
                              num_workers=0) # Set to 0 to save RAM on CPU/prevent OOM

    # 4. Inference and Loss Calculation
    criterion = nn.CrossEntropyLoss(reduction='none') # We need loss per sample
    
    results = [] # Stores (filename, loss, true_label, pred_prob_correct_class)

    print("Starting inference on training set...")
    
    current_idx = 0

    use_amp = (device.type == "cuda") and args.amp
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device)
            
            with autocast_ctx:
                # DualStreamSEMamba returns (features, logits)
                feats, logits = model(batch_x, Freq_aug=False)
                
                # Calculate Loss per sample
                losses = criterion(logits, batch_y)
                
                # Calculate probabilities
                probs = torch.softmax(logits, dim=1)
            
            batch_size = batch_x.size(0)
            
            for i in range(batch_size):
                fname = file_trn[current_idx]
                loss_val = losses[i].item()
                label_val = batch_y[i].item()
                prob_correct = probs[i, label_val].item()
                
                results.append({
                    "file": fname,
                    "loss": loss_val,
                    "label": label_val,
                    "prob": prob_correct
                })
                current_idx += 1

    # 5. Filter Top N% Dirty Data
    print("Sorting samples by loss (descending)...")
    results.sort(key=lambda x: x["loss"], reverse=True)
    
    num_samples = len(results)
    num_dirty = int(num_samples * args.filter_ratio)
    
    dirty_samples = results[:num_dirty]
    clean_samples = results[num_dirty:]
    
    print(f"Top {args.filter_ratio*100}% dirty samples: {len(dirty_samples)}")
    print(f"Sample dirty entry: {dirty_samples[0]}")
    
    # 6. Save Results
    output_file = args.output_path
    print(f"Saving dirty sample list to {output_file}...")
    
    with open(output_file, "w") as f:
        for item in dirty_samples:
            f.write(f"{item['file']} {item['loss']:.6f} {item['label']}\n")
            
    # Also save a cleaned protocol file for easy usage
    clean_protocol_path = output_file.replace(".txt", "_cleaned_protocol.txt")
    print(f"Saving cleaned protocol to {clean_protocol_path}...")
    
    with open(clean_protocol_path, "w") as f:
        for item in clean_samples:
            key = item["file"]
            if key in protocol_lines:
                f.write(protocol_lines[key] + "\n")
            else:
                label_str = "bonafide" if item['label'] == 1 else "spoof"
                f.write(f"LA_0000 {key} - - {label_str}\n")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dirty training samples based on loss.")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--output_path", type=str, default="dirty_samples.txt", help="Output file for dirty samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--filter_ratio", type=float, default=0.02, help="Ratio of samples to filter (e.g., 0.02 for 2%)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--allow_cpu", action="store_true", help="Force CPU run (may be slow/oom for WavLM-Large)")
    parser.add_argument("--amp", action="store_true", help="Enable AMP autocast on CUDA during inference")
    
    args = parser.parse_args()
    main(args)

