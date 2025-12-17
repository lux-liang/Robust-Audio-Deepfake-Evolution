import argparse
import json
import os
import sys
from pathlib import Path
from importlib import import_module

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list
from utils import str_to_bool

def get_model(model_config, device):
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    return model

def main(args):
    # Load config
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Initialize model
    model = get_model(config["model_config"], device)
    
    # Load weights
    if args.model_path:
        weights_path = args.model_path
    else:
        # Try to find best.pth in expected location
        track = config["track"]
        model_tag = "{}_{}_ep{}_bs{}".format(
            track,
            os.path.splitext(os.path.basename(args.config))[0],
            config["num_epochs"], config["batch_size"])
        if args.comment:
            model_tag = model_tag + "_{}".format(args.comment)
        weights_path = Path(args.output_dir) / model_tag / "weights" / "best.pth"

    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")
    model.eval()
    
    # Prepare data loader (Validation set)
    database_path = Path(config["database_path"])
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    
    # We need labels for visualization
    # data_utils.genSpoof_list returns (dict, list) for train, but list for dev/eval
    # We need to parse labels manually for dev set
    print("Parsing dev labels...")
    d_label_dev = {}
    file_list = []
    with open(dev_trial_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_label_dev[key] = 1 if label == "bonafide" else 0 # 1=Bonafide, 0=Spoof
            
    # For t-SNE we don't need all data, just a subset to save time
    # Let's pick N samples
    N_SAMPLES = 2000
    if len(file_list) > N_SAMPLES:
        import random
        random.shuffle(file_list)
        file_list = file_list[:N_SAMPLES]
        
    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_list, base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=False)
    
    # Extract Features
    print("Extracting features...")
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch_x, keys in dev_loader:
            batch_x = batch_x.to(device)
            batch_feats, _ = model(batch_x, Freq_aug=False)
            embeddings.append(batch_feats.cpu().numpy())
            
            # Get labels for this batch
            batch_labels = [d_label_dev[k] for k in keys]
            labels.extend(batch_labels)
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    
    print(f"Running t-SNE on {embeddings.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot Spoof (0)
    idx_spoof = np.where(labels == 0)[0]
    plt.scatter(X_embedded[idx_spoof, 0], X_embedded[idx_spoof, 1], 
                c='red', label='Spoof', alpha=0.5, s=10)
    
    # Plot Bonafide (1)
    idx_bonafide = np.where(labels == 1)[0]
    plt.scatter(X_embedded[idx_bonafide, 0], X_embedded[idx_bonafide, 1], 
                c='blue', label='Bonafide', alpha=0.5, s=10)
    
    plt.legend()
    plt.title(f"t-SNE Visualization - {config['model_config']['architecture']}")
    plt.grid(True, alpha=0.3)
    
    output_png = Path(args.output_dir) / "tsne_vis.png"
    plt.savefig(output_png)
    print(f"Saved t-SNE plot to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Features with t-SNE")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./exp_result")
    parser.add_argument("--model_path", type=str, help="Optional path to .pth file")
    parser.add_argument("--comment", type=str, default=None)
    
    args = parser.parse_args()
    main(args)

