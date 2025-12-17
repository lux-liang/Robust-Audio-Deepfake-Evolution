import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import json
from main import get_model, get_loader
from utils import str_to_bool
import os
from pathlib import Path

def extract_embeddings(model, loader, device):
    print("Starting embedding extraction...")
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(loader)}")
            
            batch_x = batch_x.to(device)
            
            # Using the model to get embeddings
            # Assuming model forward returns (features, logits)
            # features is what we want (the representation before classification)
            batch_feats, _ = model(batch_x)
            
            # If batch_feats is 3D (B, T, D), we need to pool it
            if len(batch_feats.shape) == 3:
                # Global Average Pooling
                batch_feats = batch_feats.mean(dim=1)
            
            embeddings.append(batch_feats.cpu().numpy())
            labels.append(batch_y.numpy())
            
            # Limit for visualization speed (optional, remove if you want full set)
            if len(embeddings) * loader.batch_size > 2000:
                print("Collected enough samples for visualization, stopping early.")
                break

    return np.concatenate(embeddings), np.concatenate(labels)

def plot_tsne(embeddings, labels, save_path="tsne_supcon.png"):
    print(f"Calculating t-SNE for {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    # 0: Spoof, 1: Bonafide (Check your dataset label mapping!)
    # Usually in AASIST: 1 is bonafide, 0 is spoof
    # Let's assume 1=Bonafide (Blue), 0=Spoof (Red)
    
    # Plot Spoof first (background)
    plt.scatter(X_tsne[labels==0, 0], X_tsne[labels==0, 1], c='red', label='Spoof', alpha=0.3, s=10)
    
    # Plot Bonafide (foreground)
    plt.scatter(X_tsne[labels==1, 0], X_tsne[labels==1, 1], c='blue', label='Bonafide', alpha=0.6, s=10)
    
    plt.legend()
    plt.title("Feature Distribution with SupCon (Phase 2.5)")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ t-SNE plot saved to {save_path}")

def main():
    # Hardcoded config for quick execution
    config_path = "config/CascadeMamba_SupCon.conf"
    # Use the newly finetuned model (or base model for comparison)
    # We will look for the best model in the output dir, or fallback to base
    model_path = "exp_result/cascade_mamba/LA_CascadeMamba_ep100_bs16/weights/epoch_39_0.206.pth"
    
    # Check if finetuned model exists
    finetuned_dir = "exp_result/cascade_mamba_supcon_finetune/weights"
    if os.path.exists(finetuned_dir):
        files = sorted(os.listdir(finetuned_dir))
        if files:
            # Find the latest epoch
            best_model = files[-1]
            model_path = os.path.join(finetuned_dir, best_model)
            print(f"Found finetuned model: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Config
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    
    # Load Model
    print(f"Loading model from {model_path}")
    model = get_model(config["model_config"], device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load Data (Use Dev set for visualization)
    print("Loading DataLoader...")
    # We only need dev loader
    database_path = Path(config["database_path"])
    # We can reuse get_loader but it might be slow to load train set.
    # Let's just init dev set manually if possible, or use get_loader
    # Using get_loader is safer to ensure consistency
    _, dev_loader, _ = get_loader(database_path, 1234, config)
    
    # Extract
    emb, lbl = extract_embeddings(model, dev_loader, device)
    
    # Plot
    plot_tsne(emb, lbl, save_path="supcon_tsne_visualization.png")

if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import json
from main import get_model, get_loader
from utils import str_to_bool
import os
from pathlib import Path

def extract_embeddings(model, loader, device):
    print("Starting embedding extraction...")
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(loader)}")
            
            batch_x = batch_x.to(device)
            
            # Using the model to get embeddings
            # Assuming model forward returns (features, logits)
            # features is what we want (the representation before classification)
            batch_feats, _ = model(batch_x)
            
            # If batch_feats is 3D (B, T, D), we need to pool it
            if len(batch_feats.shape) == 3:
                # Global Average Pooling
                batch_feats = batch_feats.mean(dim=1)
            
            embeddings.append(batch_feats.cpu().numpy())
            labels.append(batch_y.numpy())
            
            # Limit for visualization speed (optional, remove if you want full set)
            if len(embeddings) * loader.batch_size > 2000:
                print("Collected enough samples for visualization, stopping early.")
                break

    return np.concatenate(embeddings), np.concatenate(labels)

def plot_tsne(embeddings, labels, save_path="tsne_supcon.png"):
    print(f"Calculating t-SNE for {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    # 0: Spoof, 1: Bonafide (Check your dataset label mapping!)
    # Usually in AASIST: 1 is bonafide, 0 is spoof
    # Let's assume 1=Bonafide (Blue), 0=Spoof (Red)
    
    # Plot Spoof first (background)
    plt.scatter(X_tsne[labels==0, 0], X_tsne[labels==0, 1], c='red', label='Spoof', alpha=0.3, s=10)
    
    # Plot Bonafide (foreground)
    plt.scatter(X_tsne[labels==1, 0], X_tsne[labels==1, 1], c='blue', label='Bonafide', alpha=0.6, s=10)
    
    plt.legend()
    plt.title("Feature Distribution with SupCon (Phase 2.5)")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ t-SNE plot saved to {save_path}")

def main():
    # Hardcoded config for quick execution
    config_path = "config/CascadeMamba_SupCon.conf"
    # Use the newly finetuned model (or base model for comparison)
    # We will look for the best model in the output dir, or fallback to base
    model_path = "exp_result/cascade_mamba/LA_CascadeMamba_ep100_bs16/weights/epoch_39_0.206.pth"
    
    # Check if finetuned model exists
    finetuned_dir = "exp_result/cascade_mamba_supcon_finetune/weights"
    if os.path.exists(finetuned_dir):
        files = sorted(os.listdir(finetuned_dir))
        if files:
            # Find the latest epoch
            best_model = files[-1]
            model_path = os.path.join(finetuned_dir, best_model)
            print(f"Found finetuned model: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Config
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    
    # Load Model
    print(f"Loading model from {model_path}")
    model = get_model(config["model_config"], device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load Data (Use Dev set for visualization)
    print("Loading DataLoader...")
    # We only need dev loader
    database_path = Path(config["database_path"])
    # We can reuse get_loader but it might be slow to load train set.
    # Let's just init dev set manually if possible, or use get_loader
    # Using get_loader is safer to ensure consistency
    _, dev_loader, _ = get_loader(database_path, 1234, config)
    
    # Extract
    emb, lbl = extract_embeddings(model, dev_loader, device)
    
    # Plot
    plot_tsne(emb, lbl, save_path="supcon_tsne_visualization.png")

if __name__ == "__main__":
    main()









































