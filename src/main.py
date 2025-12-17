"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
import numpy as np
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from torch.cuda.amp import autocast, GradScaler  # Import AMP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, 
                        Dataset_ASVspoof2021_eval,
                        genSpoof_list)
from evaluation import calculate_tDCF_EER
# from evaluation_2021 import calculate_EER_2021, produce_evaluation_file_2021
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from loss import OCSoftmax, SupConLoss
from sam import SAM

warnings.filterwarnings("ignore", category=FutureWarning)


def freeze_batch_norm_stats(model: nn.Module) -> None:
    """
    Freeze BatchNorm running statistics while keeping the model in train mode.
    This mirrors the torchvision detection training recipe.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def _parse_attack_eer_from_tdcf_report(report_path: Union[str, Path], attack_id: str) -> Union[float, None]:
    """
    Parse 'EER A18 = xx %' lines from evaluation.calculate_tDCF_EER output file.
    Returns EER in percent, or None if not found.
    """
    report_path = str(report_path)
    if not os.path.exists(report_path):
        return None
    pat = re.compile(rf"\\bEER\\s+{re.escape(attack_id)}\\b\\s*=\\s*([0-9.]+)\\s*%")
    try:
        with open(report_path, "r") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    return float(m.group(1))
    except Exception:
        return None
    return None


class FGM:
    """
    Fast Gradient Method for adversarial perturbation on embeddings.
    emb_name targets the WavLM projection layer ('feature_projection' by default).
    """
    def __init__(self, model: nn.Module, emb_name: str = "feature_projection", epsilon: float = 1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad is None:
                    continue
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def apply_lora_to_wavlm(model: nn.Module, training_config: dict) -> nn.Module:
    """
    Inject LoRA adapters into the WavLM frontend using the official PEFT library.
    Only activates when a wavlm_stream with a HuggingFace WavLM model is present.
    """
    use_lora = training_config.get("use_lora", False)
    if not use_lora:
        return model

    # 强制导入，如果失败则报错（不再静默跳过）
    from peft import LoraConfig, TaskType, get_peft_model

    lora_r = training_config.get("lora_r", 8)
    lora_alpha = training_config.get("lora_alpha", 32)
    lora_dropout = training_config.get("lora_dropout", 0.1)
    target_modules = training_config.get("lora_target_modules", ["q_proj", "v_proj"])

    if hasattr(model, "wavlm_stream") and hasattr(model.wavlm_stream, "model"):
        base_model = model.wavlm_stream.model
        for p in base_model.parameters():
            p.requires_grad = False  # 只训练 LoRA 权重

        lora_cfg = LoraConfig(
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            modules_to_save=None,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        # WavLM 可能不支持 gradient checkpointing，先禁用
        if hasattr(base_model, 'gradient_checkpointing_disable'):
            base_model.gradient_checkpointing_disable()
        try:
            model.wavlm_stream.model = get_peft_model(base_model, lora_cfg)
            print(f"🚀 [LoRA] Enabled on wavlm_stream.model (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
            try:
                model.wavlm_stream.model.print_trainable_parameters()
            except Exception:
                # 计算可训练参数数量
                trainable = sum(p.numel() for p in model.wavlm_stream.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.wavlm_stream.model.parameters())
                print(f"   trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.2f}%")
        except Exception as e:
            print(f"❌ LoRA injection failed: {e}")
            print("   Falling back to full fine-tuning without LoRA.")
            # 恢复原始模型
            model.wavlm_stream.model = base_model
            for p in base_model.parameters():
                p.requires_grad = True
    else:
        print("⚠️  LoRA requested but wavlm_stream.model not found. Skipping LoRA injection.")

    return model


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
    
    training_config = config.get("training_config", {})
    
    # Check for SupCon
    use_supcon = config.get("use_supcon", False) or training_config.get("use_supcon", False)
    if use_supcon:
        print("🌟 Using Supervised Contrastive Learning (SupCon)!")

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    
    # Check if evaluating ASVspoof 2021 dataset
    is_eval_2021 = config.get("is_eval_2021", False)
    if is_eval_2021 and args.eval:
        # ASVspoof 2021: protocol file is in the database_path directly
        eval_trial_path = database_path / "ASVspoof2021.DF.cm.eval.trl.txt"
        dev_trial_path = None  # Not used in 2021 eval mode
    else:
        # Standard ASVspoof 2019 mode
        prefix_2019 = "ASVspoof2019.{}".format(track)
        dev_trial_path = (database_path /
                          "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                              track, prefix_2019))
        eval_trial_path = (
            database_path /
            "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                track, prefix_2019))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        print("⚠️  Warning: Running on CPU. Training will be very slow!")
        print("   Consider using GPU for faster training.")

    # define model architecture
    model_name = args.model if args.model else model_config["architecture"]
    model_config["architecture"] = model_name
    model = get_model(model_config, device)
    # Optional: inject LoRA into WavLM frontend (only when explicitly enabled)
    model = apply_lora_to_wavlm(model, training_config)

    # Optional: freeze SincNet stream to force WavLM/LoRA to dominate (helps A18)
    if training_config.get("freeze_sincnet", False):
        print("🧊 [Config] Freezing SincNet stream to emphasize WavLM/LoRA.")
        if hasattr(model, "sinc_stream"):
            for p in model.sinc_stream.parameters():
                p.requires_grad = False

    # Load pretrained weights for fine-tuning (if provided)
    if args.pretrained_weights:
        print(f"Loading pretrained weights for fine-tuning: {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
        # Smart prefix removal (module.)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Pretrained weights loaded successfully (smart prefix handling).")

    # Resume from checkpoint if provided
    if args.resume:
        print("Resuming from checkpoint: {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
             model.load_state_dict(checkpoint["model_state_dict"])
        else:
             model.load_state_dict(checkpoint)
        print("Model weights loaded.")

    # define loss function
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    label_smoothing = training_config.get("label_smoothing", 0.0)
    criterion_ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    criterion_supcon = None

    use_focal_loss = (config.get("loss") == "Focal") or training_config.get("use_focal_loss", False)

    if config["loss"] == "OCSoftmax":
        # Infer feature dimension
        feat_dim = 160 
        if "Wav2Vec2" in model_config["architecture"]:
             feat_dim = 160
        if "CascadeMamba" in model_config["architecture"]:
             feat_dim = 128
        if "WavLMMamba" in model_config["architecture"] or "MoEMambaASV" in model_config["architecture"]:
             feat_dim = model_config.get("emb_size", 144) # Default to 144
        if "DualStreamSEMamba" in model_config["architecture"]:
             feat_dim = model_config.get("emb_size", 144) # Default to 144
        
        print(f"Using OCSoftmax with feature dim: {feat_dim}")
        criterion = OCSoftmax(feat_dim=feat_dim, 
                              r_real=config["training_config"].get("ocsoftmax_r_real", 0.9),
                              r_fake=config["training_config"].get("ocsoftmax_r_fake", 0.5),
                              alpha=config["training_config"].get("ocsoftmax_alpha", 20.0)
                              ).to(device)
        criterion_params = list(criterion.parameters())
    elif use_focal_loss:
        try:
            import kornia
            alpha = training_config.get("focal_alpha", 0.25)
            gamma = training_config.get("focal_gamma", 2.0)
            criterion = kornia.losses.FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
            criterion_params = []
            print(f"Using Focal Loss from kornia (alpha={alpha}, gamma={gamma})")
            config["loss"] = "Focal"
        except ImportError:
            print("⚠️  `kornia` not installed, falling back to CrossEntropyLoss. Install via `pip install kornia`.")
            criterion = criterion_ce
            criterion_params = []
    else:
        criterion = criterion_ce
        criterion_params = []
    
    if use_supcon:
        criterion_supcon = SupConLoss(temperature=0.07).to(device)

    # define dataloaders
    # Check if this is eval-only mode (e.g., ASVspoof 2021)
    is_eval_only = config.get("is_eval_2021", False) and args.eval
    # Pass eval flag to get_loader
    config["eval"] = args.eval
    if is_eval_only:
        _, _, eval_loader = get_loader(
            database_path, args.seed, config)
        trn_loader, dev_loader = None, None
    else:
        trn_loader, dev_loader, eval_loader = get_loader(
            database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        if args.eval_model_weights is not None:
            model_path = args.eval_model_weights
        else:
            model_path = config["model_path"]
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle DataParallel saved weights (remove "module." prefix if present)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove "module." prefix if model was saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove "module." prefix
            else:
                new_state_dict[k] = v
        
        # IMPORTANT: Apply LoRA if configured BEFORE loading weights
        # Otherwise keys like 'base_model.model...' won't match and will be ignored
        model = apply_lora_to_wavlm(model, training_config)
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded : {}".format(model_path))
        print("Start evaluation...")
        
        # Check if evaluating ASVspoof 2021 dataset
        is_eval_2021 = config.get("is_eval_2021", False)
        if is_eval_2021:
            print("📊 Evaluating ASVspoof 2021 DF dataset with EER calculation")
            # Use 2021 format evaluation function
            produce_evaluation_file_2021(eval_loader, model, device,
                                         eval_score_path, criterion)
            
            # Calculate EER using labels from keys file
            key_file = config.get("key_file", "./keys/DF/CM/trial_metadata.txt")
            if not Path(key_file).exists():
                # Try absolute path
                key_file = "/root/aasist-main/keys/DF/CM/trial_metadata.txt"
            
            if Path(key_file).exists():
                print(f"📋 Loading labels from: {key_file}")
                eer_result_file = model_tag / "t-DCF_EER_2021DF.txt"
                eer, eer_breakdown = calculate_EER_2021(
                    cm_scores_file=eval_score_path,
                    key_file=key_file,
                    output_file=eer_result_file,
                    printout=True
                )
                print("DONE. EER calculation complete.")
                print(f"✅ ASVspoof 2021 DF EER: {eer:.4f}%")
            else:
                print(f"⚠️  Warning: Key file not found at {key_file}")
                print("   Scores saved but EER cannot be calculated.")
            print("DONE. Scores saved to: {}".format(eval_score_path))
        else:
            produce_evaluation_file(eval_loader, model, device,
                                    eval_score_path, eval_trial_path, criterion)
            calculate_tDCF_EER(cm_scores_file=eval_score_path,
                               asv_score_file=database_path /
                               config["asv_score_path"],
                               output_file=model_tag / "t-DCF_EER.txt")
            print("DONE.")
            eval_eer, eval_tdcf = calculate_tDCF_EER(
                cm_scores_file=eval_score_path,
                asv_score_file=database_path / config["asv_score_path"],
                output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # get optimizer and scheduler
    # IMPORTANT: with gradient accumulation we should schedule per *optimizer step* (not per micro-batch)
    accumulation_steps = max(1, int(training_config.get("accumulation_steps", 1)))
    optim_steps_per_epoch = math.ceil(len(trn_loader) / accumulation_steps)
    optim_config["steps_per_epoch"] = optim_steps_per_epoch
    print(f"⚙️ [Schedule] micro-batches/epoch={len(trn_loader)}, accumulation_steps={accumulation_steps} -> optimizer_steps/epoch={optim_steps_per_epoch}")
    
    # --- Differential Learning Rate Implementation (修正版) ---
    print("⚡ Implementing Differential Learning Rate Strategy")
    
    wavlm_params = []
    backbone_params = []
    loss_params = list(criterion.parameters()) if hasattr(criterion, "parameters") else []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # === 修正: 更精确的匹配，防止误伤 ===
        # 只有真正的 WavLM 前端 (wavlm_stream) 使用极低学习率
        # 注意: DualStreamFusion 中的 wavlm_proj 应该用正常学习率
        # 因为它是从头训练的投影层，不是预训练的 WavLM 权重
        if "wavlm_stream" in name:
            wavlm_params.append(param)
        else:
            # SincNet, Fusion (包括 wavlm_proj), BiMamba, Classifier 都用正常学习率
            # SincNet 是随机初始化，必须用较大学习率才能收敛
            backbone_params.append(param)
            
    # 读取 WavLM 专用学习率 (Phase 5 建议 5e-6，Phase 4 是 1e-6)
    wavlm_lr = optim_config.get('wavlm_lr', 1e-6)
    # 读取 freeze_layers 仅仅为了打印日志确认
    freeze_layers = model_config.get('wavlm_freeze_layers', 18)
    
    print(f"🚀 [Config Check] WavLM Freeze Layers: {freeze_layers}")
    print(f"🚀 [Config Check] WavLM Learning Rate: {wavlm_lr}")
    print(f"   - WavLM/SSL Params: {len(wavlm_params)} tensors (LR: {wavlm_lr})")
    print(f"   - Backbone Params: {len(backbone_params)} tensors (LR: {optim_config['base_lr']})")
    print(f"   - Loss Params: {len(loss_params)} tensors (LR: {optim_config['base_lr']})")
    
    # 验证参数分组是否正确
    if len(wavlm_params) == 0:
        print("⚠️  Warning: No WavLM parameters found! Check model architecture.")
    if len(backbone_params) == 0:
        print("⚠️  Warning: No backbone parameters found! Check model architecture.")
    
    # Create Optimizer with parameter groups
    optimizer = torch.optim.AdamW([
        {'params': wavlm_params, 'lr': wavlm_lr},       # WavLM专用LR (Phase 4: 1e-6, Phase 5: 5e-6)
        {'params': backbone_params, 'lr': optim_config['base_lr']}, # Normal LR for Backbone
        {'params': loss_params, 'lr': optim_config['base_lr']}      # Normal LR for Loss
    ], weight_decay=optim_config['weight_decay'])
    
    # Create Scheduler (Warmup + Cosine)
    total_steps = optim_config["epochs"] * optim_config["steps_per_epoch"]
    warmup_ratio = float(training_config.get("warmup_ratio", 0.05))
    warmup_steps = int(training_config.get("warmup_steps", max(1, int(total_steps * warmup_ratio))))
    warmup_steps = min(max(1, warmup_steps), max(1, total_steps - 1))
    print(f"🔥 [Schedule] total_steps={total_steps}, warmup_steps={warmup_steps} (ratio={warmup_ratio})")

    # Linear warmup from warmup_init_factor -> 1.0
    warmup_init_factor = float(training_config.get("warmup_init_factor", 0.1))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_init_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=optim_config["scheduler_config"]["eta_min"],
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    # Gradient Scaler for AMP
    scaler = GradScaler()

    optimizer_swa = SWA(optimizer)

    # Optional EMA (official torch.optim.swa_utils)
    use_ema = training_config.get("use_ema", False)
    ema_decay = training_config.get("ema_decay", 0.999)
    ema_model = None
    if use_ema:
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        print(f"[EMA] Enabled with decay={ema_decay}")

    best_dev_eer = 100.
    best_eval_eer = 100.
    best_dev_tdcf = 100.
    best_eval_tdcf = 100.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training controls
    start_epoch = args.start_epoch
    freeze_bn = training_config.get("freeze_bn", False)

    # FGM Logic Setup
    fgm = None
    if training_config.get("use_fgm", False):
        print("🛡️ [FGM] Initializing Adversarial Training...")
        
        # Explicitly unfreeze feature_projection for FGM if targeting it
        fgm_emb_name = training_config.get("fgm_emb_name", "feature_projection")
        if "feature_projection" in fgm_emb_name and hasattr(model, "wavlm_stream"):
            print("🔓 [FGM] Unfreezing WavLM feature_projection layer for attack target.")
            model.wavlm_stream.model.feature_projection.requires_grad_(True)
            # Ensure it's in the optimizer
            found = False
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p is model.wavlm_stream.model.feature_projection.projection.weight:
                        found = True
                        break
            if not found:
                print("⚠️ [FGM] feature_projection was not in optimizer. Adding it now.")
                optimizer.add_param_group({'params': model.wavlm_stream.model.feature_projection.parameters(), 'lr': wavlm_lr})
                # torchcontrib SWA expects every param_group to have 'n_avg'.
                # Since we are adding a new param_group after SWA(optimizer) is created,
                # initialize it here to prevent KeyError: 'n_avg' during update_swa().
                if "n_avg" not in optimizer.param_groups[-1]:
                    optimizer.param_groups[-1]["n_avg"] = 0

        fgm = FGM(
            model,
            emb_name=fgm_emb_name,
            epsilon=training_config.get("fgm_epsilon", 1.0),
        )

    for epoch in range(start_epoch, config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        
        # Pass scaler to train_epoch
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config, criterion, criterion_supcon, scaler,
                                   freeze_bn=freeze_bn, ema_model=ema_model,
                                   accumulation_steps=accumulation_steps,
                                   fgm=fgm) 
                                   
        eval_model = ema_model if ema_model is not None else model

        produce_evaluation_file(dev_loader, eval_model, device,
                                metric_path/"dev_score.txt", dev_trial_path, criterion)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)

        # ============================
        # Eval Diagnostics (ASVspoof2019 LA eval)
        # ============================
        # IMPORTANT: This is diagnostic only. Best model selection remains based on Dev EER (blind selection).
        eval_diag_interval = int(training_config.get("eval_diag_interval", 0))
        if eval_diag_interval > 0 and ((epoch + 1) % eval_diag_interval == 0):
            print("\n" + "-" * 70)
            print(f"🔎 [Eval-Diag] Running ASVspoof2019 LA eval diagnostics at epoch {epoch} (interval={eval_diag_interval})")
            print("-" * 70)
            diag_score_path = metric_path / f"eval_diag_score_{epoch:03d}.txt"
            diag_report_path = metric_path / f"eval_diag_tDCF_EER_{epoch:03d}.txt"
            try:
                # 1) inference on eval set (2019 LA eval)
                produce_evaluation_file(eval_loader, eval_model, device, diag_score_path, eval_trial_path, criterion)
                # 2) compute tDCF/EER report (contains per-attack EER when printout=True)
                calculate_tDCF_EER(
                    cm_scores_file=diag_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=diag_report_path,
                    printout=True
                )
                # 3) parse A18/A19 from report
                a18 = _parse_attack_eer_from_tdcf_report(diag_report_path, "A18")
                a19 = _parse_attack_eer_from_tdcf_report(diag_report_path, "A19")
                print(f"📌 [Eval-Diag] A18 EER: {a18 if a18 is not None else 'N/A'}% | A19 EER: {a19 if a19 is not None else 'N/A'}%")
            except Exception as e:
                print(f"⚠️  [Eval-Diag] Failed at epoch {epoch}: {e}")
            print("-" * 70 + "\n")

        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch, flush=True)
            best_dev_eer = dev_eer
            
            # === 优化：只保存一个最佳模型，删除旧的 ===
            # 删除之前保存的所有最佳模型（避免累积多个文件占用空间）
            for old_file in model_save_path.glob("epoch_*_*.pth"):
                if old_file.name != f"epoch_{epoch}_{dev_eer:03.3f}.pth":
                    try:
                        old_file.unlink()
                    except:
                        pass
            
            # Save best model with EER in filename
            best_model_path = model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer)
            torch.save(eval_model.state_dict(), best_model_path)
            print("Saved best model: {}".format(best_model_path), flush=True)

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, eval_model, device,
                                        eval_score_path, eval_trial_path, criterion)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                    best_eval_tdcf = eval_tdcf
                    torch.save(eval_model.state_dict(),
                               model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text, flush=True)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch), flush=True)
            optimizer_swa.update_swa()
            n_swa_update += 1
        
        # === 优化：定期保存checkpoint，但限制数量 ===
        # 只保留最近3个checkpoint，删除更早的以节省空间
        save_checkpoint_interval = 10
        if (epoch + 1) % save_checkpoint_interval == 0 or epoch == config["num_epochs"] - 1:
            checkpoint_path = model_save_path / "checkpoint_epoch_{:03d}.pth".format(epoch)
            torch.save(model.state_dict(), checkpoint_path)
            print("Checkpoint saved: {}".format(checkpoint_path), flush=True)
            
            # 只保留最近3个checkpoint，删除更早的
            checkpoint_files = sorted(model_save_path.glob("checkpoint_epoch_*.pth"), 
                                     key=lambda x: int(x.stem.split('_')[-1]))
            if len(checkpoint_files) > 3:
                for old_checkpoint in checkpoint_files[:-3]:
                    try:
                        old_checkpoint.unlink()
                        print("Deleted old checkpoint: {}".format(old_checkpoint.name), flush=True)
                    except:
                        pass

        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    final_eval_model = ema_model if ema_model is not None else model
    produce_evaluation_file(eval_loader, final_eval_model, device, eval_score_path,
                            eval_trial_path, criterion)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
                                             asv_score_file=database_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(final_eval_model.state_dict(),
                   model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))
    
    # === 自动评估ASVspoof 2021 DF数据集 ===
    auto_eval_2021 = config.get("auto_eval_2021_df", False)
    if auto_eval_2021:
        print("\n" + "=" * 70)
        print("🚀 自动评估 ASVspoof 2021 DF 数据集")
        print("=" * 70)
        
        # 检查2021数据集路径配置
        database_path_2021 = config.get("database_path_2021", None)
        key_file_2021 = config.get("key_file_2021", "/root/aasist-main/keys/DF/CM/trial_metadata.txt")
        
        if database_path_2021 is None:
            print("⚠️  Warning: database_path_2021 not configured in config file.")
            print("   Skipping automatic 2021 DF evaluation.")
        else:
            try:
                # 加载最佳模型
                best_model_path = model_save_path / "best.pth"
                if not best_model_path.exists():
                    # 如果没有best.pth，使用最新的epoch模型
                    epoch_models = list(model_save_path.glob("epoch_*_*.pth"))
                    if epoch_models:
                        best_model_path = sorted(epoch_models, key=lambda x: x.stat().st_mtime)[-1]
                        print(f"⚠️  Using latest epoch model: {best_model_path.name}")
                    else:
                        print("❌ Error: No model found for 2021 evaluation.")
                        return
                
                print(f"📦 Loading best model: {best_model_path}")
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                model.eval()
                
                # 准备2021数据集
                eval_database_path_2021 = Path(database_path_2021)
                eval_trial_path_2021 = eval_database_path_2021 / "ASVspoof2021.DF.cm.eval.trl.txt"
                
                if not eval_trial_path_2021.exists():
                    print(f"❌ Error: Protocol file not found: {eval_trial_path_2021}")
                    return
                
                print(f"📊 Loading ASVspoof 2021 DF evaluation dataset...")
                file_eval_2021 = genSpoof_list(dir_meta=eval_trial_path_2021,
                                               is_train=False,
                                               is_eval=True,
                                               is_2021=True)
                print(f"   Total evaluation files: {len(file_eval_2021)}")
                
                eval_set_2021 = Dataset_ASVspoof2021_eval(list_IDs=file_eval_2021,
                                                          base_dir=eval_database_path_2021)
                
                test_batch_size = config.get("test_config", {}).get("batch_size", 32)
                num_workers = config.get("test_config", {}).get("num_workers", 8)
                
                eval_loader_2021 = DataLoader(eval_set_2021,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              pin_memory=True,
                                              num_workers=num_workers)
                
                # 生成评估分数文件
                eval_score_path_2021 = model_tag / "eval_scores_2021DF.txt"
                print(f"\n📝 Generating scores for 2021 DF dataset...")
                produce_evaluation_file_2021(eval_loader_2021, model, device,
                                             str(eval_score_path_2021), criterion)
                
                # 计算EER
                if Path(key_file_2021).exists():
                    print(f"\n📋 Loading labels from: {key_file_2021}")
                    eer_result_file_2021 = model_tag / "t-DCF_EER_2021DF.txt"
                    eer_2021, eer_breakdown_2021 = calculate_EER_2021(
                        cm_scores_file=str(eval_score_path_2021),
                        key_file=key_file_2021,
                        output_file=str(eer_result_file_2021),
                        printout=True
                    )
                    print("\n" + "=" * 70)
                    print(f"🎯 ASVspoof 2021 DF 评估完成")
                    print("=" * 70)
                    print(f"✅ Overall EER: {eer_2021:.4f}%")
                    print(f"📁 结果文件: {eer_result_file_2021}")
                    print("=" * 70)
                    
                    # 将2021评估结果写入日志
                    f_log = open(model_tag / "metric_log.txt", "a")
                    f_log.write("\n" + "=" * 5 + "\n")
                    f_log.write("ASVspoof 2021 DF Evaluation (Cross-domain):\n")
                    f_log.write("EER: {:.4f}%\n".format(eer_2021))
                    f_log.close()
                else:
                    print(f"⚠️  Warning: Key file not found: {key_file_2021}")
                    print("   Scores saved but EER cannot be calculated.")
                    print(f"   Score file: {eval_score_path_2021}")
                
            except Exception as e:
                print(f"❌ Error during automatic 2021 DF evaluation: {e}")
                import traceback
                traceback.print_exc()
                print("   Continuing without 2021 evaluation...")


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    # Pass full args to model
    class Args:
        def __init__(self, d):
            self.__dict__.update(d)
    args = Args(model_config)
    model = _model(args, device).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    
    # Convert to Path if string
    if isinstance(database_path, str):
        database_path = Path(database_path)
    
    # Check if evaluating ASVspoof 2021 dataset
    is_eval_2021 = config.get("is_eval_2021", False)
    is_eval_mode = config.get("eval", False)
    
    if is_eval_2021 and is_eval_mode:
        # ASVspoof 2021 evaluation mode (only eval set, no train/dev)
        print("📊 Loading ASVspoof 2021 DF evaluation dataset...")
        eval_database_path = Path(config["database_path"])
        eval_trial_path = eval_database_path / "ASVspoof2021.DF.cm.eval.trl.txt"
        
        file_eval = genSpoof_list(dir_meta=eval_trial_path,
                                  is_train=False,
                                  is_eval=True,
                                  is_2021=True)
        print(f"no. evaluation files: {len(file_eval)}")
        
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
        
        test_batch_size = config.get("test_config", {}).get("batch_size", 32)
        num_workers = config.get("test_config", {}).get("num_workers", 8)
        
        print(f"Using Test Batch Size: {test_batch_size}, Workers: {num_workers}")
        
        eval_loader = DataLoader(eval_set,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True,
                                 num_workers=num_workers)
        
        # Return None for train/dev loaders in eval-only mode
        return None, None, eval_loader
    
    # Standard ASVspoof 2019 mode
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    # Allow custom training protocol from config (for Data Cleaning/Filtering)
    if "custom_train_protocol" in config["data_config"]:
        trn_list_path = Path(config["data_config"]["custom_train_protocol"])
        print(f"📋 Using Custom Training Protocol: {trn_list_path}")
    else:
        trn_list_path = (database_path /
                         "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                             track, prefix_2019))
                         
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    # Check for RawBoost algo in config
    rawboost_algo = int(config["data_config"].get("rawboost_algo", 0)) if "data_config" in config else 0
    use_codec_aug = str_to_bool(config["data_config"].get("use_codec_aug", "False")) if "data_config" in config else False
    codec_p = float(config["data_config"].get("codec_p", 0.5)) if "data_config" in config else 0.5
    rawboost_p = float(config["data_config"].get("rawboost_p", 1.0)) if "data_config" in config else 1.0
    
    if rawboost_algo != 0:
        print(f"RawBoost Augmentation Enabled (Algo: {rawboost_algo})")
    if use_codec_aug:
        print(f"Codec Augmentation Enabled (p={codec_p})")
    if rawboost_algo != 0:
        print(f"RawBoost Apply Probability (p={rawboost_p})")

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path,
                                           algo=rawboost_algo,
                                           use_codec=use_codec_aug,
                                           codec_p=codec_p,
                                           rawboost_p=rawboost_p)
    gen = torch.Generator()
    gen.manual_seed(seed)
    # === 关键设置：drop_last=True ===
    # 防止最后一个 batch 只有 1 个样本导致 BatchNorm/LayerNorm 计算出错
    # 这对于使用 LayerNorm 的模型（如我们的 DualStreamFusion）非常重要
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,  # 必须为 True，防止 BatchNorm/LayerNorm 在 batch_size=1 时出错
                            pin_memory=True,
                            worker_init_fn=seed_worker,  # 确保每个 worker 的随机种子正确
                            generator=gen)  # 使用固定的 generator 确保 shuffle 的可复现性

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    
    # Use larger batch size for validation/eval if specified in config
    test_batch_size = config.get("test_config", {}).get("batch_size", config["batch_size"])
    num_workers = config.get("test_config", {}).get("num_workers", 4)
    
    print(f"Using Test Batch Size: {test_batch_size}, Workers: {num_workers}")
                                            
    dev_loader = DataLoader(dev_set,
                            batch_size=test_batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=num_workers)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=num_workers)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    criterion=None) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            # AMP disabled for inference/eval usually fine, but can enable if OOM
            # with autocast(): 
            batch_feats, batch_out = model(batch_x)
            
            if criterion is not None and isinstance(criterion, OCSoftmax):
                import torch.nn.functional as F
                w = F.normalize(criterion.center, p=2, dim=1)
                feats_norm = F.normalize(batch_feats, p=2, dim=1)
                batch_score = (feats_norm.mm(w.transpose(0, 1))).view(-1).data.cpu().numpy().ravel()
            else:
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
    criterion: nn.Module,
    criterion_supcon: nn.Module = None,
    scaler: GradScaler = None,
    freeze_bn: bool = False,
    ema_model: AveragedModel = None,
    accumulation_steps: int = 1,
    fgm: FGM = None):  # Add scaler arg
    """Train the model for one epoch with optional BN freeze, EMA, gradient accumulation, and FGM"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()
    if freeze_bn:
        freeze_batch_norm_stats(model)
    
    lambda_supcon = config.get("lambda_supcon", 0.1)
    
    # Mixup Configuration
    training_config = config.get("training_config", {})
    use_mixup = training_config.get("use_mixup", False)
    mixup_alpha = training_config.get("mixup_alpha", 1.0)
    
    accumulation_steps = max(1, accumulation_steps)
    optim.zero_grad()

    for i, (batch_x, batch_y) in enumerate(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # Mixup Data Preparation
        if use_mixup and batch_size > 1:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)
            mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
            y_a, y_b = batch_y, batch_y[index]
        else:
            mixed_x = batch_x
            lam = 1.0
            y_a, y_b = batch_y, batch_y

        # AMP Context
        with autocast():
            # Forward pass
            batch_feats, batch_out = model(mixed_x, Freq_aug=str_to_bool(config["freq_aug"]))
            
            # Loss calculation function
            def compute_loss(out, feats, target):
                if isinstance(criterion, OCSoftmax):
                    loss = criterion(feats, target)
                else:
                    loss = criterion(out, target)
                
                if criterion_supcon is not None:
                    feats_norm = F.normalize(feats, dim=1)
                    s_loss = criterion_supcon(feats_norm, target)
                    loss = loss + lambda_supcon * s_loss
                return loss

            if use_mixup and batch_size > 1:
                loss_a = compute_loss(batch_out, batch_feats, y_a)
                loss_b = compute_loss(batch_out, batch_feats, y_b)
                batch_loss = lam * loss_a + (1 - lam) * loss_b
            else:
                batch_loss = compute_loss(batch_out, batch_feats, batch_y)

        # Gradient accumulation
        batch_loss = batch_loss / accumulation_steps

        # Backward pass with scaler
        scaler.scale(batch_loss).backward()
        
        # FGM adversarial step (only when enabled)
        if fgm is not None:
            fgm.attack()
            with autocast():
                # For FGM, we typically use the original (non-mixed) batch or the mixed batch?
                # Standard practice: Attack the mixed batch if Mixup is on, 
                # OR just attack the clean batch.
                # Attacking mixed batch is consistent with the forward pass.
                adv_feats, adv_out = model(mixed_x, Freq_aug=str_to_bool(config["freq_aug"]))
                
                if use_mixup and batch_size > 1:
                    loss_a = compute_loss(adv_out, adv_feats, y_a)
                    loss_b = compute_loss(adv_out, adv_feats, y_b)
                    adv_loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    adv_loss = compute_loss(adv_out, adv_feats, batch_y)
                    
                adv_loss = adv_loss / accumulation_steps
            scaler.scale(adv_loss).backward()
            fgm.restore()

        do_step = ((i + 1) % accumulation_steps == 0) or (i + 1 == len(trn_loader))
        if do_step:
            # Unscale before clipping
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            
            # Step with scaler
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            # EMA update after optimizer step
            if ema_model is not None:
                ema_model.update_parameters(model)
                
            # Scheduler step - MOVED INSIDE do_step block to sync with optimizer
            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()

        # Scheduler is stepped inside the optimizer-step block above (do_step=True).
        # We intentionally do nothing here to avoid stepping on micro-batches.
            
        # running_loss uses the true (non-divided) loss magnitude for logging
        running_loss += (batch_loss.item() * accumulation_steps) * batch_size
        
    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="path to checkpoint to resume training from")
    parser.add_argument("--start_epoch",
                        type=int,
                        default=0,
                        help="epoch to start training from")
    parser.add_argument("--pretrained_weights",
                        type=str,
                        default=None,
                        help="path to pretrained weights to initialize model (training mode)")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Model architecture name (override config)")
    main(parser.parse_args())
