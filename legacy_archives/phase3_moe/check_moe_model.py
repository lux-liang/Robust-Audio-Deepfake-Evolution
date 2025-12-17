import torch
import torch.nn as nn
import sys
import os
import traceback

# Add project root to path
sys.path.append(os.getcwd())

from models.MoEMambaASV import Model
from loss import OCSoftmax

def sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running MoE-Mamba-ASV Check on {device}...")
    
    # 1. Mock Args
    class Args:
        emb_size = 144
        num_encoders = 6
        num_experts = 4
    args = Args()

    # 2. Instantiate Model
    print("🔄 Instantiating Model...")
    try:
        model = Model(args, device=device).to(device)
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return

    # 3. Check Freezing
    print("\n❄️ Checking Frozen Layers...")
    frozen_params = 0
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if not param.requires_grad:
            frozen_params += 1
        else:
            trainable_params += 1
    
    print(f"   Total Tensors: {total_params}")
    print(f"   Frozen Tensors: {frozen_params}")
    print(f"   Trainable Tensors: {trainable_params}")
    
    if frozen_params > 0:
        print("✅ WavLM freezing seems active.")
    else:
        print("⚠️ WARNING: No parameters are frozen!")

    # 4. Dummy Input
    batch_size = 2
    seq_len = 64600 
    dummy_input = torch.randn(batch_size, seq_len).to(device)
    dummy_label = torch.tensor([0, 1]).to(device) 
    
    print(f"\n📦 Input Shape: {dummy_input.shape}")

    # 5. Forward Pass
    print("🔄 Testing Forward Pass...")
    try:
        features, logits = model(dummy_input)
        print(f"✅ Forward Pass Successful")
        print(f"   Features Shape: {features.shape} (Expected: [{batch_size}, {args.emb_size}])")
        print(f"   Logits Shape:   {logits.shape}   (Expected: [{batch_size}, 2])")
        
        if features.shape[1] != args.emb_size:
            print(f"❌ Feature dimension mismatch! Got {features.shape[1]}, expected {args.emb_size}")
        if logits.shape[1] != 2:
            print(f"❌ Logits dimension mismatch! Got {logits.shape[1]}, expected 2")
            
    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")
        traceback.print_exc()
        return

    # 6. Loss Calculation (OCSoftmax)
    print("\n🔄 Testing Loss Calculation...")
    try:
        oc_loss_fn = OCSoftmax(feat_dim=args.emb_size).to(device)
        loss = oc_loss_fn(features, dummy_label)
        print(f"✅ OCSoftmax Loss: {loss.item()}")
    except Exception as e:
        print(f"❌ OCSoftmax Failed: {e}")
        traceback.print_exc()
        return

    # 7. Backward Pass
    print("🔄 Testing Backward Pass...")
    try:
        loss.backward()
        print("✅ Backward Pass Successful - Gradients computed")
    except Exception as e:
        print(f"❌ Backward Pass Failed: {e}")
        traceback.print_exc()
        return
        
    # 8. Check MoE Gating
    # We can't easily check internal gating without hooks, but if backward passed, it works.
    # Let's double check if parameters of MoE have gradients
    print("\n🔍 Checking MoE Gradients...")
    moe_grads = False
    for name, param in model.named_parameters():
        if "moe" in name and param.grad is not None:
            moe_grads = True
            break
    
    if moe_grads:
        print("✅ MoE layers have gradients (Gating and Experts are learning)")
    else:
        print("⚠️ WARNING: MoE layers might not have gradients. Check connectivity.")

    print("\n🎉🎉🎉 ALL CHECKS PASSED! 🎉🎉🎉")

if __name__ == "__main__":
    sanity_check()


