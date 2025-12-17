import torch
import torch.nn as nn
import sys
import os

# Mock mamba_ssm for CPU testing if CUDA not available
try:
    import mamba_ssm
except ImportError:
    print("⚠️ mamba_ssm not found, mocking for dimension check...")
    sys.modules["mamba_ssm"] = type(sys)("mamba_ssm")
    sys.modules["mamba_ssm.modules"] = type(sys)("mamba_ssm.modules")
    sys.modules["mamba_ssm.modules.mamba_simple"] = type(sys)("mamba_ssm.modules.mamba_simple")
    
    class MockMamba(nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.d_model = d_model
        def forward(self, x, **kwargs):
            return x
            
    sys.modules["mamba_ssm.modules.mamba_simple"].Mamba = MockMamba
    sys.modules["mamba_ssm.modules.mamba_simple"].Block = MockMamba

# Add project root to path
sys.path.append(os.getcwd())

from models.WavLMMamba import Model
from loss import OCSoftmax

def sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running Check on {device}...")
    
    # 1. Mock Args
    class Args:
        emb_size = 144
        num_encoders = 6
    args = Args()

    # 2. Instantiate Model
    print("🔄 Instantiating Model...")
    try:
        model = Model(args, device=device).to(device)
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return

    # 3. Check Freezing (Tactical Check #2)
    print("\n❄️ Checking Frozen Layers...")
    frozen_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if not param.requires_grad:
            frozen_params += 1
    
    print(f"   Total Params: {total_params}")
    print(f"   Frozen Params: {frozen_params}")
    if frozen_params > 0:
        print("✅ WavLM freezing seems active.")
    else:
        print("⚠️ WARNING: No parameters are frozen! Check WavLMMamba.py")

    # 4. Dummy Input
    batch_size = 2
    # WavLM usually takes ~64000 samples (4s)
    seq_len = 64600 
    dummy_input = torch.randn(batch_size, seq_len).to(device)
    dummy_label = torch.tensor([0, 1]).to(device) # 0=Bonafide, 1=Spoof
    
    print(f"\n📦 Input Shape: {dummy_input.shape}")

    # 5. Forward Pass
    print("🔄 Testing Forward Pass...")
    try:
        # WavLMMamba returns: features (for OCSoftmax), logits (for CE)
        features, logits = model(dummy_input)
        print(f"✅ Forward Pass Successful")
        print(f"   Features Shape: {features.shape} (Expected: [{batch_size}, {args.emb_size}])")
        print(f"   Logits Shape:   {logits.shape}   (Expected: [{batch_size}, 2])")
        
        # Check dimensions
        if features.shape[1] != args.emb_size:
            print(f"❌ Feature dimension mismatch! Got {features.shape[1]}, expected {args.emb_size}")
        if logits.shape[1] != 2:
            print(f"❌ Logits dimension mismatch! Got {logits.shape[1]}, expected 2")
            
    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")
        import traceback
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
        return

    # 7. Backward Pass
    print("🔄 Testing Backward Pass...")
    try:
        loss.backward()
        print("✅ Backward Pass Successful - Gradients computed")
    except Exception as e:
        print(f"❌ Backward Pass Failed: {e}")
        return

    print("\n🎉🎉🎉 ALL CHECKS PASSED! 🎉🎉🎉")
    print("Ready for GPU training (ensure 'mamba_ssm' is installed on GPU env).")

if __name__ == "__main__":
    sanity_check()


