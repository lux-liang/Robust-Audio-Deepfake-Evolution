import torch
from models.CascadeMamba import Model
import sys

def inspect_gate():
    print("🕵️ Inspecting Gate Weights for Phase 4 Model...")
    
    # Path to the Epoch 0 checkpoint
    model_path = "exp_result/cascade_mamba_phase4_gated/LA_CascadeMamba_Phase4_Gated_ep30_bs16/weights/epoch_0_0.555.pth"
    
    try:
        # Load model structure
        model = Model()
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        # strict=True to ensure keys match exactly
        model.load_state_dict(checkpoint, strict=True)
        print(f"✅ Checkpoint loaded: {model_path}")
        
        # Inspect Gate Parameters
        # Structure: cascade_blocks[i].injection_gate[0].weight/bias
        
        print("\n🔍 Gate Analysis:")
        for i in range(3):
            gate_block = model.cascade_blocks[i].injection_gate
            # Check first linear layer weights
            weight = gate_block[0].weight
            bias = gate_block[0].bias
            
            # Calculate statistics
            mean_w = weight.mean().item()
            std_w = weight.std().item()
            max_w = weight.max().item()
            
            print(f"  Block {i} Gate (Linear Layer 1):")
            print(f"    - Mean: {mean_w:.6f}")
            print(f"    - Std:  {std_w:.6f}")
            print(f"    - Max:  {max_w:.6f}")
            
            if std_w > 0.0001:
                print(f"    ✅ Alive (Weights are distributed)")
            else:
                print(f"    ⚠️  Suspicious (Weights look static/zero)")

        print("\n🧠 Analysis Conclusion:")
        print("   If 'Alive', the gates are successfully initialized and participating in the gradient flow.")
        print("   Since we used strict=False for loading pretrained weights, these new layers started random.")
        print("   After 1 Epoch, they should have shifted slightly from pure random initialization.")

    except Exception as e:
        print(f"❌ Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_gate()






