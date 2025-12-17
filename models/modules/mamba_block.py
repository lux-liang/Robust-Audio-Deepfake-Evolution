import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        d_model: Input dimension
        d_state: SSM state dimension (N in paper)
        d_conv: Conv kernel size
        expand: Expansion factor (default 2)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Dim)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Input Projection
        x_and_res = self.in_proj(x)  # (B, L, 2*D_inner)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # 2. Convolution (Short 1D Conv)
        x_in = x_in.transpose(1, 2) # (B, D_inner, L)
        x_in = self.conv1d(x_in)[:, :, :seq_len]
        x_in = x_in.transpose(1, 2) # (B, L, D_inner)
        x_in = F.silu(x_in)

        # 3. SSM (Selective Scan)
        y = self.ssm_step(x_in)
        
        # 4. Gating & Output
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm_step(self, u):
        """
        Pure PyTorch implementation of the Selective Scan.
        Slow but functional for CPU/Validation.
        """
        batch_size, seq_len, d_inner = u.shape
        
        # Project x to parameters
        x_dbl = self.x_proj(u) # (B, L, dt_rank + 2*d_state)
        
        (dt, B, C) = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt: (B, L, dt_rank)
        # B: (B, L, d_state)
        # C: (B, L, d_state)

        dt = F.softplus(self.dt_proj(dt)) # (B, L, d_inner)
        
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Discretize
        # A_bar = exp(delta * A)
        # B_bar = delta * B
        
        # Recursive scan (simplified for readability, very slow in python loop)
        # For efficiency in pure torch, we use parallel prefix scan if possible, 
        # but here we use a loop for correctness proof.
        
        h = torch.zeros(batch_size, d_inner, self.d_state, device=u.device)
        ys = []
        
        # Note: optimizing this loop is key, but for "running logic" this is fine.
        for i in range(seq_len):
            dt_i = dt[:, i, :].unsqueeze(2) # (B, d_inner, 1)
            B_i = B[:, i, :].unsqueeze(1)   # (B, 1, d_state)
            C_i = C[:, i, :].unsqueeze(1)   # (B, 1, d_state)
            u_i = u[:, i, :].unsqueeze(2)   # (B, d_inner, 1)
            
            # Discretize A -> A_bar
            # A is (d_inner, d_state), broadcast to (B, d_inner, d_state)
            A_bar = torch.exp(A * dt_i) # Element-wise
            
            # Discretize B -> B_bar
            B_bar = B_i * dt_i # (B, d_inner, d_state)
            
            # Update state
            # h_t = A_bar * h_{t-1} + B_bar * u_t
            h = A_bar * h + B_bar * u_i
            
            # Output y_t = C_t * h_t
            y_i = torch.sum(h * C_i, dim=-1) # (B, d_inner)
            ys.append(y_i)
            
        y = torch.stack(ys, dim=1) # (B, L, d_inner)
        
        # Add residual D * u
        y = y + u * self.D
        
        return y
