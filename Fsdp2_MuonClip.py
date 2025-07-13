import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
import threading
from typing import Optional


class DistributedAttentionLogitMonitor:
    """Thread-safe distributed attention logit monitor for FSDP2"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.local_max_logit = 0.0
                    cls._instance.step_count = 0
        return cls._instance
    
    def update_max_logit(self, logit_value: float):
        """Update the local maximum logit value"""
        with self._lock:
            self.local_max_logit = max(self.local_max_logit, logit_value)
    
    def get_global_max_logit_and_reset(self) -> float:
        """Get global maximum logit across all ranks and reset"""
        with self._lock:
            local_max = self.local_max_logit
            
            # Synchronize across all ranks if distributed
            if dist.is_initialized():
                # Create tensor for all-reduce
                max_logit_tensor = torch.tensor(
                    local_max, 
                    dtype=torch.float32,
                    device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
                )
                
                # All-reduce with MAX operation to get global maximum
                dist.all_reduce(max_logit_tensor, op=dist.ReduceOp.MAX)
                global_max = max_logit_tensor.item()
            else:
                global_max = local_max
            
            # Reset for next step
            self.local_max_logit = 0.0
            self.step_count += 1
            
            return global_max


class FSDP2CompatibleMuonClip(torch.optim.Optimizer):
    """
    FSDP2-compatible MuonClip optimizer with distributed QK logit synchronization
    """
    
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        qk_clip_threshold=10.0,
        qk_clip_alpha=0.5,
        enable_qk_clip=True,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            qk_clip_threshold=qk_clip_threshold,
            qk_clip_alpha=qk_clip_alpha,
            enable_qk_clip=enable_qk_clip,
        )
        
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
            
        self.model = None
        self.qk_params = {"query": [], "key": []}
        self.logit_monitor = DistributedAttentionLogitMonitor()
        
    def register_model(self, model):
        """Register model and identify Q/K parameters"""
        self.model = model
        self._identify_qk_params()
        
    def _identify_qk_params(self):
        """Identify query and key projection parameters"""
        if self.model is None:
            return
            
        self.qk_params = {"query": [], "key": []}
        
        for name, param in self.model.named_parameters():
            if "q_proj" in name:
                self.qk_params["query"].append(param)
            elif "k_proj" in name:
                self.qk_params["key"].append(param)
                
    def update_max_attention_logit(self, max_logit: float):
        """Update the maximum attention logit (called from forward pass)"""
        self.logit_monitor.update_max_logit(max_logit)
        
    def _apply_qk_clip_distributed(self):
        """Apply QK-clip with distributed synchronization"""
        if not self.defaults["enable_qk_clip"] or len(self.qk_params["query"]) == 0:
            return
            
        # Get the GLOBAL maximum attention logit across all ranks
        global_max_logit = self.logit_monitor.get_global_max_logit_and_reset()
        
        threshold = self.defaults["qk_clip_threshold"]
        
        # Compute adaptive factor η using global max
        eta = min(threshold / global_max_logit, 1.0) if global_max_logit > threshold else 1.0
        
        if eta < 1.0:  # Only apply clipping if needed
            alpha = self.defaults["qk_clip_alpha"]
            query_scale = eta ** alpha
            key_scale = eta ** (1 - alpha)
            
            # Apply the SAME scaling across all ranks
            for q_param in self.qk_params["query"]:
                q_param.data.mul_(query_scale)
                
            for k_param in self.qk_params["key"]:
                k_param.data.mul_(key_scale)
                
            # Log only on rank 0 to avoid spam
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Applied QK-clip: global_max_logit={global_max_logit:.3f}, "
                      f"eta={eta:.3f}, q_scale={query_scale:.3f}, k_scale={key_scale:.3f}")

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * torch.sqrt(torch.tensor(max(A, B), dtype=torch.float32))
        return lr * adjusted_ratio

    @torch.compile
    def zeropower_via_newtonschulz5(self, G, steps):
        """Newton-Schulz orthogonalization"""
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        if G.size(0) > G.size(1):
            X = X.T
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def step(self, closure=None):
        """Optimization step with distributed QK-clip"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon updates
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = self.zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            # AdamW updates for non-Muon parameters  
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        # Apply distributed QK-clip AFTER all parameter updates
        self._apply_qk_clip_distributed()
        
        return loss


class DistributedMonitoredAttention(torch.nn.Module):
    """Attention layer that reports to distributed monitor"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.logit_monitor = DistributedAttentionLogitMonitor()
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Monitor maximum attention logit (local to this rank)
        if self.training:
            local_max_logit = torch.max(torch.abs(scores)).item()
            self.logit_monitor.update_max_logit(local_max_logit)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


def setup_fsdp2_model_with_muonclip(model, optimizer_class=FSDP2CompatibleMuonClip):
    """
    Setup FSDP2 model with MuonClip optimizer
    
    Example usage:
    ```python
    model = MyTransformerModel(...)
    model, optimizer = setup_fsdp2_model_with_muonclip(model)
    
    # Training loop
    for batch in dataloader:
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()  # Includes distributed QK-clip
        optimizer.zero_grad()
    ```
    """
    
    # Apply FSDP2 to the model
    model = fully_shard(model)
    
    # Create MuonClip optimizer
    muon_params = []
    adamw_params = []
    
    for name, p in model.named_parameters():
        if (p.ndim >= 2 and 
            "embedding" not in name and 
            "lm_head" not in name and 
            "pos_encoding" not in name):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    
    optimizer = optimizer_class(
        lr=1e-3,
        wd=0.1,
        muon_params=muon_params,
        adamw_params=adamw_params,
        enable_qk_clip=True,
        qk_clip_threshold=10.0,
        qk_clip_alpha=0.5,
    )
    
    optimizer.register_model(model)
    
    return model, optimizer


# Example training step with FSDP2 + MuonClip
def distributed_training_step(model, optimizer, batch, device):
    """Example training step showing the distributed QK logit flow"""
    
    batch = batch.to(device)
    
    # Forward pass - each rank computes local attention logits
    # The DistributedMonitoredAttention layers automatically update
    # the DistributedAttentionLogitMonitor with local max values
    outputs = model(input_ids=batch, labels=batch)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Optimizer step with distributed QK-clip
    # 1. Apply Muon/AdamW updates
    # 2. All-reduce the max attention logits across ranks  
    # 3. Apply consistent QK-clip scaling to all ranks
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()


if __name__ == "__main__":
    print("FSDP2-compatible MuonClip implementation with distributed QK logit synchronization")
    print("\nKey features:")
    print("- Distributed attention logit monitoring")
    print("- All-reduce MAX operation for global logit sync")
    print("- Consistent QK-clip scaling across all ranks")
    print("- FSDP2 parameter sharding compatibility")
