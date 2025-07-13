import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import threading
from typing import Dict, List, Optional, Tuple
import contextlib
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MuonConfig:
    """Configuration for Muon optimizer with FSDP2"""
    lr: float = 1e-3
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    qk_clip_threshold: float = 10.0
    qk_clip_alpha: float = 0.5
    enable_qk_clip: bool = True
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8


class FSDP2GradientInterceptor:
    """
    Intercepts gradients before FSDP2 reduce-scatter for Muon processing
    """
    def __init__(self):
        self.intercepted_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self.muon_params: set = set()
        self.hook_handles = []
        self._lock = threading.Lock()
        
    def register_muon_param(self, param: torch.nn.Parameter):
        """Register a parameter that needs Muon processing"""
        with self._lock:
            self.muon_params.add(param)
            
    def register_hooks(self, model: nn.Module):
        """Register backward hooks to intercept gradients"""
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight in self.muon_params:
                handle = module.weight.register_hook(self._gradient_hook)
                self.hook_handles.append(handle)
                
    def _gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Hook function to capture gradients before FSDP processing"""
        # Find the parameter this gradient belongs to
        for param in self.muon_params:
            if grad.shape == param.shape:
                with self._lock:
                    # Store the full gradient before FSDP2 shards it
                    self.intercepted_grads[param] = grad.detach().clone()
                break
        return grad
        
    def get_full_gradient(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get the full gradient for a parameter"""
        with self._lock:
            return self.intercepted_grads.get(param, None)
            
    def clear_gradients(self):
        """Clear stored gradients after optimizer step"""
        with self._lock:
            self.intercepted_grads.clear()
            
    def cleanup(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


class DistributedAttentionMonitor:
    """Monitor attention logits across FSDP2 shards"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.local_max_logit = 0.0
        return cls._instance
    
    def update_max_logit(self, logit_value: float):
        """Update local maximum attention logit"""
        with self._lock:
            self.local_max_logit = max(self.local_max_logit, logit_value)
    
    def get_global_max_and_reset(self) -> float:
        """Get global maximum across all ranks and reset"""
        with self._lock:
            local_max = self.local_max_logit
            
            if dist.is_initialized():
                # All-reduce to get global maximum
                max_tensor = torch.tensor(
                    local_max, 
                    dtype=torch.float32,
                    device=torch.cuda.current_device()
                )
                dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
                global_max = max_tensor.item()
            else:
                global_max = local_max
                
            self.local_max_logit = 0.0
            return global_max


@torch.compile
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize matrix using Newton-Schulz iteration"""
    assert len(G.shape) == 2, f"Expected 2D tensor, got shape {G.shape}"
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    
    if G.size(0) > G.size(1):
        X = X.T
        
    # Normalize
    X = X / (X.norm() + 1e-7)
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
        
    return X.to(G.dtype)


class FSDP2MuonOptimizer(torch.optim.Optimizer):
    """
    Production FSDP2-compatible Muon optimizer with gradient interception
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MuonConfig,
        gradient_interceptor: FSDP2GradientInterceptor,
    ):
        self.config = config
        self.gradient_interceptor = gradient_interceptor
        self.attention_monitor = DistributedAttentionMonitor()
        
        # Separate parameters for Muon vs AdamW
        muon_params, adamw_params = self._separate_parameters(model)
        
        # Register Muon parameters with interceptor
        for param in muon_params:
            self.gradient_interceptor.register_muon_param(param)
            
        # Initialize optimizer
        param_groups = [
            {
                'params': muon_params,
                'use_muon': True,
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'momentum': config.momentum,
                'nesterov': config.nesterov,
                'ns_steps': config.ns_steps,
            },
            {
                'params': adamw_params,
                'use_muon': False,
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'betas': config.adamw_betas,
                'eps': config.adamw_eps,
            }
        ]
        
        super().__init__(muon_params + adamw_params, {})
        self.param_groups = param_groups
        
        # Store QK parameters for clipping
        self.qk_params = self._identify_qk_params(model)
        
    def _separate_parameters(self, model: nn.Module) -> Tuple[List, List]:
        """Separate parameters into Muon vs AdamW categories"""
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            if (param.ndim >= 2 and 
                'embed' not in name.lower() and 
                'lm_head' not in name.lower() and
                'norm' not in name.lower()):
                muon_params.append(param)
            else:
                adamw_params.append(param)
                
        return muon_params, adamw_params
    
    def _identify_qk_params(self, model: nn.Module) -> Dict[str, List]:
        """Identify query and key projection parameters"""
        qk_params = {'query': [], 'key': []}
        
        for name, param in model.named_parameters():
            if 'q_proj' in name or 'query' in name:
                qk_params['query'].append(param)
            elif 'k_proj' in name or 'key' in name:
                qk_params['key'].append(param)
                
        return qk_params
    
    def _gather_full_gradient(self, param: torch.nn.Parameter) -> torch.Tensor:
        """All-gather full gradient across FSDP2 shards"""
        local_grad = param.grad
        
        if local_grad is None:
            return None
            
        if not dist.is_initialized():
            return local_grad
            
        # Get the process group for data parallel communication
        world_size = dist.get_world_size()
        
        if world_size == 1:
            return local_grad
        
        # All-gather gradients from all ranks
        gathered_grads = [torch.zeros_like(local_grad) for _ in range(world_size)]
        dist.all_gather(gathered_grads, local_grad)
        
        # Concatenate or sum depending on FSDP sharding strategy
        # For gradient accumulation, we sum across ranks
        full_grad = torch.stack(gathered_grads).sum(dim=0)
        
        return full_grad
    
    def _apply_muon_update(self, param: torch.nn.Parameter, group: dict):
        """Apply Muon update to a parameter"""
        # Try to get full gradient from interceptor first
        full_grad = self.gradient_interceptor.get_full_gradient(param)
        
        # Fallback to gathering current gradient
        if full_grad is None:
            full_grad = self._gather_full_gradient(param)
            
        if full_grad is None:
            return
            
        # Flatten if needed for Newton-Schulz
        orig_shape = full_grad.shape
        if full_grad.ndim > 2:
            full_grad = full_grad.view(full_grad.size(0), -1)
            
        # Get or initialize momentum buffer
        state = self.state[param]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(full_grad)
            
        buf = state['momentum_buffer']
        momentum = group['momentum']
        
        # Apply momentum
        buf.mul_(momentum).add_(full_grad)
        
        if group['nesterov']:
            update_grad = full_grad.add(buf, alpha=momentum)
        else:
            update_grad = buf
            
        # Orthogonalize using Newton-Schulz
        orthogonal_update = newton_schulz_orthogonalize(
            update_grad, 
            steps=group['ns_steps']
        )
        
        # Scale update for consistent RMS
        A, B = param.shape[:2]
        scale_factor = 0.2 * torch.sqrt(torch.tensor(max(A, B), dtype=torch.float32))
        orthogonal_update = orthogonal_update * scale_factor
        
        # Reshape back if needed
        if orig_shape != orthogonal_update.shape:
            orthogonal_update = orthogonal_update.view(orig_shape)
            
        # Apply weight decay and update
        lr = group['lr']
        weight_decay = group['weight_decay']
        
        param.data.mul_(1 - lr * weight_decay)
        param.data.add_(orthogonal_update, alpha=-lr)
    
    def _apply_adamw_update(self, param: torch.nn.Parameter, group: dict):
        """Apply AdamW update to a parameter"""
        grad = param.grad
        if grad is None:
            return
            
        state = self.state[param]
        
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param.data)
            state['exp_avg_sq'] = torch.zeros_like(param.data)
            
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1
        
        # Exponential moving average of gradient values
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        # Update parameters
        step_size = group['lr'] / bias_correction1
        bias_correction2_sqrt = torch.sqrt(torch.tensor(bias_correction2))
        
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
        
        # Apply weight decay
        param.data.mul_(1 - group['lr'] * group['weight_decay'])
        param.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    def _apply_qk_clip(self):
        """Apply QK-clip using global attention logits"""
        if not self.config.enable_qk_clip or len(self.qk_params['query']) == 0:
            return
            
        # Get global maximum attention logit
        global_max_logit = self.attention_monitor.get_global_max_and_reset()
        
        if global_max_logit <= self.config.qk_clip_threshold:
            return
            
        # Compute scaling factors
        eta = min(self.config.qk_clip_threshold / global_max_logit, 1.0)
        alpha = self.config.qk_clip_alpha
        
        query_scale = eta ** alpha
        key_scale = eta ** (1 - alpha)
        
        # Apply scaling to all query and key parameters
        for q_param in self.qk_params['query']:
            q_param.data.mul_(query_scale)
            
        for k_param in self.qk_params['key']:
            k_param.data.mul_(key_scale)
            
        # Log only on rank 0
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Applied QK-clip: max_logit={global_max_logit:.3f}, "
                  f"eta={eta:.3f}, q_scale={query_scale:.3f}, k_scale={key_scale:.3f}")
    
    def step(self, closure=None):
        """Perform optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Process each parameter group
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                if group['use_muon']:
                    self._apply_muon_update(param, group)
                else:
                    self._apply_adamw_update(param, group)
        
        # Apply QK-clip after all parameter updates
        self._apply_qk_clip()
        
        # Clear intercepted gradients
        self.gradient_interceptor.clear_gradients()
        
        return loss


class FSDP2MonitoredAttention(nn.Module):
    """Attention layer that reports logits to the monitor"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_monitor = DistributedAttentionMonitor()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Monitor attention logits during training
        if self.training:
            max_logit = torch.max(torch.abs(scores)).item()
            self.attention_monitor.update_max_logit(max_logit)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


def setup_fsdp2_muon_training(
    model: nn.Module,
    config: MuonConfig,
    device_mesh = None
) -> Tuple[nn.Module, FSDP2MuonOptimizer, FSDP2GradientInterceptor]:
    """
    Set up model with FSDP2 and Muon optimizer
    
    Args:
        model: The model to wrap with FSDP2
        config: Muon configuration
        device_mesh: Optional device mesh for advanced FSDP2 setups
        
    Returns:
        Tuple of (fsdp_model, optimizer, gradient_interceptor)
    """
    
    # Initialize gradient interceptor
    gradient_interceptor = FSDP2GradientInterceptor()
    
    # Apply FSDP2 to model
    # Use policy to wrap each transformer block
    def wrap_policy(module, recurse, nonwrapped_numel):
        if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            return True
        elif hasattr(module, 'layers') and len(list(module.children())) > 10:
            return True
        return False
    
    # Apply FSDP2 sharding
    fsdp_model = fully_shard(
        model,
        policy=ModuleWrapPolicy([nn.TransformerEncoderLayer, nn.TransformerDecoderLayer]),
        # Key FSDP2 configurations for Muon
        reshard_after_forward=True,
        mixed_precision=None,  # Set according to your training precision
        device_mesh=device_mesh,
    )
    
    # Register gradient interception hooks
    gradient_interceptor.register_hooks(fsdp_model)
    
    # Create Muon optimizer
    optimizer = FSDP2MuonOptimizer(
        model=fsdp_model,
        config=config,
        gradient_interceptor=gradient_interceptor,
    )
    
    return fsdp_model, optimizer, gradient_interceptor


def example_training_loop():
    """Example training loop with FSDP2 + Muon"""
    
    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    
    # Create model (replace with your actual model)
    model = nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ).to(device)
    
    # Configure Muon
    config = MuonConfig(
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        enable_qk_clip=True,
        qk_clip_threshold=10.0,
    )
    
    # Set up FSDP2 + Muon
    fsdp_model, optimizer, gradient_interceptor = setup_fsdp2_muon_training(
        model, config
    )
    
    # Example training loop
    fsdp_model.train()
    
    for step in range(100):  # Example training steps
        # Forward pass
        src = torch.randn(10, 32, 512, device=device)  # (seq_len, batch, d_model)
        tgt = torch.randn(10, 32, 512, device=device)
        
        output = fsdp_model(src, tgt)
        loss = torch.nn.functional.mse_loss(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step with Muon + QK-clip
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # Cleanup
    gradient_interceptor.cleanup()
    print("Training completed!")


if __name__ == "__main__":
    example_training_loop()
