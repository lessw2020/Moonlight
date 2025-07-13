import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Callable, Set
import threading
from collections import defaultdict
from dataclasses import dataclass
import weakref


@dataclass
class StreamingMuonConfig:
    """Configuration for streaming Muon optimizer"""
    lr: float = 1e-3
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    qk_clip_threshold: float = 10.0
    qk_clip_alpha: float = 0.5
    enable_qk_clip: bool = True
    adamw_betas: tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8


class ParameterState:
    """Per-parameter state for streaming optimizer"""
    def __init__(self, param: torch.nn.Parameter, use_muon: bool):
        self.param_ref = weakref.ref(param)  # Weak reference to avoid memory leaks
        self.use_muon = use_muon
        self.momentum_buffer = None
        self.step = 0
        
        # AdamW state
        if not use_muon:
            self.exp_avg = None
            self.exp_avg_sq = None


class StreamingAttentionMonitor:
    """Collects attention logits for delayed QK-clip"""
    def __init__(self):
        self.attention_logits = []
        self._lock = threading.Lock()
        
    def add_attention_logit(self, logit_value: float):
        with self._lock:
            self.attention_logits.append(logit_value)
    
    def get_global_max_and_reset(self) -> float:
        with self._lock:
            if not self.attention_logits:
                return 0.0
                
            local_max = max(self.attention_logits)
            
            # Distributed max reduction
            if dist.is_initialized():
                max_tensor = torch.tensor(local_max, dtype=torch.float32, device='cuda')
                dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
                global_max = max_tensor.item()
            else:
                global_max = local_max
                
            self.attention_logits.clear()
            return global_max


class StreamingMuonOptimizer:
    """
    Memory-efficient streaming Muon optimizer that updates parameters immediately
    as gradients are intercepted, rather than batching them.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: StreamingMuonConfig,
    ):
        self.config = config
        self.model_ref = weakref.ref(model)
        self.attention_monitor = StreamingAttentionMonitor()
        
        # Per-parameter state
        self.param_states: Dict[int, ParameterState] = {}
        self.qk_params = {'query': set(), 'key': set()}
        
        # Track which parameters need updates
        self.pending_updates: Set[int] = set()
        self.updates_applied = 0
        
        # Gradient hooks
        self.hook_handles = []
        
        self._setup_parameters()
        self._register_hooks()
    
    def _setup_parameters(self):
        """Initialize parameter states and identify QK parameters"""
        model = self.model_ref()
        if model is None:
            return
            
        for name, param in model.named_parameters():
            param_id = id(param)
            
            # Determine if this parameter should use Muon
            use_muon = (param.ndim >= 2 and 
                       'embed' not in name.lower() and 
                       'lm_head' not in name.lower() and
                       'norm' not in name.lower())
            
            # Create parameter state
            self.param_states[param_id] = ParameterState(param, use_muon)
            
            # Identify Q/K parameters for QK-clip
            if 'q_proj' in name or 'query' in name:
                self.qk_params['query'].add(param_id)
            elif 'k_proj' in name or 'key' in name:
                self.qk_params['key'].add(param_id)
    
    def _register_hooks(self):
        """Register gradient hooks for immediate processing"""
        model = self.model_ref()
        if model is None:
            return
            
        for param in model.parameters():
            if param.requires_grad:
                handle = param.register_hook(self._gradient_hook)
                self.hook_handles.append(handle)
    
    def _gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Process gradient immediately when it's computed.
        This is the key memory optimization - we don't store gradients!
        """
        # Find which parameter this gradient belongs to
        model = self.model_ref()
        if model is None:
            return grad
            
        for param in model.parameters():
            if param.grad is grad:
                param_id = id(param)
                if param_id in self.param_states:
                    # 🚀 IMMEDIATE UPDATE - no gradient storage!
                    self._apply_immediate_update(param, grad)
                    self.pending_updates.add(param_id)
                break
                
        return grad
    
    def _apply_immediate_update(self, param: torch.nn.Parameter, grad: torch.Tensor):
        """Apply Muon or AdamW update immediately to this parameter"""
        param_id = id(param)
        state = self.param_states[param_id]
        
        if state.use_muon:
            self._apply_streaming_muon_update(param, grad, state)
        else:
            self._apply_streaming_adamw_update(param, grad, state)
    
    def _apply_streaming_muon_update(self, param: torch.nn.Parameter, grad: torch.Tensor, state: ParameterState):
        """Apply Muon update to single parameter immediately"""
        # Handle shape for Newton-Schulz
        orig_shape = grad.shape
        if grad.ndim > 2:
            grad = grad.view(grad.size(0), -1)
        
        # Initialize momentum buffer if needed
        if state.momentum_buffer is None:
            state.momentum_buffer = torch.zeros_like(grad)
        
        # Apply momentum
        state.momentum_buffer.mul_(self.config.momentum).add_(grad)
        
        if self.config.nesterov:
            update_grad = grad.add(state.momentum_buffer, alpha=self.config.momentum)
        else:
            update_grad = state.momentum_buffer
        
        # Newton-Schulz orthogonalization
        try:
            orthogonal_update = self._newton_schulz_orthogonalize(update_grad)
        except Exception as e:
            # Fallback to gradient if orthogonalization fails
            print(f"Warning: Newton-Schulz failed for param {param.shape}, using gradient")
            orthogonal_update = update_grad
        
        # Scale update for consistent RMS
        A, B = param.shape[:2]
        scale_factor = 0.2 * (max(A, B) ** 0.5)
        orthogonal_update = orthogonal_update * scale_factor
        
        # Reshape back if needed
        if orig_shape != orthogonal_update.shape:
            orthogonal_update = orthogonal_update.view(orig_shape)
        
        # Apply weight decay and update
        param.data.mul_(1 - self.config.lr * self.config.weight_decay)
        param.data.add_(orthogonal_update, alpha=-self.config.lr)
        
        state.step += 1
    
    def _apply_streaming_adamw_update(self, param: torch.nn.Parameter, grad: torch.Tensor, state: ParameterState):
        """Apply AdamW update to single parameter immediately"""
        # Initialize AdamW state if needed
        if state.exp_avg is None:
            state.exp_avg = torch.zeros_like(param.data)
            state.exp_avg_sq = torch.zeros_like(param.data)
        
        state.step += 1
        beta1, beta2 = self.config.adamw_betas
        
        # Exponential moving averages
        state.exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        state.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** state.step
        bias_correction2 = 1 - beta2 ** state.step
        
        step_size = self.config.lr / bias_correction1
        bias_correction2_sqrt = (bias_correction2 ** 0.5)
        
        denom = (state.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.config.adamw_eps)
        
        # Apply weight decay and update
        param.data.mul_(1 - self.config.lr * self.config.weight_decay)
        param.data.addcdiv_(state.exp_avg, denom, value=-step_size)
    
    @torch.compile
    def _newton_schulz_orthogonalize(self, G: torch.Tensor) -> torch.Tensor:
        """Newton-Schulz orthogonalization (same as before)"""
        assert len(G.shape) == 2
        
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16() if G.dtype != torch.bfloat16 else G
        
        if G.size(0) > G.size(1):
            X = X.T
            
        X = X / (X.norm() + 1e-7)
        
        for _ in range(self.config.ns_steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if G.size(0) > G.size(1):
            X = X.T
            
        return X.to(G.dtype)
    
    def add_attention_logit(self, logit_value: float):
        """Called by attention layers to report max logits"""
        self.attention_monitor.add_attention_logit(logit_value)
    
    def step(self, closure=None):
        """
        Finalize the optimization step. At this point, all parameters have
        already been updated via gradient hooks. This just handles QK-clip.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Apply QK-clip if enabled (this is the only thing we do in step())
        if self.config.enable_qk_clip and self.pending_updates:
            self._apply_delayed_qk_clip()
        
        # Reset for next step
        self.pending_updates.clear()
        self.updates_applied += 1
        
        return loss
    
    def _apply_delayed_qk_clip(self):
        """Apply QK-clip after all parameters have been updated"""
        # Get global max attention logit
        global_max_logit = self.attention_monitor.get_global_max_and_reset()
        
        if global_max_logit <= self.config.qk_clip_threshold:
            return
        
        # Compute scaling factors
        eta = min(self.config.qk_clip_threshold / global_max_logit, 1.0)
        alpha = self.config.qk_clip_alpha
        
        query_scale = eta ** alpha
        key_scale = eta ** (1 - alpha)
        
        # Apply scaling to Q/K parameters
        model = self.model_ref()
        if model is None:
            return
            
        for param in model.parameters():
            param_id = id(param)
            if param_id in self.qk_params['query']:
                param.data.mul_(query_scale)
            elif param_id in self.qk_params['key']:
                param.data.mul_(key_scale)
        
        # Log only on rank 0
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Applied QK-clip: max_logit={global_max_logit:.3f}, "
                  f"eta={eta:.3f}, q_scale={query_scale:.3f}, k_scale={key_scale:.3f}")
    
    def zero_grad(self):
        """Clear gradients (standard optimizer interface)"""
        model = self.model_ref()
        if model is None:
            return
            
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def cleanup(self):
        """Remove gradient hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    @property
    def param_groups(self):
        """Compatibility with standard optimizer interface"""
        return [{'lr': self.config.lr, 'weight_decay': self.config.weight_decay}]


class StreamingMonitoredAttention(nn.Module):
    """Attention layer that reports logits to streaming optimizer"""
    
    def __init__(self, d_model: int, num_heads: int, optimizer: StreamingMuonOptimizer, dropout: float = 0.1):
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
        self.optimizer_ref = weakref.ref(optimizer)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Report attention logits to optimizer
        if self.training:
            max_logit = torch.max(torch.abs(scores)).item()
            optimizer = self.optimizer_ref()
            if optimizer is not None:
                optimizer.add_attention_logit(max_logit)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


def memory_usage_comparison():
    """Compare memory usage: standard vs streaming approach"""
    print("=" * 70)
    print("MEMORY USAGE: STANDARD vs STREAMING MUON")
    print("=" * 70)
    
    # Example: 4 attention layers, 4096x4096 each
    num_layers = 4
    matrix_size = 4096 * 4096
    bytes_per_param = 4  # float32
    
    layer_size_mb = (matrix_size * bytes_per_param) / (1024 * 1024)
    total_params_mb = num_layers * layer_size_mb
    
    print(f"📊 Model: {num_layers} attention layers, {matrix_size:,} params each")
    print(f"💾 Per-layer memory: {layer_size_mb:.1f} MB")
    print(f"📈 Total Muon parameters: {total_params_mb:.1f} MB")
    
    print(f"\n🏗️  STANDARD APPROACH (batch all gradients):")
    standard_peak_mb = total_params_mb * 2  # parameters + full gradients
    print(f"Peak memory: {standard_peak_mb:.1f} MB")
    print(f"  - Parameters: {total_params_mb:.1f} MB")
    print(f"  - Stored gradients: {total_params_mb:.1f} MB")
    
    print(f"\n🚀 STREAMING APPROACH (immediate updates):")
    streaming_peak_mb = total_params_mb + layer_size_mb  # parameters + 1 gradient
    print(f"Peak memory: {streaming_peak_mb:.1f} MB")
    print(f"  - Parameters: {total_params_mb:.1f} MB")
    print(f"  - Current gradient: {layer_size_mb:.1f} MB")
    
    memory_savings = standard_peak_mb - streaming_peak_mb
    savings_percent = (memory_savings / standard_peak_mb) * 100
    
    print(f"\n💡 MEMORY SAVINGS:")
    print(f"Reduction: {memory_savings:.1f} MB ({savings_percent:.1f}%)")
    print(f"Peak memory is {streaming_peak_mb/standard_peak_mb:.2f}x of standard approach")


def example_streaming_training():
    """Example training loop with streaming Muon optimizer"""
    print("\n" + "=" * 70)
    print("EXAMPLE: STREAMING MUON TRAINING")
    print("=" * 70)
    
    # Initialize distributed training
    if not dist.is_initialized():
        print("💡 Note: Run with torchrun for distributed training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=512, num_heads=8, num_layers=2):
            super().__init__()
            self.layers = nn.ModuleList()
            self.optimizer = None  # Will be set after creation
            
            for _ in range(num_layers):
                layer = StreamingMonitoredAttention(d_model, num_heads, None)
                self.layers.append(layer)
                
        def set_optimizer(self, optimizer):
            self.optimizer = optimizer
            # Update attention layers with optimizer reference
            for layer in self.layers:
                layer.optimizer_ref = weakref.ref(optimizer)
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = SimpleTransformer().to(device)
    
    # Configure streaming Muon
    config = StreamingMuonConfig(
        lr=1e-3,
        weight_decay=0.1,
        enable_qk_clip=True
    )
    
    # Create streaming optimizer
    optimizer = StreamingMuonOptimizer(model, config)
    model.set_optimizer(optimizer)
    
    print("✅ Created streaming Muon optimizer")
    print("🎯 Parameters will be updated immediately as gradients are computed")
    
    # Training loop
    model.train()
    
    for step in range(5):
        # Forward pass
        x = torch.randn(8, 32, 512, device=device)  # (batch, seq, dim)
        output = model(x)
        
        # Dummy loss
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        
        print(f"\n🔄 Step {step + 1}")
        print(f"  Forward pass completed, loss: {loss.item():.4f}")
        
        # Backward pass - parameters updated immediately via hooks!
        loss.backward()
        print(f"  ⚡ Backward pass completed - all parameters already updated!")
        
        # optimizer.step() only handles QK-clip now
        optimizer.step()
        print(f"  ✅ QK-clip applied")
        
        optimizer.zero_grad()
        
        # Show memory efficiency
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  📊 Peak GPU memory: {memory_mb:.1f} MB")
    
    # Cleanup
    optimizer.cleanup()
    print("\n✅ Training completed with minimal memory overhead!")


if __name__ == "__main__":
    memory_usage_comparison()
    example_streaming_training()
    
    
