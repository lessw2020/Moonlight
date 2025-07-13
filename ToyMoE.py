import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import threading


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


class AttentionLogitMonitor:
    """Thread-safe singleton for monitoring attention logits across the model"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.max_logit = 0.0
                    cls._instance.step_count = 0
        return cls._instance
    
    def update_max_logit(self, logit_value):
        """Update the maximum logit value seen in this step"""
        with self._lock:
            self.max_logit = max(self.max_logit, logit_value)
    
    def get_and_reset_max_logit(self):
        """Get the maximum logit and reset for next step"""
        with self._lock:
            current_max = self.max_logit
            self.max_logit = 0.0
            self.step_count += 1
            return current_max


class MonitoredMultiHeadAttention(nn.Module):
    """Multi-head attention with logit monitoring for MuonClip"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
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
        self.logit_monitor = AttentionLogitMonitor()
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Monitor maximum attention logit
        if self.training:
            max_logit = torch.max(torch.abs(scores)).item()
            self.logit_monitor.update_max_logit(max_logit)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class MoELayer(nn.Module):
    """Simple Mixture of Experts layer"""
    
    def __init__(self, d_model, num_experts=4, expert_capacity=None, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity or (d_model * 4) // num_experts
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.expert_capacity, bias=False),
                nn.ReLU(),
                nn.Linear(self.expert_capacity, d_model, bias=False)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # Compute gating scores
        gate_scores = self.gate(x_flat)  # (batch_size * seq_len, num_experts)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Route to experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_scores = top_k_scores[:, i].unsqueeze(-1)
            
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_scores[mask] * expert_output
        
        return output.view(batch_size, seq_len, d_model)


class TransformerMoEBlock(nn.Module):
    """Transformer block with MoE and monitored attention"""
    
    def __init__(self, d_model, num_heads, num_experts=4, dropout=0.1):
        super().__init__()
        self.attention = MonitoredMultiHeadAttention(d_model, num_heads, dropout)
        self.moe = MoELayer(d_model, num_experts)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # MoE with residual connection
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        
        return x


class TinyMoETransformer(nn.Module):
    """Small MoE transformer for demonstration"""
    
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, 
                 num_experts=4, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerMoEBlock(d_model, num_heads, num_experts)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.logit_monitor = AttentionLogitMonitor()
        
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return type('ModelOutput', (), {
            'loss': loss,
            'logits': logits,
            'max_attention_logit': self.logit_monitor.get_and_reset_max_logit()
        })()


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """Newton-Schulz iteration for orthogonalization"""
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


class ProductionMuonClip(torch.optim.Optimizer):
    """Production MuonClip with real attention logit monitoring"""
    
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
        self.max_attention_logit = 0.0
        
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
                
    def set_max_attention_logit(self, max_logit):
        """Set the maximum attention logit from forward pass"""
        self.max_attention_logit = max_logit
        
    def _apply_qk_clip(self):
        """Apply qk-clip using real attention logit values"""
        if not self.defaults["enable_qk_clip"] or len(self.qk_params["query"]) == 0:
            return
            
        threshold = self.defaults["qk_clip_threshold"]
        max_logit = self.max_attention_logit
        
        # Compute adaptive factor η
        eta = min(threshold / max_logit, 1.0) if max_logit > threshold else 1.0
        
        if eta < 1.0:  # Only apply clipping if needed
            alpha = self.defaults["qk_clip_alpha"]
            query_scale = eta ** alpha
            key_scale = eta ** (1 - alpha)
            
            for q_param in self.qk_params["query"]:
                q_param.data.mul_(query_scale)
                
            for k_param in self.qk_params["key"]:
                k_param.data.mul_(key_scale)
                
            logger.debug(f"Applied QK-clip: max_logit={max_logit:.3f}, eta={eta:.3f}, "
                        f"q_scale={query_scale:.3f}, k_scale={key_scale:.3f}")

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def step(self, closure=None):
        """Optimization step with production QK-clip"""
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
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

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

        # Apply QK-clip after parameter updates
        self._apply_qk_clip()
        
        return loss


def get_model_and_dataloader(dataset_name="openwebtext-100k", batch_size=8):
    """Create model and data loader"""
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    
    train_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer, max_length=256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = TinyMoETransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        num_experts=4,
        max_seq_len=256
    )
    
    return model, train_loader, tokenizer


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    """Create optimizer"""
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muonclip":
        # Separate parameters for Muon vs AdamW
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
        
        optimizer = ProductionMuonClip(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
            enable_qk_clip=True,
            qk_clip_threshold=10.0,
            qk_clip_alpha=0.5,
        )
        optimizer.register_model(model)
        return optimizer
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_model(model, optimizer, train_loader, device, num_epochs=1):
    """Training loop with attention logit monitoring"""
    model.to(device)
    model.train()
    
    total_steps = len(train_loader) * num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )
    
    step_count = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        max_logits = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            max_attention_logit = outputs.max_attention_logit
            
            # Pass attention logit info to optimizer
            if hasattr(optimizer, 'set_max_attention_logit'):
                optimizer.set_max_attention_logit(max_attention_logit)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            epoch_loss += loss.item()
            max_logits.append(max_attention_logit)
            step_count += 1
            
            if batch_idx % 10 == 0:
                avg_max_logit = sum(max_logits[-10:]) / len(max_logits[-10:])
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Avg Max Attention Logit: {avg_max_logit:.3f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_max_logit = sum(max_logits) / len(max_logits)
        logger.info(
            f"Epoch {epoch+1} completed - "
            f"Avg Loss: {avg_epoch_loss:.4f}, "
            f"Avg Max Attention Logit: {avg_max_logit:.3f}"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="muonclip", 
                       choices=["adamw", "muonclip"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    logger.add(f"logs/moe_train_{args.optimizer}_lr{args.lr}.log")
    logger.info(f"Starting training with {args.optimizer} optimizer")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_loader, tokenizer = get_model_and_dataloader(
        args.dataset, args.batch_size
    )
    optimizer = get_optimizer(args.optimizer, model, lr=args.lr, wd=args.wd)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    muon_params = sum(p.numel() for p in model.parameters() 
                     if p.ndim >= 2 and "embedding" not in str(p) and "lm_head" not in str(p))
    logger.info(f"Model created: {total_params:,} total parameters, "
               f"{muon_params:,} Muon parameters")
    
    # Train
    train_model(model, optimizer, train_loader, device, args.epochs)
    logger.info("Training completed!")
