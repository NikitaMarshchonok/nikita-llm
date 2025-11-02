"""
GPT-style transformer model architecture
Configurable decoder-only transformer for language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTConfig:
    """Configuration for GPT model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        layers: int = 6,
        heads: int = 6,
        dim: int = 384,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.layers = layers
        self.heads = heads
        self.dim = dim
        self.dropout = dropout
        self.head_dim = dim // heads
        
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_length, config.max_length))
            .view(1, 1, config.max_length, config.max_length)
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.dim, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 4 * config.dim)
        self.fc2 = nn.Linear(4 * config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.ff = FeedForward(config)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """GPT-style transformer language model"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        
        # Validate and clamp sequence length to prevent position embedding index errors
        if T > self.config.max_length:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Input sequence length {T} exceeds max_length {self.config.max_length}. Truncating to {self.config.max_length}.")
            input_ids = input_ids[:, :self.config.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.config.max_length]
            if labels is not None:
                labels = labels[:, :self.config.max_length]
            T = self.config.max_length
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(token_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_length
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.max_length else input_ids[:, -self.config.max_length:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self(input_ids_cond)
                logits = outputs['logits']
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Validate token is in vocabulary range
            if next_token.item() >= self.config.vocab_size:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Generated token {next_token.item()} exceeds vocab size "
                    f"{self.config.vocab_size}. Clamping to valid range."
                )
                next_token = torch.clamp(next_token, 0, self.config.vocab_size - 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_gpt_model(config_dict: dict) -> GPTModel:
    """Create GPT model from config dictionary"""
    config = GPTConfig(
        vocab_size=config_dict.get('vocab_size', 32000),
        max_length=config_dict.get('max_length', 512),
        layers=config_dict.get('layers', 6),
        heads=config_dict.get('heads', 6),
        dim=config_dict.get('dim', 384),
        dropout=config_dict.get('dropout', 0.1),
    )
    model = GPTModel(config)
    print(f"Created GPT model with {model.count_parameters():,} parameters")
    return model


if __name__ == '__main__':
    # Test model creation
    config = GPTConfig(
        vocab_size=32000,
        max_length=512,
        layers=6,
        heads=6,
        dim=384,
        dropout=0.1,
    )
    model = GPTModel(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
