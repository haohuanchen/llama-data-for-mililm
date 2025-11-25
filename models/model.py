import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import math
import numpy as np
from mteb.models import ModelMeta


tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=384, num_heads=12, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*hidden)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = attn_probs @ v  # (B, num_heads, T, head_dim)
        
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(context)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=384, num_heads=6, ffn_dim=1536, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        
    def forward(self, x, mask=None):
        # Self-Attention + Residual
        x = x + self.attn(self.norm1(x), mask)
        # FFN + Residual
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLMSentenceTransformer(nn.Module):
    def __init__(self, vocab_size=250002, hidden_size=384, num_layers=6, num_heads=6, ffn_dim=1536, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.mteb_model_meta = ModelMeta(
            loader=lambda: None,
            name="haohuan/MiniLM-L6-Custom",
            revision="main",
            release_date="2025-10-01",
            languages=["eng-Latn", "zho-Hans"],
            n_parameters=0,
            memory_usage_mb=0,
            max_tokens=512,
            embed_dim=384,
            license="apache-2.0",
            open_weights=None,
            public_training_code="N/A",
            public_training_data="N/A",
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=set(),
        )
        
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.size()
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # mean pooling
        mask = attention_mask.unsqueeze(-1)
        sentence_emb = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return sentence_emb
    
    def encode(self, inputs, batch_size=16, device='cpu', **kwargs):
        sentences = [text for batch in inputs for text in batch["text"]]
        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                encoded = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                emb = self.forward(input_ids, attention_mask)
                all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)

    def set_revision(self, revision):
        self.mteb_model_meta.revision = revision

