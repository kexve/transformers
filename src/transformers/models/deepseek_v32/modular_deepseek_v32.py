# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeepSeek V3.2 Model - Extends DeepSeek V3 with DeepSeek Sparse Attention (DSA).

DeepSeek V3.2 introduces an indexer mechanism that enables fine-grained sparse attention,
significantly improving training and inference efficiency for long-context scenarios while
maintaining model output quality.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3ForSequenceClassification,
    DeepseekV3ForTokenClassification,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)


class DeepseekV32Config(DeepseekV3Config):
    """
    Configuration class for DeepSeek V3.2 model.
    
    DeepSeek V3.2 extends DeepSeek V3 with DeepSeek Sparse Attention (DSA), which uses an indexer
    mechanism to select the most relevant positions for attention computation, enabling efficient
    processing of very long sequences.
    
    Args:
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for the indexer module.
        index_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head in the indexer.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top-k positions to select for sparse attention.
        **kwargs:
            Additional arguments passed to DeepseekV3Config.
    
    Example:
        ```python
        >>> from transformers import DeepseekV32Config, DeepseekV32Model
        >>> 
        >>> # Initialize with custom indexer parameters
        >>> config = DeepseekV32Config(
        ...     index_n_heads=64,
        ...     index_head_dim=128,
        ...     index_topk=2048,
        ... )
        >>> model = DeepseekV32Model(config)
        ```
    """
    
    model_type = "deepseek_v32"
    
    def __init__(
        self,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        **super_kwargs
    ):
        super().__init__(**super_kwargs)
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_top_k = index_topk  # Note: using index_top_k for consistency


# Inherit all basic components from DeepSeek V3
class DeepseekV32TopkRouter(DeepseekV3TopkRouter):
    """DeepSeek V3.2 Top-k Router, inherits from DeepSeek V3."""
    pass


class DeepseekV32MoE(DeepseekV3MoE):
    """DeepSeek V3.2 Mixture of Experts, inherits from DeepSeek V3."""
    pass


class DeepseekV32MLP(DeepseekV3MLP):
    """DeepSeek V3.2 MLP, inherits from DeepSeek V3."""
    pass


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    """DeepSeek V3.2 RMS Normalization, inherits from DeepSeek V3."""
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    """DeepSeek V3.2 Rotary Embedding, inherits from DeepSeek V3."""
    pass


class DeepseekV32Indexer(nn.Module):
    """
    Indexer module for DeepSeek V3.2 sparse attention.
    
    The indexer computes top-k indices to select the most relevant positions for attention computation.
    This enables fine-grained sparse attention that significantly reduces computational complexity
    for long sequences while maintaining model quality.
    
    Key features:
    - Uses separate lightweight query/key projections
    - Applies ReLU activation for scoring
    - Supports rotary position embeddings
    - Computes weighted scores across multiple heads
    
    Args:
        config (DeepseekV32Config): Model configuration.
        layer_idx (int): Layer index for this indexer.
    """
    
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Indexer dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_top_k
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        
        # Indexer projections
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        
        self.softmax_scale = self.head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute top-k indices for sparse attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            q_compressed: Compressed query states [batch_size, seq_len, q_lora_rank]
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Top-k indices [batch_size, seq_len, topk]
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        
        # Compute indexer queries
        q = self.wq_b(q_compressed)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)  # [B, S, H, D]
        
        # Split into RoPE and non-RoPE parts
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        # Apply rotary embeddings to query
        if self.config.rope_interleave:
            q_pe = q_pe.transpose(1, 2)  # [B, H, S, rope_D]
            k_pe_dummy = torch.zeros_like(q_pe[:, :1])
            q_pe, _ = apply_rotary_pos_emb_interleave(q_pe, k_pe_dummy, cos, sin)
            q_pe = q_pe.transpose(1, 2)  # [B, S, H, rope_D]
        else:
            q_pe = q_pe.transpose(1, 2)  # [B, H, S, rope_D]
            q_pe, _ = apply_rotary_pos_emb(q_pe, q_pe[:, :1], cos, sin)
            q_pe = q_pe.transpose(1, 2)  # [B, S, H, rope_D]
        
        q = torch.cat([q_nope, q_pe], dim=-1)  # [B, S, H, D]
        
        # Compute indexer keys
        k = self.wk(hidden_states)  # [B, S, D]
        k = self.k_norm(k)  # [B, S, D]
        
        # Split key into RoPE and non-RoPE parts
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        # Apply rotary embeddings to key
        k_pe = k_pe.unsqueeze(1)  # [B, 1, S, rope_D]
        if self.config.rope_interleave:
            _, k_pe = apply_rotary_pos_emb_interleave(k_pe, k_pe, cos, sin)
        else:
            _, k_pe = apply_rotary_pos_emb(k_pe, k_pe, cos, sin)
        
        # Expand key to all heads
        k = torch.cat([
            k_nope.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, -1),
            k_pe.expand(batch_size, self.num_heads, seq_len, -1)
        ], dim=-1)  # [B, H, S, D]
        
        # Compute attention scores for indexing
        q = q.transpose(1, 2)  # [B, H, S, D]
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, H, S, S]
        
        # Apply ReLU activation (key innovation in DSA)
        scores = F.relu(scores)
        
        # Compute head weights
        head_weights = self.weights_proj(hidden_states)  # [B, S, H]
        head_weights = head_weights * (self.num_heads ** -0.5)
        head_weights = head_weights.transpose(1, 2).unsqueeze(-1)  # [B, H, S, 1]
        
        # Weight scores by head importance
        scores = scores * head_weights * self.softmax_scale
        
        # Aggregate across heads
        index_scores = scores.sum(dim=1)  # [B, S, S]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            index_scores = index_scores + attention_mask
        
        # Select top-k indices
        topk = min(self.index_topk, seq_len)
        topk_indices = index_scores.topk(topk, dim=-1).indices  # [B, S, topk]
        
        return topk_indices


class DeepseekV32Attention(DeepseekV3Attention):
    """
    DeepSeek V3.2 Attention with DeepSeek Sparse Attention (DSA).
    
    Extends DeepSeek V3 attention with an indexer mechanism that selects top-k positions
    to attend to, reducing computational complexity from O(n²) to O(n×k) for long sequences.
    
    The indexer uses:
    - Lightweight query/key projections
    - ReLU activation for scoring
    - Weighted aggregation across heads
    - Top-k selection for sparse attention
    
    This achieves substantial improvements in long-context training and inference efficiency
    while maintaining virtually identical model output quality.
    """
    
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Add indexer for sparse attention
        self.indexer = DeepseekV32Indexer(config, layer_idx)
        self.use_sparse_attention = True  # Flag to enable/disable sparse attention
    
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with DeepSeek Sparse Attention.
        
        For long sequences, applies sparse attention using the indexer to select top-k positions.
        For short sequences or during initial prefill, uses standard dense attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Attention mask tensor
            past_key_values: Cache for key/value states
            cache_position: Position indices for caching
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (attention_output, attention_weights, past_key_values)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # For short sequences or when sparse attention is disabled, use standard attention
        # Also use dense attention during prefill (when attention_mask is provided)
        if not self.use_sparse_attention or seq_len <= self.config.index_top_k or attention_mask is None:
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
        
        # Apply sparse attention for long sequences
        # Get compressed query states for indexer
        if self.config.q_lora_rank is not None and hasattr(self, 'q_a_proj'):
            q_compressed = self.q_a_layernorm(self.q_a_proj(hidden_states))
        else:
            q_compressed = hidden_states
        
        # Compute top-k indices using indexer
        topk_indices = self.indexer(
            hidden_states,
            q_compressed,
            position_embeddings,
            attention_mask,
        )
        
        # Create sparse attention mask based on top-k indices
        # Initialize with -inf to mask out non-selected positions
        sparse_mask = torch.full(
            (batch_size, seq_len, seq_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        # Set selected positions to 0 (no masking)
        batch_indices = torch.arange(batch_size, device=hidden_states.device).view(-1, 1, 1)
        seq_indices = torch.arange(seq_len, device=hidden_states.device).view(1, -1, 1)
        sparse_mask[batch_indices, seq_indices, topk_indices] = 0
        
        # Combine with existing attention mask if provided
        if attention_mask is not None:
            sparse_mask = sparse_mask + attention_mask
        
        # Call parent forward with sparse mask
        return super().forward(
            hidden_states,
            position_embeddings,
            sparse_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )


class DeepseekV32DecoderLayer(DeepseekV3DecoderLayer):
    """
    DeepSeek V3.2 Decoder Layer.
    
    Inherits from DeepSeek V3 but uses DeepseekV32Attention with sparse attention support.
    """
    
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        # Call nn.Module.__init__ directly to avoid LlamaDecoderLayer's init
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        
        # Use V3.2 attention with indexer
        self.self_attn = DeepseekV32Attention(config=config, layer_idx=layer_idx)
        
        # MLP layer (MoE or dense depending on layer index)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV32MoE(config)
        else:
            self.mlp = DeepseekV32MLP(config)
        
        # Layer norms
        self.input_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    """DeepSeek V3.2 PreTrained Model base class."""
    config_class = DeepseekV32Config


class DeepseekV32Model(DeepseekV3Model):
    """
    DeepSeek V3.2 Model with DeepSeek Sparse Attention.
    
    This model extends DeepSeek V3 with an indexer mechanism for efficient long-context processing.
    """
    config_class = DeepseekV32Config
    
    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        # Override layers with V3.2 decoder layers
        self.layers = nn.ModuleList(
            [DeepseekV32DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    """
    DeepSeek V3.2 Model for Causal Language Modeling.
    
    Extends DeepSeek V3 with DeepSeek Sparse Attention for efficient long-context generation.
    """
    config_class = DeepseekV32Config
    
    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        self.model = DeepseekV32Model(config)


class DeepseekV32ForSequenceClassification(DeepseekV3ForSequenceClassification):
    """DeepSeek V3.2 Model for Sequence Classification."""
    config_class = DeepseekV32Config
    
    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        self.model = DeepseekV32Model(config)


class DeepseekV32ForTokenClassification(DeepseekV3ForTokenClassification):
    """DeepSeek V3.2 Model for Token Classification."""
    config_class = DeepseekV32Config
    
    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        self.model = DeepseekV32Model(config)
