import torch
import torch.nn as nn
from typing import List
from torch.nn import functional as F

class RelativePositionBias(nn.Module):
    """
    Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are
    invalid.

    We use smaller buckets for small absolute relative_position and larger buckets
    for larger absolute relative_positions. All relative positions >=max_distance
    map to the same bucket. All relative positions <=-max_distance map to the
    same bucket. This should allow for more graceful generalization to longer
    sequences than the model has been trained on.

    Args:
        bidirectional (bool): Whether the attention is bidirectional.
        num_buckets (int): Number of buckets.
        max_distance (int): Maximum distance for relative positions.
        num_heads (int): Number of attention heads.

    # REFRANCE: https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, num_heads=8):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Translate relative position to a bucket number.

        Args:
            relative_position (torch.Tensor): Relative position tensor.
            bidirectional (bool): Whether the attention is bidirectional.
            num_buckets (int): Number of buckets.
            max_distance (int): Maximum distance for relative positions.

        Returns:
            torch.Tensor: Bucket number tensor.
        """
        ret = torch.ones_like(relative_position, dtype=torch.long)

        # Handle bidirectional
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position < 0).to(torch.long) * num_buckets
            relative_position = relative_position.abs()

        max_exact = num_buckets // 2
        is_small = (relative_position < max_exact)

        # Compute val_if_large
        val_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            torch.log(torch.tensor(max_distance / max_exact).float()) * (num_buckets - max_exact)
        ).to(torch.long)

        # Clamp val_if_large
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)

        # Update ret based on is_small
        ret += torch.where(is_small, relative_position, val_if_large)

        return ret

    def compute_bias(self, qlen, klen):
        """
        Compute binned relative position bias.

        Args:
            qlen (int): Length of the query sequence.
            klen (int): Length of the key sequence.

        Returns:
            torch.Tensor: Relative position bias tensor.
        """
        # Create context and memory positions
        context_position = torch.arange(0, qlen, dtype=torch.long, device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(0, klen, dtype=torch.long, device=self.relative_attention_bias.weight.device)[None, :]

        # Compute relative position
        relative_position = memory_position - context_position

        # Compute relative position bucket
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)

        # Get values from the embedding
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        return values

    def forward(self, qlen, klen):
        """
        Forward pass.

        Args:
            qlen (int): Length of the query sequence.
            klen (int): Length of the key sequence.

        Returns:
            torch.Tensor: Relative position bias tensor.
        """
        return self.compute_bias(qlen, klen)


class AttentionHead(nn.Module):
    """
    Relation-aware attention head implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, hidden_size, head_dim):
        """
        Initializes the AttentionHead.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 relative_biases:torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query)
        key: torch.Tensor = self.key_weights(key)
        value: torch.Tensor = self.value_weights(value)

        att_scores: torch.Tensor = (torch.matmul(query, key.transpose(1, 2)) + relative_biases) / self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)
        n_value: torch.Tensor = torch.matmul(att_weights, value)

        return n_value


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        attention_heads (nn.ModuleList): List of AttentionHead layers.
        fc (nn.Linear): Fully connected layer for final projection.
    """

    def __init__(self, hidden_size, num_heads):
        """
        Initializes the MultiHeadAttention layer.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.attention_heads: nn.ModuleList = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim) for head_num in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, relative_position_bias: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask, relative_biases=relative_position_bias[:,i]) for i, attention_head in enumerate(self.attention_heads)]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state
