"""
Hybrid 1D-CNN + BiLSTM architecture for cliff detection from elevation transects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            mask: [batch, seq_len] boolean mask (True = valid)

        Returns:
            [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and split into heads
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [batch, num_heads, seq_len, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: [batch, num_heads, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            # Expand mask for attention: [batch, 1, 1, seq_len]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [batch, num_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # Final linear projection
        output = self.out(attn_output)

        return output


class CNN_BiLSTM_CliffDetector(nn.Module):
    """
    Hybrid CNN + BiLSTM + Attention architecture for cliff detection.

    Architecture:
        Input [batch, seq_len, 12]
        → 1D-CNN Feature Extractor (3 conv layers)
        → BiLSTM Encoder (2 layers)
        → Multi-Head Attention
        → Multi-Task Output Heads:
            - Segmentation: [batch, seq_len, 3] (background, base, top)
            - Regression: [batch, seq_len, 2] (base_offset, top_offset)
            - Confidence: [batch] (has_cliff probability)
    """

    def __init__(
        self,
        input_dim: int = 12,
        cnn_channels: list = [64, 128, 64],
        cnn_kernel_sizes: list = [5, 5, 3],
        cnn_dropout: float = 0.2,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_heads: int = 8,
        attention_dim: int = 256,
        attention_dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.attention_dim = attention_dim

        # 1D-CNN Feature Extractor
        self.cnn_layers = nn.ModuleList()
        in_channels = input_dim

        for out_channels, kernel_size in zip(cnn_channels, cnn_kernel_sizes):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(cnn_dropout)
            ))
            in_channels = out_channels

        self.cnn_out_channels = cnn_channels[-1]

        # BiLSTM Encoder
        self.bilstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden_size * 2  # Bidirectional

        # Project to attention dimension
        self.lstm_proj = nn.Linear(lstm_output_dim, attention_dim)

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            dim=attention_dim,
            num_heads=attention_heads,
            dropout=attention_dropout
        )

        # Residual connection + LayerNorm
        self.attention_norm = nn.LayerNorm(attention_dim)

        # Multi-Task Output Heads
        # 1. Segmentation head (3 classes: background, base, top)
        self.seg_head = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(attention_dim // 2, 3)
        )

        # 2. Regression head (2 outputs: base_offset, top_offset)
        self.reg_head = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(attention_dim // 2, 2)
        )

        # 3. Confidence head (transect-level: has_cliff)
        self.conf_head = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(attention_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, 12] input features
            mask: [batch, seq_len] boolean mask (True = valid)
            lengths: [batch] original sequence lengths

        Returns:
            Dictionary with:
                'segmentation': [batch, seq_len, 3] class logits
                'regression': [batch, seq_len, 2] distance offsets
                'confidence': [batch, 1] has_cliff probability
                'attention_output': [batch, seq_len, attention_dim] intermediate features
        """
        batch_size, seq_len, _ = x.shape

        # 1D-CNN Feature Extraction
        # Conv1d expects [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)  # [batch, 12, seq_len]

        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        # x_cnn: [batch, cnn_out_channels, seq_len]

        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, cnn_out_channels]

        # BiLSTM Encoding
        if lengths is not None:
            # Pack padded sequence for efficiency
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_cnn,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            lstm_out, _ = self.bilstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True,
                total_length=seq_len
            )
        else:
            lstm_out, _ = self.bilstm(x_cnn)
        # lstm_out: [batch, seq_len, lstm_hidden_size * 2]

        # Project to attention dimension
        x_attn = self.lstm_proj(lstm_out)  # [batch, seq_len, attention_dim]

        # Multi-Head Attention with residual connection
        attn_out = self.attention(x_attn, mask=mask)
        x_attn = self.attention_norm(x_attn + attn_out)
        # x_attn: [batch, seq_len, attention_dim]

        # Multi-Task Outputs
        # 1. Segmentation (point-level)
        seg_logits = self.seg_head(x_attn)  # [batch, seq_len, 3]

        # 2. Regression (point-level)
        reg_out = self.reg_head(x_attn)  # [batch, seq_len, 2]

        # 3. Confidence (transect-level)
        # Pool across sequence using max pooling
        if mask is not None:
            # Mask out padding before pooling
            x_masked = x_attn.clone()
            x_masked[~mask] = float('-inf')
            pooled = x_masked.max(dim=1)[0]  # [batch, attention_dim]
        else:
            pooled = x_attn.max(dim=1)[0]

        conf_out = self.conf_head(pooled)  # [batch, 1]

        return {
            'segmentation': seg_logits,
            'regression': reg_out,
            'confidence': conf_out,
            'attention_output': x_attn
        }

    def predict_cliff_positions(
        self,
        x: torch.Tensor,
        distances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Predict cliff base and top positions from raw output.

        Args:
            x: [batch, seq_len, 12] input features
            distances: [batch, seq_len] NEAR_DIST values
            mask: [batch, seq_len] boolean mask
            lengths: [batch] sequence lengths
            confidence_threshold: Minimum confidence to return prediction

        Returns:
            Dictionary with:
                'base_distances': [batch] predicted base distances (or -1 if no cliff)
                'top_distances': [batch] predicted top distances (or -1 if no cliff)
                'base_confidences': [batch] confidence scores for base
                'top_confidences': [batch] confidence scores for top
                'has_cliff': [batch] boolean (whether cliff detected)
        """
        # Get model outputs
        outputs = self.forward(x, mask=mask, lengths=lengths)

        seg_logits = outputs['segmentation']  # [batch, seq_len, 3]
        conf = outputs['confidence'].squeeze(-1)  # [batch]

        # Apply softmax to get probabilities
        seg_probs = F.softmax(seg_logits, dim=-1)  # [batch, seq_len, 3]

        batch_size = seg_probs.size(0)
        base_distances = torch.zeros(batch_size, device=x.device)
        top_distances = torch.zeros(batch_size, device=x.device)
        base_confidences = torch.zeros(batch_size, device=x.device)
        top_confidences = torch.zeros(batch_size, device=x.device)
        has_cliff = conf > confidence_threshold

        for i in range(batch_size):
            if not has_cliff[i]:
                base_distances[i] = -1.0
                top_distances[i] = -1.0
                continue

            # Get valid region
            if mask is not None:
                valid_mask = mask[i]
            else:
                valid_mask = torch.ones(seg_probs.size(1), dtype=torch.bool, device=x.device)

            # Extract probabilities for base and top
            base_probs = seg_probs[i, :, 1]  # Base channel
            top_probs = seg_probs[i, :, 2]   # Top channel

            # Zero out padded regions
            base_probs = base_probs.clone()
            top_probs = top_probs.clone()
            base_probs[~valid_mask] = 0.0
            top_probs[~valid_mask] = 0.0

            # Find argmax
            base_idx = torch.argmax(base_probs)
            top_idx = torch.argmax(top_probs)

            base_conf = base_probs[base_idx]
            top_conf = top_probs[top_idx]

            # Check if top is landward of base
            if distances[i, top_idx] > distances[i, base_idx] and base_conf > 0.1 and top_conf > 0.1:
                base_distances[i] = distances[i, base_idx]
                top_distances[i] = distances[i, top_idx]
                base_confidences[i] = base_conf
                top_confidences[i] = top_conf
            else:
                # Invalid configuration
                base_distances[i] = -1.0
                top_distances[i] = -1.0
                has_cliff[i] = False

        return {
            'base_distances': base_distances,
            'top_distances': top_distances,
            'base_confidences': base_confidences,
            'top_confidences': top_confidences,
            'has_cliff': has_cliff
        }
