"""HuBERT encoder wrapper.

API-identical to :class:`WavLMEncoder`. Useful as an ablation backbone since
HuBERT shares the same fairseq lineage but lacks WavLM's speaker-aware mixing.
"""

from __future__ import annotations

import torch.nn.functional as F
from transformers import AutoModel

from retro_profile.models.encoders.wavlm import WavLMEncoder


class HuBERTEncoder(WavLMEncoder):
    """HuBERT encoder. Drop-in replacement for WavLM."""

    def __init__(
        self,
        pretrained_name: str = "facebook/hubert-large-ll60k",
        freeze: bool = True,
        layer_aggregation: str = "weighted",
        output_dim: int | None = None,
    ) -> None:
        # Bypass WavLMEncoder's __init__ to load a HuBERT model instead, but reuse
        # all its forward / aggregation logic.
        super(WavLMEncoder, self).__init__(output_dim=output_dim or 1024)
        self.pretrained_name = pretrained_name
        self.freeze = freeze
        self.layer_aggregation = layer_aggregation
        self.model = AutoModel.from_pretrained(pretrained_name, output_hidden_states=True)
        hidden_size = int(getattr(self.model.config, "hidden_size", self.output_dim))
        self.output_dim = hidden_size
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        num_layers = int(self.model.config.num_hidden_layers) + 1
        if layer_aggregation == "weighted":
            import torch.nn as nn
            import torch
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        else:
            self.register_parameter("layer_weights", None)
