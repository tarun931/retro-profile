"""Whisper encoder-only wrapper.

We discard Whisper's decoder. The encoder ingests log-mel features and returns a
fixed-length (1500 frames @ 30 s) sequence per utterance. We mean-pool that to
match the BaseEncoder contract.

Note: Whisper's feature extractor expects inputs padded to 30 seconds. We pad
internally rather than truncating speaker bags to keep memory predictable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, WhisperFeatureExtractor

from retro_profile.models.encoders.base import BaseEncoder


class WhisperEncoder(BaseEncoder):
    """Whisper encoder (encoder-only). API matches WavLMEncoder."""

    def __init__(
        self,
        pretrained_name: str = "openai/whisper-large-v3",
        freeze: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__(output_dim=output_dim or 1280)
        self.pretrained_name = pretrained_name
        self.freeze = freeze

        full = AutoModel.from_pretrained(pretrained_name)
        # Keep only the encoder; drop the decoder to save memory.
        self.encoder = full.encoder if hasattr(full, "encoder") else full.get_encoder()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_name)
        self.output_dim = int(getattr(self.encoder.config, "d_model", self.output_dim))

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_mask: torch.Tensor,
        utt_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, K, T = waveforms.shape
        flat_wav = waveforms.reshape(B * K, T).cpu().numpy()  # FE wants numpy

        # Build log-mel features. The feature extractor pads to 30 s internally.
        feats = self.feature_extractor(
            list(flat_wav),
            sampling_rate=16_000,
            return_tensors="pt",
        )["input_features"].to(waveforms.device)

        if self.freeze:
            with torch.no_grad():
                hidden = self.encoder(feats).last_hidden_state  # (BK, T'=1500, D)
        else:
            hidden = self.encoder(feats).last_hidden_state

        # We don't have a frame-level mask for Whisper because of the 30s padding;
        # mean over time is fine — silence frames contribute near-zero activations.
        utt_emb = hidden.mean(dim=1).reshape(B, K, self.output_dim)
        utt_emb = utt_emb * utt_mask.unsqueeze(-1).to(utt_emb.dtype)
        return utt_emb
