"""WavLM encoder wrapper.

Defaults to ``microsoft/wavlm-large`` (24 transformer layers, 1024-d). We optionally
freeze the base and learn a *layer-weighted* aggregation across all hidden states:
the same trick that ECAPA-TDNN-WavLM uses for speaker verification — different
layers carry different speaker information, and a simple weighted sum is a
strong, tiny extra parameter.

Each clip is then reduced to one vector over time: a masked mean by default, or
learned frame-level attention (``frame_pooling: attention``), which together
with the clip-level MIL pooling makes the attention hierarchical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.pooling.frame_attention import FrameAttentionPooling

FRAME_POOLING = ("mean", "attention")


class WavLMEncoder(BaseEncoder):
    """WavLM (or any HF WavLM-compatible model) wrapper for trait regression."""

    def __init__(
        self,
        pretrained_name: str = "microsoft/wavlm-large",
        freeze: bool = True,
        layer_aggregation: str = "weighted",   # weighted | last | mean
        output_dim: int | None = None,         # auto-detect if None
        gradient_checkpointing: bool = False,
        frame_pooling: str = "mean",           # mean | attention (hierarchical)
        frame_attention_hidden: int = 128,
        frame_attention_dropout: float = 0.1,
    ) -> None:
        # Lazily detect dim from the model config if not provided.
        # We must defer dim detection until after model load.
        super().__init__(output_dim=output_dim or 1024)

        if frame_pooling not in FRAME_POOLING:
            raise ValueError(f"frame_pooling must be one of {FRAME_POOLING}, got {frame_pooling!r}")
        self.pretrained_name = pretrained_name
        self.freeze = freeze
        self.layer_aggregation = layer_aggregation
        self.frame_pooling = frame_pooling

        self.model = AutoModel.from_pretrained(pretrained_name, output_hidden_states=True)
        hidden_size = int(getattr(self.model.config, "hidden_size", self.output_dim))
        self.output_dim = hidden_size

        # Frame-level attention lives beside `layer_weights`, outside `.model`, so
        # it trains at the head LR (lightning_module._BACKBONE_PREFIX) and stays
        # trainable when the backbone is frozen.
        self.frame_pool = (
            FrameAttentionPooling(hidden_size, frame_attention_hidden, frame_attention_dropout)
            if frame_pooling == "attention" else None
        )
        # (weights, frame_mask), each (B*K, T'), from the latest EVAL-mode forward
        # with frame attention; read by the evaluator. Never kept while training.
        self.last_frame_attention: tuple[torch.Tensor, torch.Tensor] | None = None

        # Only models pretrained WITH an attention mask should be given one.
        # feat_extract_norm == "layer" (WavLM, wav2vec2-large) means yes;
        # "group" (wav2vec2-base) means the model never saw masks during
        # pretraining and passing one degrades its representations. This is the
        # same rule frame_encoders.py applies; without it, swapping the encoder
        # to wav2vec2-base silently feeds it an input it was not trained on.
        self.use_attention_mask = (
            getattr(self.model.config, "feat_extract_norm", "group") == "layer"
        )

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        elif gradient_checkpointing:
            # Fine-tuning in bag mode pushes batch_size * bag_size clips through
            # the encoder *with* gradients -- e.g. 16 speakers x 8 utterances =
            # 128 clips per step, against 1 clip per sample in per-utterance
            # training. Storing activations for all of them OOMs any GPU, so we
            # recompute them in the backward pass instead. Costs ~30% more time.
            self.model.gradient_checkpointing_enable()

        # Learnable layer weights for "weighted" aggregation. WavLM-large has
        # 25 hidden states (1 conv + 24 transformer layers).
        num_layers = int(self.model.config.num_hidden_layers) + 1
        if layer_aggregation == "weighted":
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        else:
            self.register_parameter("layer_weights", None)

    # ---- helpers ------------------------------------------------------------
    def _aggregate_layers(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Combine WavLM hidden states across layers."""
        if self.layer_aggregation == "last":
            return hidden_states[-1]
        if self.layer_aggregation == "mean":
            return torch.stack(hidden_states, dim=0).mean(dim=0)
        # weighted
        stacked = torch.stack(hidden_states, dim=0)               # (L, BK, T', D)
        weights = F.softmax(self.layer_weights, dim=0)            # (L,)
        return (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)   # (BK, T', D)

    # ---- forward ------------------------------------------------------------
    def forward(
        self,
        waveforms: torch.Tensor,         # (B, K, T)
        sample_mask: torch.Tensor,       # (B, K, T)
        utt_mask: torch.Tensor,          # (B, K)
    ) -> torch.Tensor:
        B, K, T = waveforms.shape
        flat_wav = waveforms.reshape(B * K, T)
        flat_mask = sample_mask.reshape(B * K, T).long()

        # (BK, T) mask with 1 for real samples -- passed only to models that were
        # pretrained with one. The frame mask below is still derived from it either
        # way, so padded frames never contribute to the pooled embedding.
        attn = flat_mask if self.use_attention_mask else None
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(input_values=flat_wav, attention_mask=attn)
        else:
            outputs = self.model(input_values=flat_wav, attention_mask=attn)

        agg = self._aggregate_layers(outputs.hidden_states)        # (BK, T', D)

        # WavLM downsamples by ~320x; recover an aligned attention mask via the model.
        downsampled_mask = self._downsample_mask(flat_mask, agg.shape[1])
        # getattr: HuBERTEncoder bypasses this class's __init__ and has no frame_pool.
        frame_pool = getattr(self, "frame_pool", None)
        if frame_pool is None:
            utt_emb = self._masked_mean_time(agg, downsampled_mask)    # (BK, D)
        else:
            utt_emb, frame_w = frame_pool(agg, downsampled_mask)      # (BK, D)
            if not self.training:
                self.last_frame_attention = (frame_w.detach(), downsampled_mask.detach())
        utt_emb = utt_emb.reshape(B, K, self.output_dim)

        # Zero out padded utterances so downstream pooling ignores them cleanly.
        utt_emb = utt_emb * utt_mask.unsqueeze(-1).to(utt_emb.dtype)
        return utt_emb

    @staticmethod
    def _downsample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Downsample a sample-level attention_mask to the encoder's frame rate.

        We don't know the exact stride of every WavLM variant, so we resample by
        nearest-neighbour. This is robust to the small off-by-one differences
        between Hugging Face WavLM revisions.
        """
        in_len = mask.shape[-1]
        if in_len == target_len:
            return mask.bool()
        # (B, T) -> (B, 1, T) -> interpolate -> (B, target_len)
        m = mask.float().unsqueeze(1)
        m = F.interpolate(m, size=target_len, mode="nearest").squeeze(1)
        return m.bool()


class Wav2Vec2Encoder(WavLMEncoder):
    """wav2vec2 utterance encoder -- the same wrapper, a different checkpoint.

    Exists so RETRO-Profile can be run on Gupta 2022's encoder. Comparing
    RETRO (WavLM) against Gupta (wav2vec2) confounds architecture with encoder;
    this makes the two directly comparable.

    Note the inherited ``use_attention_mask`` logic matters here:
    ``facebook/wav2vec2-base`` has ``feat_extract_norm == "group"`` and is
    therefore fed no attention mask, whereas WavLM Base+ is.

    A separate class rather than just a different ``pretrained_name`` so the
    provenance block written into every summary.json names the encoder that was
    actually used.
    """

    def __init__(
        self,
        pretrained_name: str = "facebook/wav2vec2-base",
        freeze: bool = True,
        layer_aggregation: str = "weighted",
        output_dim: int | None = None,
        gradient_checkpointing: bool = False,
        frame_pooling: str = "mean",
        frame_attention_hidden: int = 128,
        frame_attention_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            pretrained_name=pretrained_name,
            freeze=freeze,
            layer_aggregation=layer_aggregation,
            output_dim=output_dim,
            gradient_checkpointing=gradient_checkpointing,
            frame_pooling=frame_pooling,
            frame_attention_hidden=frame_attention_hidden,
            frame_attention_dropout=frame_attention_dropout,
        )
