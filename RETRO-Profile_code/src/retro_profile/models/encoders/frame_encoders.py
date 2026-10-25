"""Frame-level SSL encoders for the Gupta 2022 baseline.

Unlike ``BaseEncoder`` (which mean-pools each clip to one vector), the bi-encoder
transformer mixture needs the *full frame sequence* so its own transformer
encoders + statistical pooling can operate over time. These wrappers expose:

    extract_frames(waveforms, sample_mask) -> (feats, frame_mask)
        feats:      (N, T', D)  per-frame hidden states
        frame_mask: (N, T')     True where a frame is real (not padding)

Two concrete encoders:
  * ``Wav2Vec2FrameEncoder`` -- wav2vec2, TRAINABLE (conv front-end frozen);
    Gupta's native encoder, for the faithful fine-tuned run (Config A).
  * ``WavLMFrameEncoder``    -- WavLM Base+, FROZEN; the same encoder RETRO-Profile
    uses, for the controlled architecture comparison (Config B).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HFFrameEncoder(nn.Module):
    """Wrap a HuggingFace SSL model to return per-frame features + a frame mask.

    Args:
        model:      an HF model whose ``forward(input_values, attention_mask)``
                    returns an object with ``.last_hidden_state`` of shape (N,T',D).
        output_dim: D, the per-frame hidden size.
        freeze:     if True, all params are frozen and forward runs under no_grad.
        gradient_checkpointing: enable HF gradient checkpointing (trainable only;
                    trades compute for memory -- important on a Colab T4).
        use_attention_mask: pass the sample-level attention mask to the HF model.
                    Only appropriate for models pretrained with it (feat_extract_norm
                    == "layer", e.g. WavLM); wav2vec2-base ("group") is fed None.
    """

    output_dim: int

    def __init__(
        self,
        model: nn.Module,
        output_dim: int,
        freeze: bool,
        gradient_checkpointing: bool = False,
        use_attention_mask: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_dim = int(output_dim)
        self.freeze = freeze
        self.use_attention_mask = use_attention_mask

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        elif gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def extract_frames(
        self, waveforms: torch.Tensor, sample_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, T) waveforms -> (N, T', D) frames, (N, T') bool frame mask."""
        attn = (
            sample_mask.long()
            if (self.use_attention_mask and sample_mask is not None)
            else None
        )
        if self.freeze:
            with torch.no_grad():
                out = self.model(input_values=waveforms, attention_mask=attn)
        else:
            out = self.model(input_values=waveforms, attention_mask=attn)

        feats = out.last_hidden_state                      # (N, T', D)
        if sample_mask is not None:
            frame_mask = self._downsample_mask(sample_mask, feats.shape[1])
        else:
            frame_mask = torch.ones(feats.shape[:2], dtype=torch.bool, device=feats.device)
        return feats, frame_mask

    @staticmethod
    def _downsample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resize a sample-resolution mask (N, T) to frame resolution (N, T')."""
        if mask.shape[-1] == target_len:
            return mask.bool()
        m = F.interpolate(mask.float().unsqueeze(1), size=target_len, mode="nearest")
        return m.squeeze(1).bool()


class Wav2Vec2FrameEncoder(_HFFrameEncoder):
    """wav2vec2 frame encoder -- trainable (conv front-end frozen). Config A."""

    def __init__(
        self,
        pretrained_name: str = "facebook/wav2vec2-base",
        freeze: bool = False,
        freeze_feature_extractor: bool = True,
        gradient_checkpointing: bool = True,
        output_dim: int | None = None,
        **_: object,
    ) -> None:
        from transformers import Wav2Vec2Model

        model = Wav2Vec2Model.from_pretrained(pretrained_name)
        dim = int(model.config.hidden_size)
        use_attn = getattr(model.config, "feat_extract_norm", "group") == "layer"
        super().__init__(
            model=model,
            output_dim=output_dim or dim,
            freeze=freeze,
            gradient_checkpointing=gradient_checkpointing,
            use_attention_mask=use_attn,
        )
        # Freeze the CNN feature extractor even when fine-tuning the transformer
        # (standard, stabilizes wav2vec2 fine-tuning). Applied after the base set
        # requires_grad, so it re-freezes just the conv front-end.
        if freeze_feature_extractor and not freeze and hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()


class WavLMFrameEncoder(_HFFrameEncoder):
    """WavLM Base+ frame encoder -- frozen. Config B (controlled comparison)."""

    def __init__(
        self,
        pretrained_name: str = "microsoft/wavlm-base-plus",
        freeze: bool = True,
        gradient_checkpointing: bool = False,
        output_dim: int | None = None,
        **_: object,
    ) -> None:
        from transformers import WavLMModel

        model = WavLMModel.from_pretrained(pretrained_name)
        dim = int(model.config.hidden_size)
        use_attn = getattr(model.config, "feat_extract_norm", "group") == "layer"
        super().__init__(
            model=model,
            output_dim=output_dim or dim,
            freeze=freeze,
            gradient_checkpointing=gradient_checkpointing,
            use_attention_mask=use_attn,
        )
