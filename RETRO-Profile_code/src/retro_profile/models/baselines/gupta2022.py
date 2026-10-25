"""Baseline: Gupta et al. 2022 -- bi-encoder transformer mixture (RQ1 baseline #2).

Reference:
    T. Gupta et al., "Estimation of speaker age and height from speech signal
    using a bi-encoder transformer mixture model," arXiv:2203.11774, 2022.

Core idea (reproduced here, not copied line-for-line):
    * a frame-level SSL encoder (wav2vec2 for the faithful run; frozen WavLM for
      the controlled run) produces per-frame features;
    * two transformer encoders (the "bi-encoder") -- an M branch and an F branch;
    * statistical (mean+std) pooling per branch;
    * a predicted gender gate mixes the branches: out = (1-g)*xM + g*xF
      ("transformer mixture") -- gender is the routing signal;
    * age and height are regressed from the mixed vector; gender is an auxiliary
      task (needed to make the gate gender-meaningful; not reported).

Integration with our harness:
    * ``forward`` consumes a per-speaker bag, runs the model per utterance, and
      averages predictions to per-speaker (Gupta's per-utterance-then-average
      protocol) so metrics are directly comparable to RETRO-Profile.
    * predictions are returned in REAL label units (denormalized via label-stat
      buffers that are saved in the checkpoint), matching our head convention.
    * ``embed_speaker`` is intentionally not implemented -- Gupta is not a
      retrieval method, so it does not build a kNN datastore (a fair omission).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.data.collate import BagBatch
from retro_profile.models.base import BaseProfileModel


class Gupta2022Baseline(BaseProfileModel):
    """Gupta 2022 bi-encoder transformer mixture for age + height (+ gender aux)."""

    def __init__(
        self,
        encoder: nn.Module,                 # a frame encoder exposing extract_frames()
        feature_dim: int = 768,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.5,
        transformer_dropout: float = 0.1,
        fc_dim: int = 1024,
        target_names: list[str] | None = None,
        gender_name: str = "gender",
        label_stats: dict[str, dict[str, float]] | None = None,
        **_: object,
    ) -> None:
        target_names = target_names or ["age", "height"]
        super().__init__(target_names=target_names)
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.gender_name = gender_name

        # Two distinct dropout rates, as in Gupta's own code: the transformer
        # stacks use PyTorch's default 0.1, while the heavy 0.5 applies only to
        # the pooled/FC vectors. Feeding 0.5 into 12 transformer layers (6 per
        # branch) stops the model learning at all -- it previously plateaued at
        # epoch 3 of 40.
        self.transformer_dropout = float(transformer_dropout)

        def _make_transformer() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=transformer_dropout, batch_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.transformer_M = _make_transformer()      # the bi-encoder: M branch
        self.transformer_F = _make_transformer()      #                 F branch

        self.fcM = nn.Linear(2 * feature_dim, fc_dim)  # after mean+std pooling
        self.fcF = nn.Linear(2 * feature_dim, fc_dim)
        self.dropout = nn.Dropout(dropout)

        self.gender_head = nn.Linear(2 * fc_dim, 1)    # the mixture gate
        self.age_head = nn.Linear(fc_dim, 1)
        self.height_head = nn.Linear(fc_dim, 1)

        # Label mean/std as buffers so predictions can be denormalized to real
        # units and the stats travel with the checkpoint (no need to recompute
        # them at eval time). Default is identity (mean 0, std 1).
        for t in ("age", "height"):
            self.register_buffer(f"_{t}_mean", torch.tensor(0.0))
            self.register_buffer(f"_{t}_std", torch.tensor(1.0))
        if label_stats:
            self.set_label_stats(label_stats)

    # ---- label normalization ----------------------------------------------
    def set_label_stats(self, label_stats: dict[str, dict[str, float]]) -> None:
        """Set per-target mean/std (called after the DataModule computes them)."""
        for t in ("age", "height"):
            st = label_stats.get(t)
            if not st:
                continue
            getattr(self, f"_{t}_mean").fill_(float(st["mean"]))
            getattr(self, f"_{t}_std").fill_(float(st["std"]))

    # ---- pooling / aggregation helpers ------------------------------------
    @staticmethod
    def _stat_pool(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Masked mean+std over time. x:(N,T',D) m:(N,T',1) -> (N,2D)."""
        denom = m.sum(dim=1).clamp_min(1e-6)                     # (N,1)
        mean = (x * m).sum(dim=1) / denom                       # (N,D)
        var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        std = (var + 1e-8).sqrt()
        return torch.cat((mean, std), dim=1)                    # (N,2D)

    @staticmethod
    def _agg_over_bag(x: torch.Tensor, utt_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of per-utterance values over the bag. x,(B,K) -> (B,)."""
        m = utt_mask.to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)

    # ---- forward ----------------------------------------------------------
    def forward(self, batch: BagBatch) -> dict[str, torch.Tensor]:
        B, K, T = batch.waveforms.shape
        flat = batch.waveforms.reshape(B * K, T)
        flat_mask = batch.sample_mask.reshape(B * K, T)

        feats, frame_mask = self.encoder.extract_frames(flat, flat_mask)   # (BK,T',D),(BK,T')
        # No transformer padding mask (as in Gupta's code); pooling is masked
        # instead, which also avoids NaN on fully-padded utterance slots.
        xM = self.transformer_M(feats)
        xF = self.transformer_F(feats)

        m = frame_mask.unsqueeze(-1).to(feats.dtype)            # (BK,T',1)
        xM = self.dropout(self._stat_pool(xM, m))               # (BK,2D)
        xF = self.dropout(self._stat_pool(xF, m))
        xM = self.dropout(self.fcM(xM))                         # (BK,fc)
        xF = self.dropout(self.fcF(xF))

        g_logit = self.gender_head(torch.cat((xM, xF), dim=1)).squeeze(-1)  # (BK,) logit
        g = torch.sigmoid(g_logit)                                         # (BK,) gate prob
        mixed = (1 - g).unsqueeze(-1) * xM + g.unsqueeze(-1) * xF          # (BK,fc)

        age_n = self.age_head(mixed).squeeze(-1)                # (BK,) normalized
        hgt_n = self.height_head(mixed).squeeze(-1)

        # per-utterance -> per-speaker (Gupta's aggregate-at-test protocol)
        utt = batch.utt_mask                                    # (B,K)
        age_n = self._agg_over_bag(age_n.view(B, K), utt)       # (B,)
        hgt_n = self._agg_over_bag(hgt_n.view(B, K), utt)
        # Return the gender LOGIT (not the probability) so the trainer can use
        # BCE-with-logits, which is the AMP/fp16-safe, numerically-stable choice.
        # The sigmoid gate above still uses the probability internally.
        g_sp = self._agg_over_bag(g_logit.view(B, K), utt)     # (B,) gender logit

        # denormalize to real units (years, cm)
        age = age_n * self._age_std + self._age_mean
        hgt = hgt_n * self._height_std + self._height_mean
        return {"age": age, "height": hgt, self.gender_name: g_sp}

    # embed_speaker: inherit BaseProfileModel's NotImplementedError -- Gupta is
    # not a retrieval method and does not build a kNN datastore.
