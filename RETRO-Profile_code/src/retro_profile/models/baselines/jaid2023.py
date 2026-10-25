"""Baseline: Jaid & AbdulHassan 2023 -- 1D CNN with filter-bank init (RQ1 #3).

Reference:
    U. H. Jaid and A. K. AbdulHassan, "End-to-End Speaker Profiling Using 1D CNN
    Architectures and Filter Bank Initialization," International Journal of Online
    and Biomedical Engineering (iJOE), 19(10), pp. 65-81, 2023.

A true end-to-end model: raw waveform in, age/height/gender out, with **no SSL
encoder at all**. Its distinguishing idea is that the first convolution is
*initialised* from an auditory filter bank (gammatone or Morlet CWT) and then left
fully trainable -- a middle ground between handcrafted features and learned ones,
in contrast to SincNet, which constrains that layer for the whole of training.

Architecture (their Tables 1 and 2):

    Conv1D(1 -> 64, k=100, s=100) + ReLU     <- filter-bank initialised
    ResidualBlock(-> 32)                     } each: Conv1D(k=1) shortcut,
    ResidualBlock(-> 64)                     }        Conv1D(k=3)+ReLU, Conv1D(k=3),
    ResidualBlock(-> 64)                     }        Add, MaxPool(2), Dropout
    AvgPool1D(k=3, s=3)
    Flatten -> Dense(2)  [height, age]  and  Dense(1) [gender]

Input protocol (their Sec. 4.1): every utterance is forced to exactly 3 s
(48,000 samples at 16 kHz) -- longer clips truncated, shorter ones *repeated* then
truncated.

Deviations from the paper, and why:
  * **Parameter count.** We obtain ~71.2 K against their reported 77,603. The
    paper does not specify convolution padding, whether the dense layers follow a
    flatten or act per-timestep, or how the gender head is connected, so an exact
    match is not derivable from the text. Note that 'same' padding on the k=3
    convolutions is *forced*: with 'valid' padding the residual add would be
    shape-incompatible with the k=1 shortcut.
  * **Output parameterisation.** The paper applies ReLU to a raw-unit 2-neuron
    output. We keep that, but zero-initialise the final weight and set its bias to
    the label means, so predictions start at the dataset mean rather than at 0.
    (A raw ReLU head initialised at 0 is the degenerate start we already had to
    fix in RETRO's direct head.)
  * **Gender output.** We return a *logit*, not a sigmoid probability, so the loss
    can use BCE-with-logits -- the numerically stable, AMP-safe form.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from retro_profile.data.collate import BagBatch
from retro_profile.models.base import BaseProfileModel
from retro_profile.models.filterbanks import build_filterbank


class _ResidualBlock1D(nn.Module):
    """The paper's Table 2 block.

    A k=1 convolution forms the shortcut (and sets the channel count); two k=3
    convolutions form the residual branch. 'same' padding on the k=3 convolutions
    is required for the add to line up.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv_a = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_b = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        h = F.relu(self.conv_a(x))
        h = self.conv_b(h)
        out = F.relu(identity + h)
        return self.dropout(self.pool(out))


class Jaid2023Baseline(BaseProfileModel):
    """1D CNN with auditory filter-bank initialisation, operating on raw audio."""

    def __init__(
        self,
        n_samples: int = 48_000,               # 3 s @ 16 kHz, as in the paper
        sample_rate: int = 16_000,
        filterbank: str = "gammatone",         # gammatone | morlet | random | he_uniform
        n_filters: int = 64,                   # paper: 64 gammatone, 40 Morlet
        kernel_size: int = 100,
        block_channels: tuple[int, ...] = (32, 64, 64),
        dropout: float = 0.1,
        target_names: list[str] | None = None,
        gender_name: str = "gender",
        label_stats: dict[str, dict[str, float]] | None = None,
        **_: object,
    ) -> None:
        target_names = target_names or ["age", "height"]
        super().__init__(target_names=target_names)
        self.n_samples = int(n_samples)
        self.sample_rate = sample_rate
        self.filterbank = filterbank
        self.gender_name = gender_name

        # ---- front end: filter-bank-initialised, then freely trainable --------
        self.frontend = nn.Conv1d(1, n_filters, kernel_size=kernel_size, stride=kernel_size)
        self._init_frontend(filterbank, n_filters, kernel_size, sample_rate)

        # ---- residual trunk ---------------------------------------------------
        blocks, in_ch = [], n_filters
        for out_ch in block_channels:
            blocks.append(_ResidualBlock1D(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=3)

        # ---- heads ------------------------------------------------------------
        feat_dim = self._feature_dim(in_ch, len(block_channels))
        self.feature_dim = feat_dim
        self.regressor = nn.Linear(feat_dim, len(self.target_names))   # [age, height]
        self.gender_head = nn.Linear(feat_dim, 1)                       # logit

        # Start predictions at the dataset means instead of at 0 (see module
        # docstring). Consequence: at step 0 the regression path contributes no
        # gradient to the trunk (dL/dfeats = W^T g = 0); the regressor's own
        # weight still receives gradient, so this bootstraps after one update,
        # and the gender head feeds the trunk in the meantime.
        nn.init.zeros_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
        for t in ("age", "height"):
            self.register_buffer(f"_{t}_mean", torch.tensor(0.0))
        if label_stats:
            self.set_label_stats(label_stats)

    # ---- construction helpers ---------------------------------------------
    def _feature_dim(self, channels: int, n_blocks: int) -> int:
        """Length after the front end, the blocks' max-pools, and the avg-pool."""
        length = self.n_samples // self.frontend.stride[0]
        for _ in range(n_blocks):
            length //= 2
        length //= 3
        if length < 1:
            raise ValueError(
                f"n_samples={self.n_samples} is too short for this stack "
                f"(need >= {self.frontend.stride[0] * 2 ** n_blocks * 3} samples)."
            )
        return channels * length

    def _init_frontend(self, name: str, n_filters: int, kernel_size: int, sample_rate: int) -> None:
        """Seed the first conv with an auditory filter bank (or leave it random)."""
        if name in ("random", "he_uniform"):
            if name == "he_uniform":
                nn.init.kaiming_uniform_(self.frontend.weight, nonlinearity="relu")
            return
        bank = build_filterbank(name, n_filters=n_filters, length=kernel_size,
                                **({"sample_rate": sample_rate} if name == "gammatone" else {}))
        with torch.no_grad():
            self.frontend.weight.copy_(bank.unsqueeze(1))   # (n_filters, 1, kernel)
            self.frontend.bias.zero_()

    def set_label_stats(self, label_stats: dict[str, dict[str, float]]) -> None:
        """Bias the regressor so it starts at the training-set means."""
        for t in ("age", "height"):
            st = label_stats.get(t)
            if st:
                getattr(self, f"_{t}_mean").fill_(float(st["mean"]))
        with torch.no_grad():
            for i, t in enumerate(self.target_names):
                self.regressor.bias[i] = float(getattr(self, f"_{t}_mean"))

    # ---- input conditioning -------------------------------------------------
    def _fix_length(self, waveforms: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        """Force every utterance to exactly ``n_samples`` (the paper's protocol).

        Longer clips are truncated; shorter ones are *repeated* and then truncated,
        rather than zero-padded -- zero padding would feed silence to a model that
        has no masking anywhere in its stack.
        """
        n, target = waveforms.shape[0], self.n_samples
        lengths = sample_mask.sum(dim=1).clamp(min=1)                      # (N,)
        idx = torch.arange(target, device=waveforms.device).unsqueeze(0)   # (1, target)
        wrapped = idx % lengths.unsqueeze(1)                               # tile within real audio
        return waveforms.gather(1, wrapped.clamp(max=waveforms.shape[1] - 1)).view(n, target)

    @staticmethod
    def _agg_over_bag(x: torch.Tensor, utt_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of per-utterance values over the bag. (B, K) -> (B,)."""
        m = utt_mask.to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)

    # ---- forward ------------------------------------------------------------
    def forward(self, batch: BagBatch) -> dict[str, torch.Tensor]:
        B, K, T = batch.waveforms.shape
        flat = batch.waveforms.reshape(B * K, T)
        flat_mask = batch.sample_mask.reshape(B * K, T)

        x = self._fix_length(flat, flat_mask).unsqueeze(1)     # (BK, 1, n_samples)
        x = F.relu(self.frontend(x))
        x = self.blocks(x)
        x = self.avgpool(x)
        feats = x.flatten(start_dim=1)                         # (BK, feature_dim)

        preds = F.relu(self.regressor(feats))                  # (BK, n_targets), real units
        gender_logit = self.gender_head(feats).squeeze(-1)     # (BK,)

        out: dict[str, torch.Tensor] = {}
        for i, t in enumerate(self.target_names):
            out[t] = self._agg_over_bag(preds[:, i].view(B, K), batch.utt_mask)
        out[self.gender_name] = self._agg_over_bag(gender_logit.view(B, K), batch.utt_mask)
        return out

    # embed_speaker: inherit BaseProfileModel's NotImplementedError -- this is not
    # a retrieval method and does not build a kNN datastore.
