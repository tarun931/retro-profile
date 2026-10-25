"""Mixup for speaker-trait regression (Gupta et al. 2022; Truong et al. 2022), training batches only (2026-10-18).

Each example is blended with a random other example of the same batch: audio ``lam * a + (1 - lam) * b`` and every
label the same way (age, height, and gender as a soft 0-1 target), with ``lam ~ Beta(alpha, alpha)`` per example
(``alpha = 1``: uniform on [0, 1], as Truong et al. write it). Utterance-level training (Gupta, Jaid: bags of one
clip) mixes clip with clip, as in the papers; bag-level training (RETRO) mixes two speakers' bags clip by clip with
one ``lam`` per pair. A label missing on either side stays missing.

Off unless ``training.mixup.enabled`` is set; when off nothing here runs, so no random number is drawn and every
earlier result trains exactly as before.
"""

from __future__ import annotations

import torch

from retro_profile.data.collate import BagBatch


def mixup_batch(batch: BagBatch, alpha: float) -> BagBatch:
    """A new batch: each row blended with a random partner row (a fresh permutation of the batch)."""
    n = batch.batch_size
    if n < 2:
        return batch
    device = batch.waveforms.device
    lam = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha))).sample((n,)).to(device)
    perm = torch.randperm(n, device=device)
    w = lam.view(n, 1, 1).to(batch.waveforms.dtype)
    labels, label_mask = {}, {}
    for t, y in batch.labels.items():
        both = batch.label_mask[t] & batch.label_mask[t][perm]
        lt = lam.to(y.dtype)
        mixed = lt * torch.nan_to_num(y) + (1 - lt) * torch.nan_to_num(y[perm])
        labels[t] = torch.where(both, mixed, torch.full_like(y, float("nan")))
        label_mask[t] = both
    return BagBatch(
        waveforms=w * batch.waveforms + (1 - w) * batch.waveforms[perm],
        utt_mask=batch.utt_mask | batch.utt_mask[perm],
        sample_mask=batch.sample_mask | batch.sample_mask[perm],
        labels=labels,
        label_mask=label_mask,
        speaker_ids=list(batch.speaker_ids),
        utt_ids=list(batch.utt_ids),
    )
