"""Regression metrics.

We avoid pulling in ``torchmetrics`` heavyweight base classes — for scalar
regression a few lines of NaN-aware torch are clearer and easier to debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Chance-level MAE on TIMIT -- predicting the training mean for every speaker
# (SESSION_HANDOFF.md §2). Used only to put age (years) and height (cm) on a
# common scale for `combined_mae`; override per corpus via the constructor.
DEFAULT_COMBINED_NORMALISERS: dict[str, float] = {"age": 5.91, "height": 7.40}


@dataclass
class RegressionMetrics:
    """Streaming MAE / RMSE / Pearson r per target.

    Use:
        m = RegressionMetrics(["age", "height"])
        for batch in loader:
            preds = model(batch)
            m.update(preds, batch.labels, batch.label_mask)
        results = m.compute()    # {"age/mae": ..., "height/mae": ..., ...}
        m.reset()
    """

    target_names: list[str]
    combined_normalisers: dict[str, float] | None = None
    _state: dict[str, dict[str, list[torch.Tensor]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state = {t: {"pred": [], "target": []} for t in self.target_names}

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        label_mask: dict[str, torch.Tensor],
    ) -> None:
        for t in self.target_names:
            if t not in predictions or t not in labels:
                continue
            mask = label_mask[t] & ~torch.isnan(labels[t])
            if not mask.any():
                continue
            self._state[t]["pred"].append(predictions[t][mask].detach().cpu())
            self._state[t]["target"].append(labels[t][mask].detach().cpu())

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for t, st in self._state.items():
            if not st["pred"]:
                continue
            pred = torch.cat(st["pred"])
            tgt = torch.cat(st["target"])
            err = pred - tgt
            mae = err.abs().mean().item()
            rmse = (err.pow(2).mean().sqrt()).item()
            # Pearson r (population), guarded against zero variance
            if pred.numel() > 1 and pred.std().item() > 1e-8 and tgt.std().item() > 1e-8:
                cov = ((pred - pred.mean()) * (tgt - tgt.mean())).mean()
                pearson = (cov / (pred.std() * tgt.std())).item()
            else:
                pearson = float("nan")
            results[f"{t}/mae"] = mae
            results[f"{t}/rmse"] = rmse
            results[f"{t}/pearson"] = pearson
            results[f"{t}/n"] = float(pred.numel())

        # Combined selection metric (SESSION_HANDOFF.md §7.2).
        #
        # Checkpointing on `val/height/mae` truncates training before age
        # converges (rule R5); checkpointing on `val/total` is no better, because
        # that is the uncertainty-weighted *training loss* and is dominated by
        # age, whose dynamic range over a run is ~10x height's. On E11 it chose
        # epoch 48 (val age 4.54 / height 5.97) when epoch 26 was better on BOTH
        # targets (4.13 / 5.29). This sums the reported MAEs, each divided by its
        # chance-level MAE so years and centimetres are comparable.
        #
        # Only targets that (a) appeared in this epoch and (b) have a normaliser
        # contribute, so HeightCeleb's all-NaN age column drops out instead of
        # poisoning the sum, and non-regression targets like `gender` are never
        # included. Runs are only comparable when the same targets are present.
        norms = self.combined_normalisers or DEFAULT_COMBINED_NORMALISERS
        terms = [
            results[f"{t}/mae"] / norms[t]
            for t in self.target_names
            if f"{t}/mae" in results and norms.get(t, 0.0) > 0
        ]
        if terms:
            results["combined_mae"] = sum(terms)
        return results
