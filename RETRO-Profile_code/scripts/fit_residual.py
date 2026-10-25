"""Second stage for RETRO (2026-10-15): train ONLY the residual correction on top of kNN, model frozen.

The final TIMIT RETRO trains with kNN off, so each trait's residual network (``ResidualRetrievalHead.residual``)
never receives a gradient: with kNN switched on at test its correction is exactly 0 and the prediction is the kNN
estimate alone. The proposal's model is "retrieval estimate + small learned correction". This script trains that
correction afterwards, separately for every trait, without touching anything else:

  1. load training's chosen checkpoint (select_checkpoint) and freeze the whole model;
  2. build the kNN datastore from the training speakers, every clip (the build_datastore.py recipe);
  3. embed train / validation / test speakers as bags of up to --bag-size clips (TIMIT 10 = all, fixed order);
  4. choose k per trait on VALIDATION with kNN alone (lowest MAE; ties -> smaller k);
  5. train only that trait's residual MLP on the training speakers, each against its own kNN estimate with itself
     left out of the lookup, keeping the epoch with the lowest VALIDATION MAE (epoch 0 = no correction);
  6. score validation and test three ways: ``direct`` (kNN off, the trained head), ``knn`` (kNN alone) and
     ``knn_residual`` (kNN + trained correction).

Rule fixed in advance (2026-10-15): the residual is kept for a trait only if it lowers that trait's validation MAE;
both readings are written either way. Test labels never enter any choice. Before training, the script checks that
its kNN-off test predictions reproduce the training job's own (``--check-sub``), so it scores the same model.

Cross-corpus (Table 5, 2026-10-16): ``--train-from-run`` builds the datastore and the residual's training speakers from
the run's OWN training corpora (its .hydra data config, e.g. HeightCeleb + NISP), while validation and test stay on
``--corpus`` (TIMIT). ``--calibrated`` then chooses k and the residual's epoch on the CALIBRATED validation MAE (each
validation speaker calibrated by a 2-number fit on the others, as calibrate_predictions.py) and also writes calibrated
readings: test with the fit on all validation speakers, validation leave-one-out (``predictions_<split>_<reading>_cal.csv``,
``results['<split>_cal/<reading>']``). Same model, same three readings.

Writes ``runs/<run>/stage2<suffix>/``: predictions_<split>_<reading>.csv, summary.json, residual_<target>.pt.

    python scripts/fit_residual.py --run runs/retro_ordcon_fixed_l100_s5_b10_timit_s42 --bag-size 10
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from retro_profile.data.collate import make_collate_fn
from retro_profile.data.registry import build_datamodule
from retro_profile.retrieval.builder import build_datastore
from retro_profile.retrieval.datastore import RetrievalDatastore
from retro_profile.utils.config import instantiate, load_config
from retro_profile.utils.seed import seed_everything

K_GRID = (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300)
READINGS = ("direct", "knn", "knn_residual")
_SCRIPTS = Path(__file__).resolve().parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- model + data -------------------------------------------------------------------------------------------------
def load_frozen_model(run: Path):
    """The run's model with training's chosen weights, eval mode, every parameter frozen."""
    cfg = load_config(run / ".hydra" / "config.yaml")
    model = instantiate(cfg.model)
    ckpt_path = _script("select_checkpoint").select_checkpoint(run)
    state = torch.load(str(ckpt_path), map_location="cpu")
    state = state.get("state_dict", state)
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt_path


def corpus_data_cfg(corpus: str, bag_size: int, num_workers: int, root: str | None = None) -> dict:
    """The corpus data config as evaluate.py builds it, with the bag size set and augmentation OFF."""
    path = _SCRIPTS.parent / "configs" / "data" / f"{corpus}.yaml"
    cfg = OmegaConf.to_container(load_config(path), resolve=True)
    cfg.pop("_target_", None)
    cfg.pop("name", None)
    cfg = {**cfg, "bag_size": int(bag_size), "augmentation": {"enabled": False}, "num_workers": num_workers}
    return {**cfg, "root": root} if root else cfg


def _bag_loader(dm, ds) -> DataLoader:
    return DataLoader(ds, batch_size=dm.batch_size, shuffle=False, num_workers=dm.num_workers,
                      collate_fn=make_collate_fn(dm.targets, mode="bag"))


def bag_loaders(corpus: str, data_cfg: dict, with_train: bool = True) -> tuple[object, dict[str, DataLoader]]:
    """Datamodule (for the utterance-level datastore) and fixed-order bag loaders for train / val / test."""
    dm = build_datamodule(corpus, **data_cfg)
    dm.setup("fit")
    dm.setup("test")
    loaders = {"val": _bag_loader(dm, dm.val_set), "test": _bag_loader(dm, dm.test_set)}
    if with_train:
        train_bags = dm.DATASET_CLS(split=dm.splits["train"], mode="bag", transforms=None, root=dm.root,
                                    sample_rate=dm.sample_rate, max_audio_seconds=dm.max_audio_seconds,
                                    min_audio_seconds=dm.min_audio_seconds, bag_size=dm.bag_size, targets=dm.targets)
        loaders["train"] = _bag_loader(dm, train_bags)
    return dm, loaders


def run_training_side(run: Path, num_workers: int) -> tuple[DataLoader, DataLoader]:
    """(utterance loader for the datastore, fixed-order bag loader) over the run's OWN training corpora, no augmentation."""
    cfg = load_config(run / ".hydra" / "config.yaml")
    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    dm = instantiate({**dcfg, "num_workers": num_workers, "augmentation": {"enabled": False}})
    if hasattr(dm, "_dataset"):                                    # MultiCorpusDataModule
        bags = dm._dataset("train", "bag", None)
    else:
        dm.setup("fit")
        bags = dm.DATASET_CLS(split=dm.splits["train"], mode="bag", transforms=None, root=dm.root,
                              sample_rate=dm.sample_rate, max_audio_seconds=dm.max_audio_seconds,
                              min_audio_seconds=dm.min_audio_seconds, bag_size=dm.bag_size, targets=dm.targets)
    return dm.utterance_loader("train"), _bag_loader(dm, bags)


@torch.no_grad()
def embed(model, loader: DataLoader, device: str) -> dict:
    """Per speaker: per-target head input, shared kNN query, kNN-off prediction, ground truth (NaN if missing)."""
    out: dict = {"speaker_ids": [], "query": [], "z": {}, "direct": {}, "truth": {}}
    for batch in loader:
        batch = batch.to(device)
        projected, _, query = model._pool_and_project(batch)
        out["speaker_ids"] += [str(s) for s in batch.speaker_ids]
        out["query"].append(query.float().cpu())
        for t, head in model.heads.items():
            out["z"].setdefault(t, []).append(projected[t].float().cpu())
            out["direct"].setdefault(t, []).append(head(projected[t]).float().cpu())   # no datastore: direct
            y = batch.labels[t].float().cpu().clone()
            y[~batch.label_mask[t].cpu()] = float("nan")
            out["truth"].setdefault(t, []).append(y)
    out["query"] = torch.cat(out["query"])
    for key in ("z", "direct", "truth"):
        out[key] = {t: torch.cat(v) for t, v in out[key].items()}
    return out


# --- kNN + residual -------------------------------------------------------------------------------------------------
def knn_estimate(ds: RetrievalDatastore, split: dict, target: str, k: int) -> torch.Tensor:
    """kNN estimate (unclamped, as the head computes it); each speaker is left out of its own lookup."""
    return ds.knn_anchor(split["query"], target=target, k=min(k, len(ds)),
                         exclude_speaker_ids=split["speaker_ids"]).float()


def residual_predict(head, z: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """The head's own retrieval formula: clamp(anchor + delta_clip * tanh(residual(z)))."""
    delta = torch.tanh(head.residual(z).squeeze(-1)) * head.delta_clip
    return (anchor + delta).clamp(head.label_min, head.label_max)


def _mae(pred: torch.Tensor, y: torch.Tensor, calibrated: bool = False) -> float:
    """MAE over rows with a label; ``calibrated`` = after the 2-number fit, each row left out of its own fit."""
    ok = ~torch.isnan(y)
    if not calibrated:
        return float((pred[ok] - y[ok]).abs().mean())
    v = pd.DataFrame({"prediction": pred[ok].numpy().astype(float), "ground_truth": y[ok].numpy().astype(float)})
    return float((_script("calibrate_predictions").loo_predictions(v) - v["ground_truth"]).abs().mean())


def choose_k(ds: RetrievalDatastore, val: dict, target: str, label_range: tuple[float, float],
             k_grid=K_GRID, calibrated: bool = False) -> tuple[int, dict[int, float]]:
    """k with the lowest validation MAE of the kNN estimate alone (calibrated, if asked; ties -> smaller k)."""
    table = {}
    for k in sorted({min(k, len(ds)) for k in k_grid}):
        table[k] = _mae(knn_estimate(ds, val, target, k).clamp(*label_range), val["truth"][target], calibrated)
    best = min(table, key=lambda k: (round(table[k], 9), k))
    return best, table


def train_residual(head, z_tr, a_tr, y_tr, z_va, a_va, y_va, y_scale: float, epochs: int = 300,
                   lr: float = 1e-3, weight_decay: float = 1e-2, batch_size: int = 64, patience: int = 50,
                   seed: int = 42, calibrated: bool = False) -> dict:
    """Train ``head.residual`` only (everything else stays frozen); keep the best-validation epoch.

    Epoch 0 is the untrained, zero-output residual, i.e. kNN alone, so the result can never be worse on
    validation than kNN alone. Loss: smooth L1 on errors divided by the training labels' spread (as in training).
    """
    torch.manual_seed(seed)
    ok = ~torch.isnan(y_tr)
    z_tr, a_tr, y_tr = z_tr[ok], a_tr[ok], y_tr[ok]
    params = list(head.residual.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def val_mae() -> float:
        head.residual.eval()
        with torch.no_grad():
            return _mae(residual_predict(head, z_va, a_va), y_va, calibrated)

    best = {"epoch": 0, "val_mae": val_mae(), "state": copy.deepcopy(head.residual.state_dict())}
    history = [best["val_mae"]]
    since = 0
    for epoch in range(1, epochs + 1):
        head.residual.train()
        for idx in torch.randperm(len(y_tr)).split(batch_size):
            loss = F.smooth_l1_loss(residual_predict(head, z_tr[idx], a_tr[idx]) / y_scale, y_tr[idx] / y_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = val_mae()
        history.append(m)
        if m < best["val_mae"] - 1e-9:
            best = {"epoch": epoch, "val_mae": m, "state": copy.deepcopy(head.residual.state_dict())}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    head.residual.load_state_dict(best["state"])
    head.residual.eval()
    for p in params:
        p.requires_grad_(False)
    return {"best_epoch": best["epoch"], "best_val_mae": best["val_mae"], "knn_val_mae": history[0],
            "epochs_run": len(history) - 1, "history": history}


# --- scoring -----------------------------------------------------------------------------------------------------
def genders(root: str | Path) -> pd.Series:
    """speaker_id -> 'M'/'F' from every manifest of the corpus."""
    frames = [pd.read_csv(f, usecols=["speaker_id", "gender"]) for f in sorted(Path(root, "manifests").glob("*.csv"))]
    g = pd.concat(frames).drop_duplicates("speaker_id")
    return g.set_index(g["speaker_id"].astype(str))["gender"].astype(str).str.upper().str[0]


def frame(split: dict, preds: dict[str, torch.Tensor], gender: pd.Series) -> pd.DataFrame:
    rows = []
    for t, p in preds.items():
        rows.append(pd.DataFrame({"speaker_id": split["speaker_ids"], "target": t, "prediction": p.numpy(),
                                  "ground_truth": split["truth"][t].numpy()}))
    df = pd.concat(rows, ignore_index=True)
    df["gender"] = df["speaker_id"].map(gender)
    return df


def calibrated_frames(val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per target: test calibrated with the 2-number fit on ALL validation rows; validation leave-one-out."""
    cal = _script("calibrate_predictions")
    vout, tout = [], []
    for t, tg in test.groupby("target"):
        vg = val[(val.target == t) & val.ground_truth.notna()].copy()
        p = cal.fit(vg)
        tout.append(tg.assign(prediction=p["a"] * tg["prediction"] + p["b"]))
        vout.append(vg.assign(prediction=cal.loo_predictions(vg).to_numpy()))
    return pd.concat(vout, ignore_index=True), pd.concat(tout, ignore_index=True)


def check_reproduces(direct_test: pd.DataFrame, reference: Path, tol: float = 0.05) -> float:
    """Max |difference| between our kNN-off test predictions and the training job's own; stop if too large."""
    ref = pd.read_csv(reference, dtype={"speaker_id": str}).set_index(["speaker_id", "target"])["prediction"]
    ours = direct_test.set_index(["speaker_id", "target"])["prediction"]
    common = ours.index.intersection(ref.index)
    if len(common) != len(ours):
        raise SystemExit(f"{reference}: covers {len(common)} of our {len(ours)} rows -- different speakers. Stop.")
    diff = float((ours.loc[common] - ref.loc[common]).abs().max())
    if not diff <= tol:
        raise SystemExit(f"kNN-off test predictions differ from {reference} by up to {diff:.4g} -- this would not "
                         "score the model training evaluated. Stop.")
    return diff


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--corpus", default="timit")
    ap.add_argument("--bag-size", type=int, default=10)
    ap.add_argument("--suffix", default="_b10", help="output folder is stage2<suffix>")
    ap.add_argument("--check-sub", default="eval_spk",
                    help="the run's kNN-off TEST predictions at the same bag size, to reproduce first "
                         "(b10 runs: eval_spk; 8-clip runs re-scored on 10: eval_spk_b10); '' skips")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default=None, help="corpus root override (default: the corpus config's)")
    ap.add_argument("--train-from-run", action="store_true",
                    help="datastore + residual training speakers from the run's own training corpora (cross-corpus)")
    ap.add_argument("--calibrated", action="store_true",
                    help="choose k and the residual epoch on calibrated validation MAE; also write calibrated readings")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.run / f"stage2{args.suffix}"
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_frozen_model(args.run)
    model.to(device)
    data_cfg = corpus_data_cfg(args.corpus, args.bag_size, args.num_workers, args.data_root)
    dm, loaders = bag_loaders(args.corpus, data_cfg, with_train=not args.train_from_run)
    gender = genders(data_cfg["root"])
    store_loader = dm.utterance_loader("train")
    if args.train_from_run:
        store_loader, loaders["train"] = run_training_side(args.run, args.num_workers)

    ds = build_datastore(model=model, dataloader=store_loader, targets=model.target_names,
                         metric="cosine", temperature=0.1, weighting="softmax_distance", device=device)
    ds.save(out / "datastore")
    splits = {name: embed(model, ld, device) for name, ld in loaders.items()}
    model.to("cpu")
    print(f"{args.run.name}: checkpoint {ckpt.name}; datastore {len(ds)} speakers; bags of up to {args.bag_size}; "
          + ", ".join(f"{s} {len(v['speaker_ids'])}" for s, v in splits.items()))

    direct_test = frame(splits["test"], splits["test"]["direct"], gender)
    reproduced = None
    if args.check_sub:
        reproduced = check_reproduces(direct_test, args.run / args.check_sub / f"predictions_{args.corpus}.csv")
        print(f"reproduces the job's kNN-off test predictions (max diff {reproduced:.2g})")

    summary: dict = {"run": args.run.name, "checkpoint": ckpt.name, "bag_size": args.bag_size,
                     "train_from_run": args.train_from_run, "calibrated_selection": args.calibrated,
                     "datastore_size": len(ds), "reproduction_max_diff": reproduced,
                     "residual_training": {"epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
                                           "patience": args.patience, "seed": args.seed},
                     "targets": {}, "results": {}}
    preds = {s: {r: {} for r in READINGS} for s in ("val", "test")}
    for t, head in model.heads.items():
        rng = (float(head.label_min), float(head.label_max))
        k, table = choose_k(ds, splits["val"], t, rng, calibrated=args.calibrated)
        a = {s: knn_estimate(ds, splits[s], t, k) for s in ("train", "val", "test")}
        y_scale = float(np.nanstd(splits["train"]["truth"][t].numpy())) or 1.0
        fit = train_residual(head, splits["train"]["z"][t], a["train"], splits["train"]["truth"][t],
                             splits["val"]["z"][t], a["val"], splits["val"]["truth"][t], y_scale,
                             epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                             patience=args.patience, seed=args.seed, calibrated=args.calibrated)
        torch.save(head.residual.state_dict(), out / f"residual_{t}.pt")
        for s in ("val", "test"):
            preds[s]["direct"][t] = splits[s]["direct"][t]
            preds[s]["knn"][t] = a[s].clamp(*rng)
            with torch.no_grad():
                preds[s]["knn_residual"][t] = residual_predict(head, splits[s]["z"][t], a[s])
        keep = fit["best_epoch"] > 0
        summary["targets"][t] = {"k": k, "val_mae_by_k": table, "residual_kept": keep,
                                 **{key: fit[key] for key in ("best_epoch", "best_val_mae", "knn_val_mae", "epochs_run")}}
        print(f"  {t:6s} k {k:3d} | {'calibrated ' if args.calibrated else ''}val MAE kNN alone {fit['knn_val_mae']:.3f} -> + residual {fit['best_val_mae']:.3f} "
              f"(epoch {fit['best_epoch']} of {fit['epochs_run']}) -> {'KEEP' if keep else 'no gain'}")

    blocks = _script("evaluate")._gender_blocks
    for r in READINGS:
        frames = {s: frame(splits[s], preds[s][r], gender) for s in ("val", "test")}
        for s, df in frames.items():
            df.to_csv(out / f"predictions_{s}_{r}.csv", index=False)
            summary["results"][f"{s}/{r}"] = blocks(df)
        if args.calibrated:
            vcal, tcal = calibrated_frames(frames["val"], frames["test"])
            for s, df in (("val", vcal), ("test", tcal)):
                df.to_csv(out / f"predictions_{s}_{r}_cal.csv", index=False)
                summary["results"][f"{s}_cal/{r}"] = blocks(df)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for s in ("val", "test") + (("val_cal", "test_cal") if args.calibrated else ()):
        line = " | ".join(f"{r} " + " / ".join(f"{summary['results'][f'{s}/{r}']['overall'][f'{t}/mae']:.2f}"
                                               for t in model.heads) for r in READINGS)
        print(f"  {s:8s} MAE ({' / '.join(model.heads)}): {line}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
