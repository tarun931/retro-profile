"""Fun demo (2026-10-18): one recording's age and height from each Table 1 model, as a deployed system would do it.

Not part of any result. Per model, ONE trained copy is used -- the seed whose training-chosen checkpoint has the best
validation score (the val= in best_checkpoint.txt; what a team would deploy). It gets the clip as a one-clip bag, cut
to 8 s like every score. RETRO is read as in Table 1: kNN + residual from that run's second stage (its lookup table of
392 TIMIT training speakers, its k per trait, its residual weights); its kNN-off prediction and the 5 nearest training
speakers are reported too. One clip, recorded outside TIMIT (studio, American read speech): expect large errors -- age
drifts towards TIMIT's typical 20-40, height towards a gender average.

    python scripts/predict_clip.py --audio fun/clip_001.wav --out fun/predictions_clip_001.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import torch

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import make_collate_fn
from retro_profile.retrieval.datastore import RetrievalDatastore
from retro_profile.utils.audio import load_audio

_SCRIPTS = Path(__file__).resolve().parent
TABLE1 = {"RETRO-Profile": "retro_ordcon_fixed_l100_s5_b10", "WavLM + MLP": "wavlm_direct_budget_b10_e100",
          "Gupta 2022 (WavLM)": "gupta2022_finetune_wavlm_lr3e6", "Jaid 2023": "jaid2023_lr3e3"}
SEEDS = (42, 1, 2, 3, 4)


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def best_seed_run(exp: str, runs: Path = Path("runs")) -> tuple[Path, float]:
    """The seed whose training-chosen checkpoint has the lowest validation score (val=...), and that score."""
    scored = []
    for seed in SEEDS:
        run = runs / f"{exp}_timit_s{seed}"
        name = Path((run / "best_checkpoint.txt").read_text().strip()).name
        scored.append((float(re.search(r"val=([0-9.]+)\.ckpt", name).group(1)), run))
    val, run = min(scored, key=lambda x: x[0])
    return run, val


def clip_batch(path: Path, targets, sample_rate: int = 16_000, max_seconds: float = 8.0):
    """The clip as a one-speaker, one-clip bag (no labels), trimmed to ``max_seconds`` from the start."""
    wav = load_audio(path, target_sr=sample_rate)[: int(max_seconds * sample_rate)]
    bag = SpeakerBag(waveforms=[wav], sample_rate=sample_rate, speaker_id="clip", utt_ids=["clip"], labels={})
    return make_collate_fn(list(targets), mode="bag")([bag])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--neighbours", type=int, default=5)
    args = ap.parse_args(argv)

    fr = _script("fit_residual")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timit = pd.read_csv("data/timit/manifests/train.csv", dtype={"speaker_id": str}).drop_duplicates("speaker_id")
    timit = timit.set_index("speaker_id")
    rows, nbrs = [], []
    for model_name, exp in TABLE1.items():
        run, val = best_seed_run(exp)
        model, ckpt = fr.load_frozen_model(run)
        model.to(device)
        batch = clip_batch(args.audio, model.target_names).to(device)
        row = {"model": model_name, "run": run.name, "checkpoint": ckpt.name, "val_score": val}
        if model_name == "RETRO-Profile":
            out = fr.embed(model, [batch], device)
            stage2 = run / "stage2_b10"
            summary = json.loads((stage2 / "summary.json").read_text())
            ds = RetrievalDatastore.load(stage2 / "datastore")
            split = {"speaker_ids": ["clip"], "query": out["query"]}
            for t, head in model.heads.items():
                head.cpu()
                head.residual.load_state_dict(torch.load(stage2 / f"residual_{t}.pt", map_location="cpu"))
                k = int(summary["targets"][t]["k"])
                with torch.no_grad():
                    pred = fr.residual_predict(head, out["z"][t], fr.knn_estimate(ds, split, t, k))
                row.update({t: float(pred[0]), f"{t}_kNN_off": float(out["direct"][t][0]), f"{t}_k": k})
            sims, idx = ds.search(out["query"], args.neighbours)
            for rank, (sim, i) in enumerate(zip(sims[0].tolist(), idx[0].tolist(), strict=True), start=1):
                e = ds._labels_df.iloc[i]
                meta = timit.loc[str(e["speaker_id"])]
                nbrs.append({"rank": rank, "speaker_id": e["speaker_id"], "similarity": sim, "age": e["age"],
                             "height": e["height"], "gender": meta["gender"],
                             "dialect_region": meta.get("dialect_region")})
        else:
            with torch.no_grad():
                pred = model(batch)
            row.update({t: float(pred[t][0]) for t in ("age", "height")})
        rows.append(row)
        model.cpu()
        print(f"{model_name:20s} ({run.name}, val {val:.4f}): age {row['age']:5.1f} yr, height {row['height']:5.1f} cm")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    nb = pd.DataFrame(nbrs)
    nb.to_csv(args.out.with_name(args.out.stem + "_neighbours.csv"), index=False)
    r = df[df.model == "RETRO-Profile"].iloc[0]
    print(f"RETRO kNN off: age {r.age_kNN_off:5.1f} yr, height {r.height_kNN_off:5.1f} cm")
    print(f"\nNearest TIMIT training speakers (RETRO's lookup):\n{nb.to_string(index=False)}")
    print(f"Wrote {args.out} and its _neighbours.csv")


if __name__ == "__main__":
    main()
