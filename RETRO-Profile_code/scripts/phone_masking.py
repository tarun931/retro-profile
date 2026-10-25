"""Phone-class masking on the TIMIT test set (2026-10-18): how much each Table 1 model's error rises when one class of
phones is silenced -- Gupta et al. 2022's Table 4 analysis, run for all four Table 1 models (4 x 5 seeds), no retraining.

TIMIT's time-aligned phone labels (<utt>.PHN beside each <utt>.WAV.wav; start / end in 16 kHz samples) mark every phone.
For one class at a time -- TIMIT's own classes (PHONCODE.DOC), stop closures counted with Stops -- every sample of those
phones is set to 0 (silence, as Gupta et al.) in all 1,680 test clips. Training data, checkpoints and RETRO's lookup
table are untouched; only the test audio changes. Each model is read exactly as in Table 1:

  RETRO-Profile  its second stage (kNN + residual: the run's lookup table, k per trait and residual weights), groups of
                 all 10 clips; the kNN-off reading is written too;
  WavLM + MLP    groups of all 10 clips;
  Gupta, Jaid    one prediction per clip, averaged per speaker.

The condition 'none' re-scores the unmasked audio first and must reproduce the stored Table 1 predictions, or the script
stops. TIMIT test clips are all shorter than 8 s, so nothing is cropped at random.

Writes <out>/predictions_<model>_s<seed>.csv (condition, speaker_id, target, prediction, ground_truth[, kNN-off]) and
<out>/masked_fraction.csv (share of test audio silenced per class).

    python scripts/phone_masking.py --out runs/phone_masking_table1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from retro_profile.data.collate import make_collate_fn
from retro_profile.data.timit import TimitDataset
from retro_profile.retrieval.datastore import RetrievalDatastore
from retro_profile.utils.audio import load_audio

_SCRIPTS = Path(__file__).resolve().parent
TARGETS = ["age", "height"]
SEEDS = (42, 1, 2, 3, 4)
# model -> (experiment, how Table 1 reads it, the stored Table 1 predictions the 'none' condition must reproduce)
TABLE1 = {
    "RETRO-Profile": ("retro_ordcon_fixed_l100_s5_b10", "stage2", "stage2_b10/predictions_test_knn_residual.csv"),
    "WavLM + MLP": ("wavlm_direct_budget_b10_e100", "bag", "eval_spk/predictions_timit.csv"),
    "Gupta 2022": ("gupta2022_finetune_wavlm_lr3e6", "clip_mean", "eval_utt/predictions_timit_speaker.csv"),
    "Jaid 2023": ("jaid2023_lr3e3", "clip_mean", "eval_utt/predictions_timit_speaker.csv"),
    # Table 1 reference row (2026-10-19): Gupta as published, with its paper encoder wav2vec 2.0
    "Gupta 2022 (wav2vec 2.0)": ("gupta2022_finetune_w2v2_lr3e6", "clip_mean", "eval_utt/predictions_timit_speaker.csv"),
}
# TIMIT PHONCODE.DOC classes; the stop closures (bcl ... kcl) are the closure part of stops.
PHONE_CLASSES = {
    "vowels": {"iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax",
               "ix", "axr", "ax-h"},
    "stops": {"b", "d", "g", "p", "t", "k", "dx", "q", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl"},
    "affricates": {"jh", "ch"},
    "fricatives": {"s", "sh", "z", "zh", "f", "th", "v", "dh"},
    "nasals": {"m", "n", "ng", "em", "en", "eng", "nx"},
    "semivowels_glides": {"l", "r", "w", "y", "hh", "hv", "el"},
    "others": {"pau", "epi", "h#"},
}
CONDITIONS = ["none", *PHONE_CLASSES]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def phn_path(audio_path: str | Path) -> Path:
    """<utt>.PHN beside <utt>.WAV.wav (Kaggle layout) or <utt>.WAV / <utt>.wav."""
    p = Path(audio_path)
    for suffix in (".WAV.wav", ".wav", ".WAV"):
        if p.name.endswith(suffix):
            return p.with_name(p.name[: -len(suffix)] + ".PHN")
    raise ValueError(f"not a TIMIT audio path: {audio_path}")


def read_phn(path: Path) -> list[tuple[int, int, str]]:
    """(start, end, phone) per line; start / end are sample indices at 16 kHz."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            s, e, ph = line.split()
            rows.append((int(s), int(e), ph))
    known = set().union(*PHONE_CLASSES.values())
    unknown = {ph for _, _, ph in rows} - known
    if unknown:
        raise ValueError(f"{path}: phones outside every class: {sorted(unknown)}")
    return rows


def mask_waveform(wav: torch.Tensor, segments, phones: set[str]) -> torch.Tensor:
    """A copy of ``wav`` with every sample of the given phones set to 0."""
    out = wav.clone()
    for s, e, ph in segments:
        if ph in phones:
            out[..., s:e] = 0.0
    return out


class MaskedTimitDataset(TimitDataset):
    """TIMIT with one phone class silenced in every clip (``mask=None``: the unmasked audio)."""

    def __init__(self, *args, mask: str | None = None, **kwargs) -> None:
        self.mask_phones = PHONE_CLASSES[mask] if mask else None
        super().__init__(*args, **kwargs)

    def _load_and_trim(self, path):
        wav = load_audio(path, target_sr=self.sample_rate)
        if wav.shape[-1] > self.max_samples:              # never on TIMIT test; a random crop would break the check
            raise SystemExit(f"{path} is longer than max_audio_seconds -- it would be cropped at random. Stop.")
        if self.mask_phones:
            if self.sample_rate != 16_000:
                raise SystemExit("TIMIT phone times are 16 kHz sample indices; sample_rate must be 16000")
            wav = mask_waveform(wav, read_phn(phn_path(path)), self.mask_phones)
        return wav


def test_loader(data_cfg: dict, mode: str, mask: str | None, num_workers: int) -> DataLoader:
    """The test loader exactly as evaluate.py / fit_residual.py build it (manifest order, same batch size)."""
    ds = MaskedTimitDataset(root=data_cfg["root"], split="test", mode=mode, transforms=None, targets=TARGETS,
                            sample_rate=data_cfg["sample_rate"], max_audio_seconds=data_cfg["max_audio_seconds"],
                            min_audio_seconds=data_cfg["min_audio_seconds"], bag_size=10, mask=mask)
    return DataLoader(ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=num_workers,
                      collate_fn=make_collate_fn(TARGETS, mode=mode))


@torch.no_grad()
def predict(model, loader: DataLoader, device: str) -> pd.DataFrame:
    """Per-row (speaker or clip) predictions with ground truth."""
    rows = []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        for t in TARGETS:
            for sid, p, y in zip(batch.speaker_ids, preds[t].float().cpu().tolist(),
                                 batch.labels[t].float().cpu().tolist(), strict=True):
                rows.append({"speaker_id": str(sid), "target": t, "prediction": p, "ground_truth": y})
    return pd.DataFrame(rows)


def stage2_reading(fr, model, run: Path, loader: DataLoader, device: str) -> pd.DataFrame:
    """RETRO's Table 1 reading on this loader: kNN + residual from the run's second stage, plus kNN off."""
    out = fr.embed(model, loader, device)
    stage2 = run / "stage2_b10"
    summary = json.loads((stage2 / "summary.json").read_text())
    ds = RetrievalDatastore.load(stage2 / "datastore")
    split = {"speaker_ids": out["speaker_ids"], "query": out["query"]}
    frames = []
    for t, head in model.heads.items():
        head.cpu()
        head.residual.load_state_dict(torch.load(stage2 / f"residual_{t}.pt", map_location="cpu"))
        with torch.no_grad():
            pred = fr.residual_predict(head, out["z"][t], fr.knn_estimate(ds, split, t, int(summary["targets"][t]["k"])))
        frames.append(pd.DataFrame({"speaker_id": out["speaker_ids"], "target": t, "prediction": pred.numpy(),
                                    "ground_truth": out["truth"][t].numpy(), "kNN_off": out["direct"][t].numpy()}))
    model.to(device)
    return pd.concat(frames, ignore_index=True)


def check_reproduces(ours: pd.DataFrame, reference: Path, tol: float = 0.05) -> float:
    """Max |ours - stored Table 1 prediction| over every speaker and target; stop above ``tol``."""
    ref = pd.read_csv(reference, dtype={"speaker_id": str}).set_index(["speaker_id", "target"])["prediction"]
    o = ours.set_index(["speaker_id", "target"])["prediction"]
    if set(o.index) != set(ref.index):
        raise SystemExit(f"{reference}: different speakers / targets than ours. Stop.")
    diff = float((o - ref.loc[o.index]).abs().max())
    if not diff <= tol:
        raise SystemExit(f"unmasked predictions differ from {reference} by up to {diff:.4g} -- not the Table 1 model. Stop.")
    return diff


def masked_fraction(data_cfg: dict) -> pd.DataFrame:
    """Share of all test audio (samples) each class silences."""
    paths = pd.read_csv(Path(data_cfg["root"]) / "manifests" / "test.csv")["audio_path"]
    total, per = 0, dict.fromkeys(PHONE_CLASSES, 0)
    for p in paths:
        p = p if Path(p).is_absolute() else str(Path(data_cfg["root"]) / p)
        segs = read_phn(phn_path(p))
        total += max(e for _, e, _ in segs)
        for name, phones in PHONE_CLASSES.items():
            per[name] += sum(e - s for s, e, ph in segs if ph in phones)
    return pd.DataFrame([{"condition": k, "masked_fraction": v / total} for k, v in per.items()])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=list(TABLE1), choices=list(TABLE1))
    ap.add_argument("--only-exp", nargs="+", default=None, choices=[v[0] for v in TABLE1.values()],
                    help="score only these experiments (no spaces, for job scripts); other models' files are untouched")
    args = ap.parse_args(argv)
    if args.only_exp:
        args.models = [m for m in TABLE1 if TABLE1[m][0] in args.only_exp]

    fr = _script("fit_residual")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = fr.corpus_data_cfg("timit", 10, args.num_workers)
    args.out.mkdir(parents=True, exist_ok=True)
    masked_fraction(data_cfg).to_csv(args.out / "masked_fraction.csv", index=False)
    for name in args.models:
        exp, reading, ref = TABLE1[name]
        for seed in SEEDS:
            run = Path("runs") / f"{exp}_timit_s{seed}"
            model, ckpt = fr.load_frozen_model(run)
            model.to(device)
            frames = []
            for cond in CONDITIONS:
                mask = None if cond == "none" else cond
                if reading == "stage2":
                    df = stage2_reading(fr, model, run, test_loader(data_cfg, "bag", mask, args.num_workers), device)
                elif reading == "bag":
                    df = predict(model, test_loader(data_cfg, "bag", mask, args.num_workers), device)
                else:                                      # per clip, then the speaker mean (evaluate.py --unit utterance)
                    df = predict(model, test_loader(data_cfg, "utterance", mask, args.num_workers), device)
                    df = df.groupby(["speaker_id", "target"], as_index=False).agg(
                        prediction=("prediction", "mean"), ground_truth=("ground_truth", "first"))
                if cond == "none":
                    diff = check_reproduces(df, run / ref)
                    print(f"{name} s{seed} ({ckpt.name}): unmasked reproduces Table 1 (max diff {diff:.2g})")
                frames.append(df.assign(condition=cond))
            pd.concat(frames, ignore_index=True).to_csv(args.out / f"predictions_{exp}_s{seed}.csv", index=False)
            model.cpu()
            print(f"{name} s{seed}: wrote {len(CONDITIONS)} conditions")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
