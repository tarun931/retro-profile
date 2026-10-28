"""Rebuild the paper's result tables from the saved per-speaker predictions in runs/.

Part 2: Table 2 (TIMIT), Table 3 (ablations), Table 4 (neighbour inspection), Table 5 (transfer to HeightCeleb).
Appendix: A2-A3 (paired differences), A4 (RETRO-Profile per seed), A5 (transfer before calibration), A6 (what the
84 HeightCeleb speakers changed), A11 (neighbour quality), A12 (neighbour examples).

Inputs are the files the run scripts write (scripts/reproduce/*.sh): evaluate.py's predictions, stage 2's
``stage2_b10/`` (fit_residual.py), calibrate_predictions.py's ``eval_spk_cal/``, neighbour_quality.py's and
neighbour_inspection.py's JSON files, and the data manifests. Nothing is trained or re-scored here.

Conventions (as in the paper):
  * MAE and RMSE per seed for men, women and all speakers; a cell reads "men / women / all +- SD", where men and
    women are the mean over seeds and SD is the sample SD of the all-speaker value over seeds.
  * Paired differences: each speaker's absolute error is averaged over the seeds of a model, then the difference
    of the two models is bootstrapped over speakers (10,000 resamples, seed 0; scripts/paired_bootstrap.py).
    * = the 95 % interval, at the printed precision, excludes zero.
  * Age is scored only for speakers with a known age (HeightCeleb test: 118 of 249).

    python scripts/make_tables.py                     # prints every table it has the runs for
    python scripts/make_tables.py --out tables.md     # also writes them to a file

A table is skipped, with the missing file named, when any run it needs is absent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("paired_bootstrap", _SCRIPTS / "paired_bootstrap.py")
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

TARGETS = ("age", "height")
SEEDS_TIMIT = (42, 1, 2, 3, 4)
SEEDS_TRANSFER = (42, 1, 2)
HC = "heightceleb_split2"

# Table 2 rows: name -> (configuration, prediction file inside the run folder)
TIMIT = {
    "WavLM + MLP": ("wavlm_direct_budget_b10_e100", "eval_spk/predictions_timit.csv"),
    "Gupta et al. (WavLM Base+)": ("gupta2022_finetune_wavlm_lr3e6", "eval_utt/predictions_timit_speaker.csv"),
    "Gupta et al. as published (wav2vec 2.0)": ("gupta2022_finetune_w2v2_lr3e6",
                                                "eval_utt/predictions_timit_speaker.csv"),
    "Jaid et al.": ("jaid2023_lr3e3", "eval_utt/predictions_timit_speaker.csv"),
    "RETRO-Profile": ("retro_ordcon_fixed_l100_s5_b10", "stage2_b10/predictions_test_knn_residual.csv"),
}
RETRO = "retro_ordcon_fixed_l100_s5_b10"
# Table 5 rows: name -> (configuration suffix after timit_nisp[_hcs]_, file); RETRO's "{cal}" is "_cal" or ""
TRANSFER = {
    "WavLM + MLP": ("mlp_b10_e100", "eval_spk{cal}/predictions_heightceleb_split2.csv"),
    "Gupta et al. (WavLM Base+)": ("gupta_lr3e6", "eval_spk{cal}/predictions_heightceleb_split2.csv"),
    "Gupta et al. as published (wav2vec 2.0)": ("gupta_w2v2_lr3e6",
                                                "eval_spk{cal}/predictions_heightceleb_split2.csv"),
    "Jaid et al.": ("jaid_lr3e3", "eval_spk{cal}/predictions_heightceleb_split2.csv"),
    "RETRO-Profile": ("b10_l100_s5", "stage2_b10/predictions_test_knn_residual{cal}.csv"),
    "RETRO-Profile, kNN off": ("b10_l100_s5", "stage2_b10/predictions_test_direct{cal}.csv"),
}
VERSIONS = {"4a": "timit_nisp", "4b": "timit_nisp_hcs"}


class MissingError(Exception):
    """A file a table needs is not there."""


# ----------------------------------------------------------------------------------------------- loading and maths
def load(path: Path, target: str) -> pd.DataFrame:
    """One row per speaker with a known label: prediction, ground_truth, gender (index speaker_id)."""
    if not path.exists():
        raise MissingError(path)
    df = pd.read_csv(path)
    df = df[(df["target"] == target) & df["ground_truth"].notna()]
    if df["speaker_id"].duplicated().any():
        raise ValueError(f"{path}: more than one {target} row per speaker")
    return df.set_index("speaker_id")[["prediction", "ground_truth", "gender"]]


def scores(df: pd.DataFrame) -> dict[str, float]:
    """MAE and RMSE for men, women and all speakers."""
    out = {}
    for g, part in (("M", df[df.gender == "M"]), ("F", df[df.gender == "F"]), ("all", df)):
        e = part["prediction"] - part["ground_truth"]
        out[f"mae_{g}"], out[f"rmse_{g}"] = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
    return out


def cell(per_seed: list[dict[str, float]], metric: str) -> str:
    """'men / women / all +- SD' over seeds."""
    m = np.mean([s[f"{metric}_M"] for s in per_seed])
    f = np.mean([s[f"{metric}_F"] for s in per_seed])
    a = [s[f"{metric}_all"] for s in per_seed]
    return f"{m:.2f} / {f:.2f} / {np.mean(a):.2f} ± {np.std(a, ddof=1):.2f}"


def all_cell(per_seed: list[dict[str, float]], metric: str) -> str:
    a = [s[f"{metric}_all"] for s in per_seed]
    return f"{np.mean(a):.2f} ± {np.std(a, ddof=1):.2f}"


def ref_cells(test: pd.DataFrame, means: pd.Series) -> dict[str, float]:
    """Scores of predicting each speaker as their gender's mean (the no-audio reference)."""
    return scores(test.assign(prediction=test["gender"].map(means)))


def mean_error(frames: list[pd.DataFrame]) -> pd.Series:
    """Each speaker's absolute error, averaged over runs (seeds)."""
    errs = pd.concat([(f["prediction"] - f["ground_truth"]).abs() for f in frames], axis=1)
    if errs.isna().any().any():
        raise ValueError("runs were scored on different speakers")
    return errs.mean(axis=1)


def diff(a: list[pd.DataFrame], b: list[pd.DataFrame]) -> str:
    """Paired 'A minus B' in MAE with its 95 % bootstrap interval; * when the printed interval excludes zero."""
    r = _pb.paired_bootstrap(mean_error(a), mean_error(b), n_boot=10_000, seed=0)
    d, lo, hi = (round(r[k], 2) for k in ("diff", "ci_lo", "ci_hi"))
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def md(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def speakers(manifest: Path) -> pd.DataFrame:
    """One row per speaker: gender and the speaker's age / height (mean over clips where they vary)."""
    if not manifest.exists():
        raise MissingError(manifest)
    df = pd.read_csv(manifest)
    return df.groupby("speaker_id").agg(gender=("gender", "first"), age=("age", "mean"), height=("height", "mean"))


# -------------------------------------------------------------------------------------------------------- tables
class Tables:
    def __init__(self, runs: Path, data: Path):
        self.runs, self.data = runs, data

    def timit(self, cfg: str, file: str, target: str) -> list[pd.DataFrame]:
        return [load(self.runs / f"{cfg}_timit_s{s}" / file, target) for s in SEEDS_TIMIT]

    def transfer(self, version: str, model: str, target: str, cal: bool) -> list[pd.DataFrame]:
        suffix, file = TRANSFER[model]
        cfg = f"{VERSIONS[version]}_{suffix}"
        return [load(self.runs / f"{cfg}_{HC}_s{s}" / file.format(cal="_cal" if cal else ""), target)
                for s in SEEDS_TRANSFER]

    # ---- Table 2 and A2
    def table2(self) -> str:
        rows = []
        for name, (cfg, file) in TIMIT.items():
            per = {t: [scores(f) for f in self.timit(cfg, file, t)] for t in TARGETS}
            rows.append([name, cell(per["age"], "mae"), cell(per["age"], "rmse"),
                         cell(per["height"], "mae"), cell(per["height"], "rmse")])
        train = speakers(self.data / "timit/manifests/train.csv")
        test = speakers(self.data / "timit/manifests/test.csv")
        ref = {t: ref_cells(test.rename(columns={t: "ground_truth"}).dropna(subset=["ground_truth"]),
                            train.groupby("gender")[t].mean()) for t in TARGETS}
        rows.append(["No-audio reference (gender mean, training)"] + [
            f"{ref[t][f'{m}_M']:.2f} / {ref[t][f'{m}_F']:.2f} / {ref[t][f'{m}_all']:.2f}"
            for t in TARGETS for m in ("mae", "rmse")])
        return "### Table 2. TIMIT test (168 speakers), five seeds\n\n" + md(
            ["Model", "Age MAE (M / F / All)", "Age RMSE (M / F / All)", "Height MAE (M / F / All)",
             "Height RMSE (M / F / All)"], rows)

    def table_a2(self) -> str:
        p = {n: {t: self.timit(c, f, t) for t in TARGETS} for n, (c, f) in TIMIT.items()}
        pairs = [("RETRO-Profile", "Gupta et al. (WavLM Base+)"),
                 ("RETRO-Profile", "Gupta et al. as published (wav2vec 2.0)"),
                 ("RETRO-Profile", "Jaid et al."), ("RETRO-Profile", "WavLM + MLP"),
                 ("Gupta et al. as published (wav2vec 2.0)", "Gupta et al. (WavLM Base+)")]
        rows = [[f"{a} minus {b}"] + [diff(p[a][t], p[b][t]) for t in TARGETS] for a, b in pairs]
        return "### Table A2. TIMIT test, paired differences\n\n" + md(
            ["Comparison", "Age MAE [95% CI]", "Height MAE [95% CI]"], rows)

    # ---- Table 3 and A4
    def table3(self) -> str:
        stage2 = "stage2_b10/predictions_test_{}.csv"
        variants = [
            ("Full RETRO-Profile (kNN + residual)", RETRO, stage2.format("knn_residual")),
            ("- MIL: averaged clips (RQ2)", f"{RETRO}_nomil", stage2.format("knn_residual")),
            ("- retrieval: direct regression only, kNN off (RQ3)", RETRO, stage2.format("direct")),
            ("- residual: kNN alone", RETRO, stage2.format("knn")),
            ("- OrdCon (RQ4)", f"{RETRO}_noordcon", stage2.format("knn_residual")),
            ("- all three parts (= WavLM + MLP, Table 2)", *TIMIT["WavLM + MLP"]),
        ]
        full = {t: self.timit(RETRO, stage2.format("knn_residual"), t) for t in TARGETS}
        rows = []
        for name, cfg, file in variants:
            fr = {t: self.timit(cfg, file, t) for t in TARGETS}
            per = {t: [scores(f) for f in fr[t]] for t in TARGETS}
            change = "n/a" if name.startswith("Full") else " / ".join(diff(fr[t], full[t]) for t in TARGETS)
            rows.append([name, all_cell(per["age"], "mae"), all_cell(per["age"], "rmse"),
                         all_cell(per["height"], "mae"), all_cell(per["height"], "rmse"), change])
        return "### Table 3. Ablations on TIMIT test, five seeds (change = ablated minus full)\n\n" + md(
            ["Model", "Age MAE", "Age RMSE", "Height MAE", "Height RMSE",
             "Change vs full: age / height [95% CI]"], rows)

    def table_a4(self) -> str:
        rows = []
        for s in SEEDS_TIMIT:
            run = self.runs / f"{RETRO}_timit_s{s}" / "stage2_b10"
            if not (run / "summary.json").exists():
                raise MissingError(run / "summary.json")
            summ = json.loads((run / "summary.json").read_text())
            m = re.search(r"epoch=(\d+)-.*val=([0-9.]+)\.ckpt", summ["checkpoint"])
            kept = [t for t in TARGETS if summ["targets"][t]["residual_kept"]]
            mae = {r: " / ".join(f"{scores(load(run / f'predictions_test_{r}.csv', t))['mae_all']:.2f}"
                                 for t in TARGETS) for r in ("knn_residual", "knn", "direct")}
            rows.append([str(s), f"{m.group(1)} ({m.group(2)})",
                         f"{summ['targets']['age']['k']} / {summ['targets']['height']['k']}",
                         ", ".join(kept) or "none", mae["knn_residual"], mae["knn"], mae["direct"]])
        return "### Table A4. RETRO-Profile per seed, TIMIT test (MAE: age / height)\n\n" + md(
            ["Seed", "Epoch (validation score)", "k: age / height", "Residual kept", "Full", "kNN alone",
             "kNN off"], rows)

    # ---- Table 5, A3, A5, A6
    def _val_reference(self, version: str, target: str) -> pd.Series:
        """Gender means of the 251 HeightCeleb validation speakers' labels (stage 2's validation file)."""
        path = self.runs / f"{VERSIONS[version]}_b10_l100_s5_{HC}_s42" / "stage2_b10" / "predictions_val_direct.csv"
        return load(path, target).groupby("gender")["ground_truth"].mean()

    def _train_reference(self, version: str, target: str) -> pd.Series:
        """Gender means of the training speakers (TIMIT train + NISP [+ the 84-speaker sample])."""
        parts = [speakers(self.data / "timit/manifests/train.csv"),
                 speakers(self.data / "nisp/manifests/train.csv"), speakers(self.data / "nisp/manifests/test.csv")]
        if version == "4b":
            parts.append(speakers(self.data / f"{HC}_age_sample/manifests/train.csv"))
        return pd.concat(parts).groupby("gender")[target].mean()

    def _transfer_rows(self, version: str, cal: bool) -> tuple[list[list[str]], list[list[str]]]:
        mae_rows, rmse_rows = [], []
        for name in list(TRANSFER)[:5]:
            per = {t: [scores(f) for f in self.transfer(version, name, t, cal)] for t in TARGETS}
            mae_rows.append([name, cell(per["age"], "mae"), cell(per["height"], "mae")])
            rmse_rows.append([name, cell(per["age"], "rmse"), cell(per["height"], "rmse")])
        test = {t: self.transfer(version, "RETRO-Profile", t, cal)[0] for t in TARGETS}
        means = {t: (self._val_reference(version, t) if cal else self._train_reference(version, t)) for t in TARGETS}
        ref = {t: ref_cells(test[t], means[t]) for t in TARGETS}
        label = "No-audio reference (gender mean, " + ("HeightCeleb validation)" if cal else "training speakers)")
        fmt = lambda t, m: f"{ref[t][f'{m}_M']:.2f} / {ref[t][f'{m}_F']:.2f} / {ref[t][f'{m}_all']:.2f}"  # noqa: E731
        mae_rows.append([label, fmt("age", "mae"), fmt("height", "mae")])
        rmse_rows.append([label, fmt("age", "rmse"), fmt("height", "rmse")])
        return mae_rows, rmse_rows

    def table5(self) -> str:
        out = ["### Table 5. Transfer to HeightCeleb test (249 speakers; 118 with an age), three seeds, "
               "after calibration"]
        for v in VERSIONS:
            mae_rows, rmse_rows = self._transfer_rows(v, cal=True)
            out.append(f"\n**{v}** MAE\n\n" + md(["Model", "Age MAE (M / F / All)", "Height MAE (M / F / All)"],
                                                 mae_rows))
            out.append(f"\n**{v}** RMSE\n\n" + md(["Model", "Age RMSE (M / F / All)", "Height RMSE (M / F / All)"],
                                                  rmse_rows))
        knn_off = {v: {t: all_cell([scores(f) for f in self.transfer(v, "RETRO-Profile, kNN off", t, True)], "mae")
                       for t in TARGETS} for v in VERSIONS}
        out.append("\nRETRO-Profile with kNN off: age " + " / ".join(knn_off[v]["age"].split(" ±")[0] for v in VERSIONS)
                   + " years, height " + " / ".join(knn_off[v]["height"].split(" ±")[0] for v in VERSIONS)
                   + " cm (4a / 4b)")
        return "\n".join(out)

    def table_a3(self) -> str:
        rows = []
        for other in ["WavLM + MLP", "Gupta et al. (WavLM Base+)", "Gupta et al. as published (wav2vec 2.0)",
                      "Jaid et al.", "RETRO-Profile, kNN off"]:
            row = [other]
            for v in VERSIONS:
                for t in TARGETS:
                    row.append(diff(self.transfer(v, "RETRO-Profile", t, True), self.transfer(v, other, t, True)))
            rows.append(row)
        return "### Table A3. Transfer, paired differences, RETRO-Profile minus each model (calibrated)\n\n" + md(
            ["RETRO-Profile minus ...", "4a age", "4a height", "4b age", "4b height"], rows)

    def table_a5(self) -> str:
        out = ["### Table A5. Transfer to HeightCeleb test before calibration, three seeds"]
        for v in VERSIONS:
            mae_rows, rmse_rows = self._transfer_rows(v, cal=False)
            rows = [[m[0], m[1], r[1], m[2], r[2]] for m, r in zip(mae_rows, rmse_rows, strict=True)]
            out.append(f"\n**{v}**\n\n" + md(["Model", "Age MAE", "Age RMSE", "Height MAE", "Height RMSE"], rows))
        return "\n".join(out)

    def table_a6(self) -> str:
        rows = []
        for name in ["WavLM + MLP", "Gupta et al. (WavLM Base+)", "Gupta et al. as published (wav2vec 2.0)",
                     "Jaid et al.", "RETRO-Profile", "RETRO-Profile, kNN off"]:
            row = [name]
            for t, cal in (("age", True), ("height", True), ("age", False)):
                a, b = self.transfer("4a", name, t, cal), self.transfer("4b", name, t, cal)
                ma = np.mean([scores(f)["mae_all"] for f in a])
                mb = np.mean([scores(f)["mae_all"] for f in b])
                row.append(f"{ma:.2f} -> {mb:.2f} ({diff(b, a)})")
            rows.append(row)
        return "### Table A6. What the 84 HeightCeleb speakers changed (4b minus 4a, paired)\n\n" + md(
            ["Model", "Age MAE calibrated: 4a -> 4b (change [95% CI])", "Height MAE calibrated: 4a -> 4b (change)",
             "Age MAE raw: 4a -> 4b (change [95% CI])"], rows)

    # ---- Table 4, A11, A12
    def _inspection(self) -> dict:
        path = self.runs / f"{RETRO}_timit_s42" / "neighbour_inspection.json"
        if not path.exists():
            raise MissingError(path)
        return json.loads(path.read_text())

    def table4(self) -> str:
        d = self._inspection()
        k3, k10, c = d["k3"], d["k10"], d["confidence"]
        rows = [
            ["Same gender", f"{100 * k3['same_gender']:.1f} %", f"{100 * k3['gender_chance']:.1f} % (chance)",
             diff_json(k3["same_gender_minus_chance"], 100, " pts")],
            ["Age gap (years)", f"{k3['age_gap']:.2f}",
             f"{k3['age_gap_same_gender_random']:.2f} (random same-gender speaker)",
             diff_json(k3["age_gap_minus_random"], 1, "", nd=2)],
            ["Same dialect region (accent)", f"{100 * k3['same_dialect']:.1f} %",
             f"{100 * k3['dialect_chance_same_gender']:.1f} % (same-gender chance)",
             diff_json(k3["same_dialect_minus_chance"], 100, " pts")],
            ["Same dialect region, k=10", f"{100 * k10['same_dialect']:.1f} %",
             f"{100 * k10['dialect_chance_same_gender']:.1f} % (same-gender chance)",
             diff_json(k10["same_dialect_minus_chance"], 100, " pts")],
            ["Effective number of neighbours (age)", f"{d['effective_neighbours']['age']:.2f} of {d['k_age']}",
             f"{d['k_age']} (equal weights)", "weights almost equal"],
            ["Nearest-neighbour similarity vs age error",
             f"Spearman {c['spearman_nearest_sim_vs_abs_error']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]",
             "0 (no relation)", "lower similarity, larger error"],
        ]
        return (f"### Table 4. Neighbour inspection, RETRO-Profile seed 42, k={d['k_age']} age neighbours, "
                f"{d['n_test']} TIMIT test speakers\n\n" + md(["Measure", "Neighbours", "Baseline",
                                                               "Difference [95% CI]"], rows))

    def table_a12(self) -> str:
        rows = []
        for label, ex in self._inspection()["examples"].items():
            nb = "; ".join(f"{n['speaker']} ({n['gender']}, {n['dialect_region']}, {n['age']:.1f}, cos "
                           f"{n['similarity']:.3f})" for n in ex["neighbours"])
            rows.append([f"{label.capitalize()}: {ex['speaker']} ({ex['gender']}, {ex['dialect_region']}), age "
                         f"{ex['age']:.1f}", f"{ex['knn_age']:.1f}", f"{ex['abs_error']:.1f}", nb])
        return "### Table A12. Age neighbours behind three kNN estimates, seed 42\n\n" + md(
            ["Test speaker", "kNN age", "Error", "Neighbours"], rows)

    def table_a11(self) -> str:
        q = {}
        for v in ("", "_noordcon"):
            q[v] = []
            for s in SEEDS_TIMIT:
                path = self.runs / f"{RETRO}{v}_timit_s{s}" / "stage2_b10" / "neighbour_quality.json"
                if not path.exists():
                    raise MissingError(path)
                q[v].append(json.loads(path.read_text())["targets"])
        train = speakers(self.data / "timit/manifests/train.csv")
        test = speakers(self.data / "timit/manifests/test.csv")
        ref_any = {t: _mean_pair_gap(test[t], train[t]) for t in TARGETS}
        ref_same = _mean_pair_gap(test["height"], train["height"], test["gender"], train["gender"])
        gender = {t: ref_cells(test.rename(columns={t: "ground_truth"}).dropna(subset=["ground_truth"]),
                               train.groupby("gender")[t].mean())["mae_all"] for t in TARGETS}

        def ms(v: str, t: str, key: str, k: str | None = None) -> str:
            x = [r[t][key][k] if k else r[t][key] for r in q[v]]
            sign = "+" if key == "distance_spearman" else ""
            return f"{np.mean(x):{sign}.2f} ± {np.std(x, ddof=1):.2f}"

        rows = []
        for t, unit_ref in (("age", f"{ref_any['age']:.2f} (random training speaker)"),
                            ("height", f"{ref_same:.2f} same gender ({ref_any['height']:.2f} any)")):
            for i, k in enumerate(("1", "5", "20")):
                label = {"1": "nearest neighbour", "5": "5 nearest", "20": "20 nearest"}[k]
                ref = unit_ref if i == 0 else (f"{ref_any['age']:.2f}" if t == "age"
                                               else f"{ref_same:.2f} ({ref_any['height']:.2f})")
                rows.append([f"{t.capitalize()}: label gap, {label}", ms("", t, "gap_at_k", k),
                             ms("_noordcon", t, "gap_at_k", k), ref])
            rows.append([f"{t.capitalize()}: rank correlation, distance vs label gap", ms("", t, "distance_spearman"),
                         ms("_noordcon", t, "distance_spearman"), "0 (no ordering)"])
            rows.append([f"{t.capitalize()}: kNN-only test MAE (k chosen on validation)", ms("", t, "knn_only_mae"),
                         ms("_noordcon", t, "knn_only_mae"), f"{gender[t]:.2f} (gender average)"])
        return "### Table A11. Neighbour quality on TIMIT test, five seeds (mean ± SD)\n\n" + md(
            ["Trait / measure", "With OrdCon (full RETRO)", "Without OrdCon", "Reference (no audio)"], rows)


def diff_json(d: dict, scale: float, unit: str, nd: int = 1) -> str:
    """A difference stored by neighbour_inspection.py, printed like diff() (nd decimals)."""
    m, lo, hi = (round(x * scale, nd) for x in (d["mean"], d["ci"][0], d["ci"][1]))
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{m:+.{nd}f}{unit} [{lo:+.{nd}f}, {hi:+.{nd}f}]{star}"


def _mean_pair_gap(test: pd.Series, train: pd.Series, g_test: pd.Series | None = None,
                   g_train: pd.Series | None = None) -> float:
    """Mean |test label - training label| over all test-training pairs (optionally same gender only)."""
    test, train = test.dropna(), train.dropna()
    gaps = []
    for spk, y in test.items():
        pool = train if g_test is None else train[g_train.loc[train.index] == g_test.loc[spk]]
        gaps.append(float((pool - y).abs().mean()))
    return float(np.mean(gaps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None, help="also write the tables to this Markdown file")
    args = ap.parse_args()
    t = Tables(args.runs, args.data)
    blocks = []
    for fn in (t.table2, t.table3, t.table4, t.table5, t.table_a2, t.table_a3, t.table_a4, t.table_a5, t.table_a6,
               t.table_a11, t.table_a12):
        try:
            blocks.append(fn())
        except MissingError as e:
            blocks.append(f"### {fn.__name__}: skipped, missing {e}")
    text = "\n\n".join(blocks) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
