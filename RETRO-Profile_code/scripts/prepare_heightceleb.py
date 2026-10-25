"""Prepare HeightCeleb (VoxCeleb1 audio + crowd-sourced heights + our ages) for retro-profile.

Inputs:
    --wav-root  VoxCeleb1 audio: <wav-root>/<VoxCeleb1 ID>/<YouTube video id>/<nnnnn>.wav
                (scripts/cluster/unpack_voxceleb1.slurm unpacks it to <data-root>/voxceleb1/wav)
    --heights   HeightCeleb's labels, heightceleb/height_celeb.csv (VoxCeleb1 ID, height, sex, split)
    --ages      our VoxCeleb1 ages, heightceleb/vox1_height_age.csv (heightceleb/build_vox1_ages.py:
                one dated video and the speaker's age in it, 573 speakers)

Output: <out>/manifests/{train,dev,test}.csv with columns

    speaker_id, utt_id, audio_path, video_id, age, height, gender

Splits: HeightCeleb's official TEST (= the VoxCeleb1 test set, 40 speakers) is ``test``. ``dev`` is
carved out of TRAIN -- speaker-disjoint, stratified by gender, by whether the speaker has an age, and
by height. An age belongs to ONE video (the year it was recorded): in dev and test a speaker with an
age keeps only that video's clips, so the labels that choose the checkpoint are exact; in train the
speaker keeps every clip under that one age (median 4 clips in the dated video -- too few for an
8-clip bag; other dated videos of the same speaker differ by <= 2 years for half of them, <= 3 for
75 %, agreed 2026-10-12). When TIMIT is the test set, test.csv is training data too (pooled
"train+test"): build it with ``--test-age-clips all``. Speakers without an age keep all clips with
age NaN, which the masked loss skips (CLAUDE.md detail 3). Never reads audio.

``--split-mode grouped`` (agreed 2026-10-12) ignores the official split -- whose test has only 40
speakers, 15 with an age -- and divides all 1,251 speakers ``--fractions`` (default 60/20/20) by
GROUP: speakers who appear in the same YouTube video (co-star interviews: same recording, and
VoxCeleb's automatic segmentation can file one person's voice under the other) or who share an
identical clip (``--md5-csv``, from scripts/cluster/vox1_duplicates.slurm) always land in the same split,
so no test speaker shares any recording with a training speaker (SESSION_HANDOFF §0.7.14). Groups
are balanced on whether they hold an aged speaker, gender and height.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
COLS = ["speaker_id", "utt_id", "audio_path", "video_id", "age", "height", "gender"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--wav-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--heights", type=Path, default=REPO / "heightceleb" / "height_celeb.csv")
    p.add_argument("--ages", type=Path, default=REPO / "heightceleb" / "vox1_height_age.csv")
    p.add_argument("--train-age-clips", choices=["all", "video"], default="all",
                   help="train: a speaker with an age keeps every clip under that age (all, default) "
                        "or only the dated video's clips (video).")
    p.add_argument("--dev-age-clips", choices=["video", "all"], default="video",
                   help="dev: the same choice; default video, so the ages that choose the "
                        "checkpoint are exact.")
    p.add_argument("--test-age-clips", choices=["video", "all"], default="video",
                   help="test: the same choice; 'all' when test.csv is pooled into training.")
    p.add_argument("--val-frac", type=float, default=0.1, help="official mode: dev share of TRAIN.")
    p.add_argument("--split-mode", choices=["official", "grouped"], default="official",
                   help="official: HeightCeleb's TRAIN/TEST (+ dev from TRAIN). grouped: all speakers, "
                        "by co-appearance group, into --fractions.")
    p.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                   metavar=("TRAIN", "DEV", "TEST"), help="grouped mode: speaker shares.")
    p.add_argument("--md5-csv", type=Path, default=None,
                   help="grouped mode: clip fingerprints (speaker_id, md5); speakers sharing one are grouped.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def scan_wavs(wav_root: Path) -> pd.DataFrame:
    """One row per clip from the <speaker>/<video>/<nnnnn>.wav tree."""
    rows = [{"speaker_id": w.parts[-3], "video_id": w.parts[-2],
             "utt_id": f"{w.parts[-2]}_{w.stem}", "audio_path": str(w.resolve())}
            for w in sorted(wav_root.expanduser().glob("*/*/*.wav"))]
    if not rows:
        raise SystemExit(f"No clips under {wav_root} (expected <speaker>/<video>/<nnnnn>.wav).")
    return pd.DataFrame(rows)


def load_labels(heights_path: Path, ages_path: Path) -> pd.DataFrame:
    """Per-speaker labels: height, gender, official split, and (573 speakers) age + its video."""
    h = pd.read_csv(heights_path).rename(columns={"VoxCeleb1 ID": "speaker_id", "sex": "gender"})
    need = {"speaker_id", "height", "gender", "split"}
    if not need <= set(h.columns):
        raise SystemExit(f"{heights_path} lacks columns {sorted(need - set(h.columns))}")
    h["gender"] = h["gender"].str.strip().str.upper()
    h["split"] = h["split"].str.strip().str.upper()
    if not set(h["gender"]) <= {"M", "F"} or not set(h["split"]) <= {"TRAIN", "TEST"}:
        raise SystemExit(f"{heights_path}: unexpected sex {set(h['gender'])} or split {set(h['split'])}")
    if h["speaker_id"].duplicated().any():
        raise SystemExit(f"{heights_path}: duplicated speakers")

    a = pd.read_csv(ages_path)[["speaker_id", "video_id", "age", "height", "sex"]]
    if a["speaker_id"].duplicated().any():
        raise SystemExit(f"{ages_path}: more than one age per speaker")
    both = a.merge(h, on="speaker_id", how="left", suffixes=("_ages", ""))
    bad = both[both["height"].isna() | (both["height_ages"] != both["height"])
               | (both["sex"].str.upper() != both["gender"])]
    if len(bad):
        raise SystemExit(f"{ages_path} disagrees with {heights_path} on {len(bad)} speakers, "
                         f"e.g. {bad['speaker_id'].head(3).tolist()}")
    a = a.rename(columns={"video_id": "age_video"})[["speaker_id", "age_video", "age"]]
    return h[["speaker_id", "height", "gender", "split"]].merge(a, on="speaker_id", how="left")


def split_speakers(labels: pd.DataFrame, val_frac: float, seed: int) -> dict[str, str]:
    """speaker -> train/dev/test. dev = every k-th TRAIN speaker up the height order, within each
    (gender, has-age) group, so dev matches train on all three; ``seed`` picks the phase."""
    step = max(2, round(1 / val_frac))
    out = {s: "test" for s in labels.loc[labels["split"] == "TEST", "speaker_id"]}
    train = labels[labels["split"] == "TRAIN"].assign(has_age=lambda d: d["age"].notna())
    for _, g in train.groupby(["gender", "has_age"]):
        ranked = g.sort_values(["height", "speaker_id"])["speaker_id"].tolist()
        dev = set(ranked[seed % step::step])
        out.update({s: "dev" if s in dev else "train" for s in ranked})
    return out


def speaker_groups(clips: pd.DataFrame, md5: pd.DataFrame | None = None) -> dict[str, str]:
    """speaker -> group: speakers linked by a shared video folder (a YouTube ID) or, with ``md5``, a
    shared clip fingerprint, joined transitively (union-find). Named by the group's first speaker."""
    parent: dict[str, str] = {s: s for s in clips["speaker_id"].unique()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = [clips[["video_id", "speaker_id"]].drop_duplicates().groupby("video_id")["speaker_id"]]
    if md5 is not None:
        links.append(md5[md5["speaker_id"].isin(parent)].groupby("md5")["speaker_id"])
    for grouped in links:
        for spks in grouped.agg(lambda s: sorted(set(s))):
            for other in spks[1:]:
                a, b = find(spks[0]), find(other)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return {s: find(s) for s in parent}


def grouped_split(labels: pd.DataFrame, groups: dict[str, str], fractions: list[float],
                  seed: int) -> dict[str, str]:
    """speaker -> train/dev/test, whole groups at a time. Within each stratum (holds an aged speaker x
    majority gender) multi-speaker groups go first in seeded random order, then single speakers up the
    height order (shuffled within blocks of 5, so the seed varies the split while height stays spread);
    each goes to the split furthest below its target share."""
    names = ("train", "dev", "test")
    target = np.asarray(fractions, dtype=float) / sum(fractions)
    units = labels.assign(group=labels["speaker_id"].map(groups)).groupby("group").agg(
        speakers=("speaker_id", list), n=("speaker_id", "size"),
        has_age=("age", lambda a: bool(a.notna().any())),
        male=("gender", lambda g: bool((g == "M").mean() >= 0.5)), height=("height", "mean"))
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for _, g in units.groupby(["has_age", "male"]):
        multi = list(g.index[g["n"] > 1])
        rng.shuffle(multi)
        single = list(g[g["n"] == 1].sort_values(["height"]).index)
        order = multi
        for i in range(0, len(single), 5):
            block = single[i:i + 5]
            rng.shuffle(block)
            order += block
        counts = np.zeros(len(names))
        for u in order:
            n = g.at[u, "n"]
            k = int(np.argmax(target * (counts.sum() + n) - counts))
            counts[k] += n
            out.update({s: names[k] for s in g.at[u, "speakers"]})
    return out


def build_manifest(clips: pd.DataFrame, labels: pd.DataFrame, split_of: dict[str, str],
                   age_clips: dict[str, str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Join clips to labels and splits; per split, a speaker with an age keeps "all" clips or only
    its dated "video" (``age_clips``, default all in train, video in dev and test). Returns the
    manifest and the aged speakers whose dated video has no audio (they fall back to all clips with
    age NaN)."""
    no_audio = sorted(set(labels["speaker_id"]) - set(clips["speaker_id"]))
    if no_audio:
        raise SystemExit(f"{len(no_audio)} labelled speakers have no audio, e.g. {no_audio[:5]}")
    unlabelled = sorted(set(clips["speaker_id"]) - set(labels["speaker_id"]))
    if unlabelled:
        print(f"[prepare_heightceleb] WARNING: dropping {len(unlabelled)} speakers with no height")
    df = clips.merge(labels, on="speaker_id", how="inner")
    df["split"] = df["speaker_id"].map(split_of)
    aged = df["age"].notna()
    on_video = df["video_id"] == df["age_video"]
    has_video = on_video.groupby(df["speaker_id"]).transform("any")
    missing = sorted(df.loc[aged & ~has_video, "speaker_id"].unique())
    df.loc[aged & ~has_video, "age"] = float("nan")
    mode = df["split"].map({"train": "all", "dev": "video", "test": "video", **(age_clips or {})})
    df = df[~(aged & has_video) | on_video | (mode == "all")]
    return df.reset_index(drop=True), missing


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    clips = scan_wavs(args.wav_root)
    print(f"[prepare_heightceleb] {len(clips):,} clips, {clips.speaker_id.nunique()} speakers, "
          f"{clips[['speaker_id', 'video_id']].drop_duplicates().shape[0]:,} videos "
          "(full VoxCeleb1: 153,516 / 1,251 / 22,496)")
    labels = load_labels(args.heights, args.ages)
    age_clips = {"train": args.train_age_clips, "dev": args.dev_age_clips, "test": args.test_age_clips}
    groups = speaker_groups(clips, pd.read_csv(args.md5_csv, usecols=["speaker_id", "md5"])
                            if args.md5_csv else None)
    if args.split_mode == "grouped":
        split_of = grouped_split(labels, groups, args.fractions, args.seed)
    else:
        split_of = split_speakers(labels, args.val_frac, args.seed)
    df, missing = build_manifest(clips, labels, split_of, age_clips)
    if missing:
        print(f"[prepare_heightceleb] WARNING: {len(missing)} aged speakers' dated video has no "
              f"audio -> age dropped, all clips kept: {missing[:10]}")

    out_dir = args.out.expanduser().resolve() / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        sub = df[df["split"] == split]
        sub[COLS].to_csv(out_dir / f"{split}.csv", index=False)
        spk = sub.drop_duplicates("speaker_id")
        print(f"  {split:5s} {len(sub):7,d} clips {len(spk):5d} speakers "
              f"(M {int((spk.gender == 'M').sum())} / F {int((spk.gender == 'F').sum())}) | "
              f"height mean {spk.height.mean():.1f} | with age {int(spk.age.notna().sum())}, "
              f"median {spk.age.median():.0f}")

    for split, how in age_clips.items():
        per_aged = df[(df["split"] == split) & df["age"].notna()].groupby("speaker_id").size()
        if len(per_aged):
            print(f"aged speakers' clips, {split} ({how}): min {per_aged.min()}, median "
                  f"{per_aged.median():.0f}, max {per_aged.max()}; fewer than 8 (a full bag): "
                  f"{int((per_aged < 8).sum())} of {len(per_aged)}")
    # Leakage check (SESSION_HANDOFF §0.7.14): groups / videos that span two splits.
    g = pd.Series(groups).rename("group").rename_axis("speaker_id").reset_index()
    g["split"] = g["speaker_id"].map(split_of)
    split_groups = int((g.groupby("group")["split"].nunique() > 1).sum())
    split_videos = int((clips.assign(split=clips["speaker_id"].map(split_of))
                        .groupby("video_id")["split"].nunique() > 1).sum())
    print(f"groups: {g.group.nunique()} (largest {g.group.value_counts().max()} speakers); "
          f"spanning 2+ splits: {split_groups} groups, {split_videos} videos ({args.split_mode} mode)")
    if args.split_mode == "grouped":
        g.to_csv(out_dir.parent / "groups.csv", index=False)
        if split_groups or split_videos:
            raise SystemExit("grouped split left a group or video in two splits -- a bug")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
