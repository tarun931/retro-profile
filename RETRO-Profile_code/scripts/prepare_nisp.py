"""Prepare NISP for retro-profile training.

Rewritten 2026-08-31 against the ACTUAL release. The previous version assumed
`<raw>/<language>/audio/<speaker>/*.wav` plus a per-language `metadata.csv`; no
such layout exists. See SESSION_HANDOFF.md §11.

REAL LAYOUT, after downloading the archives from github.com/iiscleap/NISP-Dataset and extracting them:

    <raw>/<Lang>_master/<session>/RECS/<NNNN>/<file>.wav
    <raw>/total_spkrinfo.list      space-separated, WITH header
    <raw>/train_spkrID             one id per line, e.g. "Hin_0002___TRN"
    <raw>/test_spkrID              same, "___TST"

`<session>` is `<Lang>` (native speech) or `English_<Lang>`. Filenames look like
`Hin_0037_Hin_m_0026.wav`:

    Hin   language code        0037  speaker number
    Hin   session language     m     gender        0026  utterance index

`speaker_id` is the first 8 characters (`Hin_0037`), which is the official
convention and joins directly to `total_spkrinfo.list`.

TWO THINGS THE OFFICIAL SCRIPT GETS WRONG, deliberately not copied:

1. It reads `train_spkrID` with `pd.read_csv(header=None)` and then indexes
   column 3 for height-stratified splitting. That file has ONE field per line,
   so column 3 does not exist and the code raises KeyError. We take the official
   train/test split from those files and stratify the val split on height read
   from `total_spkrinfo.list`, where height actually lives -- their intent,
   without the bug.
2. It calls `librosa.load(file)`, which defaults to `sr=22050`, then resamples to
   16 kHz: 44.1k -> 22.05k -> 16k, discarding quality for nothing. We do not
   resample here at all; the manifest points at the source audio and the loader
   resamples once, at its native rate.

Usage:
    python scripts/prepare_nisp.py --raw <data-root>/nisp --out data/nisp
    python scripts/prepare_nisp.py --raw ... --out ... --sessions english
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LANGUAGES = ["Hindi", "Kannada", "Malayalam", "Tamil", "Telugu"]

# Columns of total_spkrinfo.list, in order. Height in cm, weight in kg, age in
# years (with decimals, unlike TIMIT's integers).
SPKR_COLUMNS = [
    "Speaker_ID", "Gender", "Mother_Tongue", "Height", "Shoulder_size",
    "Waist_size", "Weight", "Age", "Native_State", "Native_District",
]


def read_speaker_info(raw: Path) -> pd.DataFrame:
    path = raw / "total_spkrinfo.list"
    if not path.is_file():
        raise SystemExit(
            f"missing {path}. Fetch it with:\n"
            f"  wget -q -O {path} https://raw.githubusercontent.com/"
            f"iiscleap/NISP-Dataset/master/total_spkrinfo.list"
        )
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in SPKR_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing expected columns: {missing}")
    return df


def read_split_ids(raw: Path, name: str) -> set[str]:
    """Official split files. Lines look like `Hin_0002___TRN`; the id is [:8]."""
    path = raw / name
    if not path.is_file():
        raise SystemExit(f"missing {path} (fetch it from the NISP repo root)")
    ids = {line.strip()[:8] for line in path.read_text().splitlines() if line.strip()}
    return ids


def load_transcripts(raw: Path) -> dict[tuple[str, int], str]:
    """Map (recording stem, utterance index) -> sentence id, e.g. "item0004".

    Layout: <Lang>_master/scripts/<something>_txt/<stem>.txt, where <stem> is the
    wav name minus its index — `Hin_0001_Eng_f.txt` covers
    `Hin_0001_Eng_f_0000.wav` through `..._0040.wav`. Each line is
    `itemNNNN<TAB>the sentence text`.

    THE MAPPING IS BY ITEM NUMBER, not by line position. Corrected 2026-09-12:
    speaker Hin_0001's English wavs run `_0000`.._0005` and then jump to
    `_4808`.._4842`, exactly mirroring the transcript's `item0000`..`item0005`,
    `item4808`.. — so the wav's four-digit suffix IS the item number.

    The first six coincide with their line positions, which is why an earlier
    positional reading appeared to work: it tagged exactly 6 items per speaker
    (2,070 of 14,692 rows) and left the rest UNK.

    That identity is what makes NISP a replication test for the SA finding: the
    first six items are read by every speaker, and item0004 / item0005 are
    TIMIT's sa1 and sa2 verbatim ("She had your dark suit in greasy wash water
    all year" / "Don't ask me to carry an oily rag like that").
    """
    out: dict[tuple[str, int], str] = {}
    # ENGLISH ONLY, and deliberately so. English transcripts are one file per
    # speaker (`English_<lang>_txt/Hin_0001_Eng_f.txt`, 41 lines for 41 wavs), so
    # the positional mapping is exact. Native-session transcripts are SPLIT
    # across `<Lang>_txt/para/` and `<Lang>_txt/uniq/`, each restarting its line
    # numbering at zero, and the concatenation order that would reconstruct the
    # recording sequence is not documented. Malayalam is different again
    # (`1-30_uniq_plus_novel/speaker_7.txt`, keyed by ordinal, not speaker id).
    # Rather than guess an ordering we cannot verify, native utterances are left
    # untyped -- use `--sessions english` so every row in the manifest carries a
    # real sentence_type.
    for txt in raw.glob("*_master/scripts/English_*/*.txt"):
        stem = txt.stem                      # Hin_0001_Eng_f
        for line in txt.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            item = line.split("\t", 1)[0].strip()
            if not item.startswith("item"):
                continue
            try:
                number = int(item[4:])       # item4808 -> 4808, matches _4808.wav
            except ValueError:
                continue
            out[(stem, number)] = item
    return out


def discover(raw: Path, sessions: str,
             transcripts: dict[tuple[str, int], str] | None = None) -> pd.DataFrame:
    """Walk <Lang>_master/<session>/RECS/<NNNN>/*.wav."""
    transcripts = transcripts or {}
    rows: list[dict] = []
    for lang in LANGUAGES:
        base = raw / f"{lang}_master"
        if not base.is_dir():
            print(f"[prepare_nisp] missing language directory: {base}")
            continue
        wanted = (
            [f"English_{lang}"] if sessions == "english"
            else [lang] if sessions == "native"
            else [lang, f"English_{lang}"]
        )
        for sess in wanted:
            recs = base / sess / "RECS"
            if not recs.is_dir():
                print(f"[prepare_nisp] missing session directory: {recs}")
                continue
            for wav in sorted(recs.rglob("*.wav")):
                stem = wav.stem                       # Hin_0037_Hin_m_0026
                parts = stem.split("_")
                if len(parts) < 5:
                    print(f"[prepare_nisp] unexpected filename, skipping: {wav.name}")
                    continue
                rows.append({
                    "speaker_id": stem[:8],           # Hin_0037
                    "utt_id": stem,
                    "audio_path": str(wav.resolve()),
                    "language": lang,
                    # "Hin" = native session, "Eng" = English session. Kept
                    # because it is a within-speaker contrast: the same person
                    # reading in two languages.
                    "session_lang": parts[2],
                    "utt_index": parts[4],
                    # Sentence identity, shared across speakers. The wav's
                    # four-digit suffix is the item number, so this is a direct
                    # lookup. "" when the transcripts are absent.
                    "item_id": transcripts.get(
                        ("_".join(parts[:4]), int(parts[4])), ""),
                    # Assigned after the walk, from how many speakers read
                    # each item (see classify_sentence_types).
                    "sentence_type": "UNK",
                })
    if not rows:
        raise SystemExit(
            f"no wav files found under {raw}. Has the archive been extracted? "
            "It is DOUBLE-gzipped: zcat RECS.tar.gz | tar -xzf -"
        )
    return pd.DataFrame(rows)


def classify_sentence_types(df: pd.DataFrame, common_frac: float = 0.5) -> pd.DataFrame:
    """Tag each utterance COMMON / SHARED / UNIQUE, the analogue of TIMIT SA/SX/SI.

    Derived from the data rather than assumed: count how many distinct SPEAKERS
    read each `item_id`, within each session language separately (an item read by
    every Hindi speaker is not "common" across the corpus if only Hindi speakers
    saw it).

        COMMON  read by >= common_frac of that session's speakers  -> TIMIT "SA"
        SHARED  read by more than one speaker, but fewer           -> TIMIT "SX"
        UNIQUE  read by exactly one speaker                        -> TIMIT "SI"

    VERIFIED 2026-09-12 by reading the actual transcripts: item0000..item0005 are
    read by all 345 speakers with byte-identical text, and every other item number
    belongs to exactly one speaker (Hin_0001 has 4808-4842, Hin_0002 4843-4872,
    Tel_0001 0006-0046, Tel_0002 0047-0090 -- contiguous per language, no overlap).

    So NISP is a TWO-way contrast, not TIMIT's three-way: COMMON versus UNIQUE,
    with SHARED expected to be empty. The paper's "20-25 unique + 20-25 contextual"
    sentences are all speaker-specific, so item ids cannot separate those two --
    both land in UNIQUE. That is fine for the SA test, which needs COMMON versus
    everything else.

    COMMON is the class that matters: it holds TIMIT's sa1 and sa2 verbatim
    (item0004 / item0005) plus the three news sentences the NISP authors added
    "to perform speaker profiling in a text dependent manner". SESSION_HANDOFF.md
    §3.8c reports that height attention ranks TIMIT's SA first in all seven
    measurements while age ranks it last; §3.9b then shows that only happens in
    the two-target model. This column is what lets both be tested here.
    """
    if "item_id" not in df.columns or not (df["item_id"].astype(str) != "").any():
        return df

    def _label(item: str, per_item: pd.Series, thresh: float) -> str:
        if not item:
            return "UNK"
        c = int(per_item.get(item, 0))
        if c >= thresh:
            return "COMMON"
        return "SHARED" if c > 1 else "UNIQUE"

    df = df.copy()
    # Group by session language AND mother-tongue language. Item numbers are
    # only unique WITHIN a language: Hindi's non-common items run 4808-4842,
    # Kannada's 2349-2389, Tamil's 8081-8117, but Telugu (0006-0046) and
    # Malayalam (0006-0045) both start at 0006 and collide. Verified 2026-09-12
    # by reading the transcripts -- Tel item0006 is "But retailers have seen
    # strong sales of big-ticket items..." while Mal item0006 is "The south
    # Carolina educational radio network...".
    #
    # Pooling them produced 1,545 phantom SHARED items, every one read by
    # exactly 2 speakers -- the signature of a two-way collision, not of
    # genuinely shared text. Counting per language removes them.
    for _key, block in df.groupby(["session_lang", "language"]):
        counts = block.groupby("item_id")["speaker_id"].nunique()
        cutoff = max(2.0, common_frac * block["speaker_id"].nunique())
        df.loc[block.index, "sentence_type"] = [
            _label(str(i), counts, cutoff) for i in block["item_id"]
        ]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="the nisp/ directory")
    ap.add_argument("--out", required=True, type=Path)
    # english by default: only the English transcripts map positionally to wavs,
    # so it is the only setting where every utterance gets a sentence_type. See
    # load_transcripts.
    ap.add_argument("--sessions", choices=["both", "english", "native"], default="english")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of TRAIN speakers held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = args.raw.expanduser().resolve()
    info = read_speaker_info(raw)
    transcripts = load_transcripts(raw)
    if not transcripts:
        print('[prepare_nisp] WARNING: no transcripts found under '
              f'{raw}/*_master/scripts/. sentence_type will be UNK, and the '
              'SA-equivalent analysis cannot run.')
    else:
        print(f'[prepare_nisp] {len(transcripts)} transcript lines loaded')
    files = discover(raw, args.sessions, transcripts)

    df = files.merge(
        info.rename(columns={
            "Speaker_ID": "speaker_id", "Height": "height", "Age": "age",
            "Weight": "weight", "Shoulder_size": "shoulder_size",
            "Gender": "gender", "Mother_Tongue": "mother_tongue",
        })[["speaker_id", "height", "age", "weight", "shoulder_size",
            "gender", "mother_tongue"]],
        on="speaker_id", how="left",
    )
    unlabelled = df["height"].isna().sum()
    if unlabelled:
        print(f"[prepare_nisp] WARNING: {unlabelled} utterances have no speaker metadata")

    # Gender as a single letter, matching the TIMIT manifests (M/F).
    df["gender"] = df["gender"].astype(str).str[0].str.upper()
    df = classify_sentence_types(df)

    train_ids = read_split_ids(raw, "train_spkrID")
    test_ids = read_split_ids(raw, "test_spkrID")

    # Val is carved out of TRAIN speakers, stratified by height so the two sets
    # are comparable -- what the official script intended. Speaker-disjoint.
    train_spk = sorted(train_ids)
    heights = info.set_index("Speaker_ID")["Height"]
    # Systematic sampling up the height order: take every k-th speaker once they
    # are sorted by height. This stratifies exactly (the val set spans the same
    # height range as train) and is reproducible, where a random stratified split
    # can leave the tails uneven at n=210. `--seed` chooses the phase, so
    # different seeds give different — but equally stratified — splits.
    ranked = sorted(train_spk, key=lambda s: (float(heights.get(s, 0.0)), s))
    step = max(2, int(round(1 / args.val_frac)))
    start = args.seed % step
    val_spk = set(ranked[start::step])
    train_spk = set(train_spk) - val_spk
    print(f"[prepare_nisp] split {len(train_spk)} train / {len(val_spk)} val "
          f"/ {len(test_ids)} test speakers "
          f"(val = {len(val_spk) / max(1, len(val_spk) + len(train_spk)):.0%})")

    def split_of(spk: str) -> str:
        if spk in test_ids:
            return "test"
        if spk in val_spk:
            return "dev"
        if spk in train_spk:
            return "train"
        return "unassigned"

    df["split"] = df["speaker_id"].map(split_of)
    stray = int((df["split"] == "unassigned").sum())
    if stray:
        print(f"[prepare_nisp] WARNING: {stray} utterances from speakers in neither "
              "split file — dropped")
        df = df[df["split"] != "unassigned"]

    out_dir = args.out.expanduser().resolve() / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["speaker_id", "utt_id", "audio_path", "age", "height", "gender",
            "weight", "shoulder_size", "mother_tongue", "language",
            "session_lang", "utt_index", "item_id", "sentence_type"]
    for split in ("train", "dev", "test"):
        sub = df[df["split"] == split][cols]
        sub.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"  {split:5s} {len(sub):6d} utts  {sub.speaker_id.nunique():4d} speakers")

    print(f"\nwrote {out_dir}")
    counts = df["sentence_type"].value_counts().to_dict()
    print(f"\nsentence types: {counts}")
    if counts.get("COMMON"):
        common = (df[df.sentence_type == "COMMON"]
                  .groupby("item_id")["speaker_id"].nunique().sort_values(ascending=False))
        print("COMMON items (item -> speakers who read it):")
        for item, n in common.head(10).items():
            print(f"    {item}  {n}")
        print("  item0004/item0005 are TIMIT sa1/sa2 verbatim -- the SA-equivalent set.")
    else:
        print("NOTE: no COMMON items found. Without transcripts the SA-equivalent "
              "analysis cannot run; see SESSION_HANDOFF.md §11.2.")


if __name__ == "__main__":
    main()
