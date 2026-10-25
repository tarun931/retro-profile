# Datasets

The codebase supports three speaker-profile corpora. Each has a
`scripts/prepare_*.py` script that converts the raw release into the standard
manifest format expected by `BaseSpeakerDataset`:

```
<root>/
    manifests/
        train.csv
        dev.csv
        test.csv
    <audio files referenced from the manifests>
```

Each manifest has at least:

| column | type | meaning |
|---|---|---|
| `speaker_id` | str | unique speaker identifier |
| `utt_id` | str | unique utterance identifier within the speaker |
| `audio_path` | str | absolute or root-relative path to the audio file |
| `<target>` | float | speaker-level target label (`age`, `height`, ...) |

## TIMIT

  * 630 speakers, 8 dialect regions, ~10 utterances/speaker
  * Targets: `age` (years), `height` (cm)
  * Standard split: 462 train + 168 test; we carve a 10 % speaker-disjoint
    dev set out of train.
  * Audio: SPHERE-headered .WAV at 16 kHz; modern `libsndfile`/`soundfile`
    reads SPHERE natively (no `sph2pipe` needed in most environments).

```
python scripts/prepare_timit.py --raw /data/timit/raw --out data/timit
```

## NISP

  * 5 Indian languages (Hindi, Tamil, Telugu, Malayalam, Kannada)
  * 345 speakers, ~120 utts/speaker
  * Targets: `age`, `height`, `weight`, `shoulder_size`
  * Useful for cross-lingual transfer experiments

```
python scripts/prepare_nisp.py --raw /data/nisp/raw --out data/nisp
```

The discovery script in `scripts/prepare_nisp.py` assumes a layout of
`<lang>/audio/<speaker>/*.wav` and a `<lang>/metadata.csv`. Adjust to match
your local copy of the corpus.

## HeightCeleb

  * Height-labeled subset of VoxCeleb (large-scale, in-the-wild)
  * Target: `height` only (no age labels)
  * Crowdsourced labels — useful for noisy cross-corpus evaluation

```
python scripts/prepare_heightceleb.py --raw /data/heightceleb/raw --out data/heightceleb
```

The script expects a `heights.csv` at the corpus root with at minimum
`speaker_id` and `height_cm` columns, plus an `audio/<speaker>/*.wav` tree.

## NaN labels and cross-corpus training

`MaskedRegressionLoss` ignores NaN positions, so you can train on a union of
corpora that have non-overlapping label sets:

```yaml
data:
  _target_: retro_profile.data.HeightCelebDataModule
  targets: [age, height]   # age will be NaN for HeightCeleb rows
```

The collate puts NaN in `labels["age"]` and 0 in `label_mask["age"]` for
HeightCeleb rows; the loss skips them; OrdCon also skips per-target if there
are too few valid pairs.
