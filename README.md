# RETRO-Profile

Code for the master's thesis **"Prediction of Age and Height Using Audio Recording Data"** (Tarun Singh, 2026).

RETRO-Profile estimates a speaker's **age** (years) and **height** (cm) from a set of their recordings. It treats each
speaker as a bag of clips and combines three parts, each tested by ablation in the paper:

1. **Per-trait MIL attention**: a separate gated attention over the speaker's clips for age and for height (RQ2).
2. **OrdCon**: an ordinal contrastive loss that orders the speaker embedding by age and by height (RQ4).
3. **Retrieval**: a second stage that predicts from the k most similar training speakers, plus a small bounded
   correction, so every estimate can be traced to real speakers (RQ3).

This repository contains what is needed to reproduce the paper's result tables: the model, the comparison models,
the configuration of every reported run, the scripts that train and score them, and a script that prints the tables.
No data, trained models or derived labels are included.

---

## Results reported in the paper

TIMIT test set, 168 speakers (112 men, 56 women), one prediction per speaker from all 10 of their clips. Mean ± SD
over five seeds (42, 1, 2, 3, 4). Lower is better.

| Model | Age MAE (years) | Height MAE (cm) |
|---|---|---|
| **RETRO-Profile** | **3.83 ± 0.11** | **5.05 ± 0.17** |
| Gupta et al. (WavLM Base+) | 3.78 ± 0.17 | 5.23 ± 0.17 |
| Gupta et al. as published (wav2vec 2.0) | 3.82 ± 0.22 | 5.15 ± 0.10 |
| WavLM + MLP | 3.97 ± 0.17 | 5.30 ± 0.11 |
| Jaid et al. | 5.60 ± 0.01 | 5.32 ± 0.04 |
| No-audio reference (each gender's training mean) | 5.94 | 5.31 |

Ablations (change in MAE when a part is removed, with the paired 95 % bootstrap interval):

| Removed | Age (years) | Height (cm) |
|---|---|---|
| MIL (clips averaged instead) | −0.05 [−0.20, +0.12] | −0.10 [−0.24, +0.03] |
| Retrieval (direct regression only) | +0.18 [+0.04, +0.32] | +0.18 [+0.08, +0.29] |
| Residual correction (kNN alone) | +0.01 [−0.01, +0.03] | +0.00 [+0.00, +0.00] |
| OrdCon | +2.05 [+1.47, +2.64] | +0.35 [+0.07, +0.65] |

Transfer to HeightCeleb test (249 speakers, 118 with an age), three seeds, after the two-number calibration.
4a: trained on TIMIT + NISP (695 speakers). 4b: plus 84 age-balanced HeightCeleb training speakers (779).

| Model | 4a age MAE | 4a height MAE | 4b age MAE | 4b height MAE |
|---|---|---|---|---|
| RETRO-Profile | 8.71 ± 0.13 | 5.40 ± 0.01 | 7.96 ± 0.32 | 5.43 ± 0.10 |
| WavLM + MLP | 8.04 ± 0.14 | 5.24 ± 0.03 | 7.91 ± 0.63 | 5.53 ± 0.17 |
| Gupta et al. (WavLM Base+) | 8.82 ± 0.64 | 5.38 ± 0.05 | 8.33 ± 0.14 | 5.49 ± 0.09 |
| Gupta et al. as published (wav2vec 2.0) | 11.29 ± 0.14 | 5.47 ± 0.02 | 8.82 ± 0.32 | 5.49 ± 0.38 |
| Jaid et al. | 11.09 ± 0.02 | 6.51 ± 0.08 | 10.87 ± 0.14 | 6.04 ± 0.05 |
| No-audio reference | 10.76 | 5.58 | 10.76 | 5.58 |

Paired differences are always "first model minus second model" in MAE (negative favours the first), with a 95 %
bootstrap interval over test speakers (10,000 resamples). A difference is called clear when its interval excludes 0.

**What "similar results" means.** GPU training is not bit-for-bit deterministic, so retraining gives numbers close
to these, not identical ones; the seed-to-seed SDs above (about 0.1–0.2 years and 0.1–0.2 cm on TIMIT) show the size
of variation to expect. Run on the original runs' prediction files, `scripts/make_tables.py` reproduces every number
of the paper's result tables exactly (860 numbers, Tables 2–5, A2–A6, A11, A12).

---

## The model in brief

| Part | Setting |
|---|---|
| Input | All 10 clips per speaker, 16 kHz, each cut to at most 8 s |
| Encoder | WavLM Base+ (`microsoft/wavlm-base-plus`), **fine-tuned**; learned weighted sum of layers; mean over time → 768-d per clip |
| MIL pooling | Gated attention, one stack per trait, 4 heads each, hidden 256 |
| Projector | Shared MLP 768 → 512 → 256, L2-normalised; its output r is the retrieval embedding |
| Stage 1 loss | Smooth L1 on z-scored age and height + OrdCon on r (λ = 1, margin 0.1, scale 5 SD) |
| Training | AdamW, LR 1e-5 encoder / 1e-4 rest, 60 epochs, 4 speakers × 10 clips per step, 16-bit mixed precision |
| Checkpoint | Best validation score S = MAE_age / 5.91 + MAE_height / 7.40 on 70 validation speakers |
| Stage 2 | Frozen model; datastore of the 392 training speakers; cosine similarity, softmax weights (τ = 0.1); k per trait chosen on validation from 17 values (1–300); bounded residual (±6 years, ±5 cm), kept only if it lowers validation MAE |

The full list is in Appendix C.1 of the paper (Table A13). The comparison models are reimplemented from their papers
(`src/retro_profile/models/baselines/gupta2022.py`, `jaid2023.py`). WavLM + MLP is RETRO-Profile with all three parts
removed, set by configuration only.

---

## Repository layout

```
retro-profile/
├── README.md                        this file
└── RETRO-Profile_code/              run every command below from inside this folder
    ├── configs/
    │   ├── experiment/              the paper's runs (table below) and the files they build on
    │   ├── data/                    TIMIT; TIMIT + NISP; TIMIT + NISP + HeightCeleb sample; HeightCeleb split
    │   ├── model/                   RETRO-Profile, Gupta et al., Jaid et al.
    │   └── training/                optimiser and schedule per model family
    ├── src/retro_profile/           model, losses (incl. OrdCon), retrieval datastore, training loop, evaluation
    ├── scripts/
    │   ├── prepare_timit.py, prepare_nisp.py, prepare_heightceleb.py    data preparation
    │   ├── clip_md5.py              audio fingerprints used to build the HeightCeleb split (Appendix A.3)
    │   ├── sample_heightceleb_ages.py   the 84-speaker age-balanced sample (Table 5, 4b)
    │   ├── train.py                 stage 1 training (Hydra)
    │   ├── select_checkpoint.py     the checkpoint training chose
    │   ├── evaluate.py              per-speaker and per-clip predictions
    │   ├── fit_residual.py          stage 2: kNN datastore, k, residual correction
    │   ├── calibrate_predictions.py two-number calibration for transfer (Section 4.6)
    │   ├── build_datastore.py, neighbour_quality.py, neighbour_inspection.py   Table 4, A11, A12
    │   ├── paired_bootstrap.py      paired 95 % intervals
    │   ├── make_tables.py           prints the paper's tables from the saved predictions
    │   └── reproduce/               timit.sh, transfer.sh, neighbours.sh
    ├── heightceleb/build_vox1_ages.py    derives HeightCeleb ages (Appendix A.2); labels not included
    ├── tests/                       pytest suite
    └── requirements.txt, pyproject.toml, LICENSE
```

### The runs behind each table

Each run is one configuration file in `configs/experiment/` plus one seed. Output goes to
`runs/<configuration>_<test corpus>_s<seed>/`.

| Paper | Model | Configuration | Seeds |
|---|---|---|---|
| Table 2 | RETRO-Profile | `retro_ordcon_fixed_l100_s5_b10` | 42, 1, 2, 3, 4 |
| | WavLM + MLP | `wavlm_direct_budget_b10_e100` | 42, 1, 2, 3, 4 |
| | Gupta et al. (WavLM Base+) | `gupta2022_finetune_wavlm_lr3e6` | 42, 1, 2, 3, 4 |
| | Gupta et al. as published (wav2vec 2.0) | `gupta2022_finetune_w2v2_lr3e6` | 42, 1, 2, 3, 4 |
| | Jaid et al. | `jaid2023_lr3e3` | 42, 1, 2, 3, 4 |
| Table 3 | − MIL / − OrdCon | `retro_ordcon_fixed_l100_s5_b10_nomil` / `_noordcon` | 42, 1, 2, 3, 4 |
| Table 5 (4a) | the five models | `timit_nisp_b10_l100_s5`, `timit_nisp_mlp_b10_e100`, `timit_nisp_gupta_lr3e6`, `timit_nisp_gupta_w2v2_lr3e6`, `timit_nisp_jaid_lr3e3` | 42, 1, 2 |
| Table 5 (4b) | the five models | the same names with `timit_nisp_hcs_` | 42, 1, 2 |

"− retrieval" and "− residual" in Table 3 are readings of the full model written by stage 2; they need no extra
training. Some configurations build on a parent file (for example `retro_ordcon_fixed_l100_s5_b10` extends
`retro_ordcon_fixed_l100_s5`); the parents are included unchanged, exactly as they were run. Comments in the
configuration files name the cluster job scripts originally used; `scripts/reproduce/*.sh` run the same steps.

---

## Setup

Python 3.10 or later and one GPU with 48 GB of memory (the paper used one NVIDIA L40S per run, 16-bit mixed
precision). A RETRO-Profile seed trains in about 43 minutes on TIMIT; the transfer runs, on more data, take about
1–3 hours each. Training keeps a few checkpoints per run (several GB); once a run is scored, only the checkpoint
printed by `python scripts/select_checkpoint.py runs/<run>` is needed.

```bash
cd RETRO-Profile_code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Use `PYTHONPATH=src` rather than `pip install -e .`: the pinned versions in `requirements.txt` are the tested set.
The pre-trained encoders (`microsoft/wavlm-base-plus`, `facebook/wav2vec2-base`) are downloaded from Hugging Face on
first use.

Check the installation (no data or GPU needed):

```bash
pytest tests/
```

The tests cover the released code: units, initialisation, missing labels, checkpoint selection, the reproduction
checks, and that compared runs differ only where intended (for example, WavLM + MLP is the final RETRO-Profile
configuration minus MIL, OrdCon and retrieval; the transfer runs change only the training data). The paper's count
of 461 tests includes tests of development code that is not part of this release.

---

## Data

The corpora are not included; obtain each from its provider.

| Corpus | Used for | Source |
|---|---|---|
| TIMIT (LDC93S1) | training and test (Tables 2–4) | https://catalog.ldc.upenn.edu/LDC93S1 |
| NISP | extra training data for transfer (Table 5) | https://github.com/iiscleap/NISP-Dataset |
| VoxCeleb1 audio | HeightCeleb audio (Table 5) | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html |
| HeightCeleb heights | HeightCeleb labels (Table 5) | https://github.com/stachu86/HeightCeleb |

```bash
# TIMIT: 462 training speakers -> 392 train + 70 validation; 168 test speakers
python scripts/prepare_timit.py --raw /path/to/timit --out data/timit

# NISP: English sessions; all 303 speakers are used for training
python scripts/prepare_nisp.py --raw /path/to/nisp --out data/nisp
```

**HeightCeleb ages (Appendix A.2).** HeightCeleb gives heights only. Ages are derived per video as upload year minus
birth year, with the birth year confirmed by both Wikidata and DBpedia. Copy HeightCeleb's `height_celeb.csv` into
`heightceleb/`, download the dataset files of the VoxCeleb enrichment of Hechmi et al.
(https://github.com/hechmik/voxceleb_enrichment_age_gender), then:

```bash
python heightceleb/build_vox1_ages.py videos --hechmi-repo /path/to/voxceleb_enrichment_age_gender
python heightceleb/build_vox1_ages.py lookup     # queries the public Wikidata and DBpedia services
python heightceleb/build_vox1_ages.py ages       # writes heightceleb/vox1_height_age.csv
```

Wikidata and DBpedia change over time, so a fresh lookup can give slightly different ages from the 573 aged
speakers used in the paper; this affects the age columns of Table 5 only.

**HeightCeleb split and the 84-speaker sample (Appendix A.3).** All 1,251 speakers are re-split 60/20/20
(751 / 251 / 249); speakers who share a video or an identical clip stay in the same split.

```bash
python scripts/clip_md5.py --wav-root /path/to/voxceleb1/wav --out data/vox1_clip_md5.csv
python scripts/prepare_heightceleb.py --wav-root /path/to/voxceleb1/wav --out data/heightceleb_split2 \
    --split-mode grouped --md5-csv data/vox1_clip_md5.csv --seed 42
python scripts/sample_heightceleb_ages.py --manifest data/heightceleb_split2/manifests/train.csv \
    --out-root data/heightceleb_split2_age_sample --list data/table4b_age_sample.csv --per-band 15 --seed 0
```

The sample takes the same number of speakers from each of seven age bands; the smallest band holds 12, so it draws
12 per band (84 speakers).

---

## Reproducing the tables

### Tables 2 and 3 (TIMIT)

```bash
bash scripts/reproduce/timit.sh                      # all 7 configurations x 5 seeds (35 training runs)
SEEDS="42" bash scripts/reproduce/timit.sh           # one seed of each
CONFIGS="retro_ordcon_fixed_l100_s5_b10" SEEDS="42" bash scripts/reproduce/timit.sh   # one run
```

For one run the script does the following (RETRO-Profile, seed 42):

```bash
cfg=retro_ordcon_fixed_l100_s5_b10; seed=42; run=runs/${cfg}_timit_s${seed}
python scripts/train.py experiment=$cfg seed=$seed run_name=${cfg}_timit_s${seed} num_workers=8   # stage 1
ckpt=$(python scripts/select_checkpoint.py $run)                                       # the epoch training chose
python scripts/evaluate.py --ckpt $ckpt --config-dir $run/.hydra --eval timit \
    --unit speaker --bag-size 10 --output $run/eval_spk                                 # kNN off
python scripts/fit_residual.py --run $run --corpus timit --bag-size 10 --suffix _b10 --check-sub eval_spk   # stage 2
```

Stage 2 writes `runs/<run>/stage2_b10/` with three readings for validation and test: `direct` (kNN off), `knn`
(kNN alone) and `knn_residual` (the full model); before training anything it checks that it reproduces the stage-1
predictions. WavLM + MLP is scored the same way without stage 2. Gupta et al. and Jaid et al. predict per clip
(`--unit utterance`); following the paper's protocol, so that every model is compared per speaker, a speaker's
estimate is the mean over their 10 clips.

### Table 4, A11 and A12 (neighbours)

```bash
bash scripts/reproduce/neighbours.sh                 # needs the RETRO-Profile and - OrdCon runs of timit.sh
```

### Table 5 (transfer to HeightCeleb)

```bash
bash scripts/reproduce/transfer.sh 4a                # 5 models x 3 seeds
bash scripts/reproduce/transfer.sh 4b                # needs the 84-speaker sample above
```

Each baseline is scored on the 251 HeightCeleb validation speakers and on the test speakers;
`calibrate_predictions.py` fits `corrected = a × prediction + b` per trait on the validation speakers and applies it
to test. RETRO-Profile's calibration happens inside its stage 2 (`fit_residual.py --train-from-run --calibrated`),
whose datastore is built from the run's own training corpora.

### Print the tables

```bash
python scripts/make_tables.py                        # prints every table it has the runs for
python scripts/make_tables.py --out tables.md        # also writes them to a file
```

It reads only the saved per-speaker predictions in `runs/` and the data manifests, and skips (naming the missing
file) any table whose runs are not all there.

| Paper table | Runs needed |
|---|---|
| Table 2, A2 | `timit.sh` (25 runs: the five models) |
| Table 3, A4 | `timit.sh` (all 35 runs) |
| Table 4, A11, A12 | `timit.sh`, then `neighbours.sh` |
| Table 5, A3, A5, A6 | `transfer.sh 4a` and `transfer.sh 4b` |

---

## Licence

Code: MIT (`RETRO-Profile_code/LICENSE`). The corpora keep their own licences (TIMIT: LDC; NISP, VoxCeleb1 and
HeightCeleb: their providers' terms). This repository contains code only.

## Citation

```bibtex
@misc{singh2026retroprofile,
  author       = {Tarun Singh},
  title        = {Prediction of Age and Height Using Audio Recording Data},
  howpublished = {Master's thesis},
  year         = {2026},
  url          = {https://github.com/tarun931/retro-profile}
}
```
