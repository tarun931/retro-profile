# RETRO-Profile architecture

This document explains the architecture and the rationale behind each
component. For a higher-level description see the proposal in the thesis;
this file is for engineers reading the code.

## High-level dataflow

```
Speaker S has utterances {x₁, ..., x_K}
                   │
                   ▼
        ┌─────────────────────┐
        │   Encoder (WavLM)    │   per-utterance embeddings (B, K, D_enc)
        └──────────┬───────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  MIL Attention Pool  │   per-target speaker embedding (B, D_enc)
        │    (per-target)      │   one head stack per trait
        └──────────┬───────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │       Projector      │   shared, L2-normalized (B, D_proj)
        │   (OrdCon-organized) │   used for FAISS indexing
        └──────────┬───────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Residual Retrieval   │   ŷ = anchor(z) + Δ(z)
        │   Head (per target)  │   anchor = kNN over datastore
        └─────────────────────┘    Δ = small MLP, tanh-clipped
```

## Component responsibilities

### Encoder (`retro_profile.models.encoders`)

Self-supervised speech encoder. We default to WavLM-Large (1024-dim) with
weighted layer aggregation — the same recipe used by ECAPA-TDNN-WavLM in
speaker verification, which has consistently outperformed using only the last
layer for any speaker-related task.

The encoder is **frozen** by default. Only the layer weights, MIL pooling,
projector, and heads are trained — that's roughly 1-3 % of total parameters,
which is what makes the model tractable on a single GPU.

### MIL pooling (`retro_profile.models.pooling.MILAttentionPooling`)

Gated multi-head attention pooling adapted from Ilse et al. 2018, with one
key extension: a **separate head stack per target trait** (`age`, `height`).
Each head stack averages `num_heads` independently parameterised gated
attention heads.

Why per-target pooling? Vowel-rich utterances reveal vocal tract length →
height; sustained voicing reveals jitter and shimmer → age. The same speaker
bag should be pooled differently for the two traits. Sharing the pooling
across traits hurts because the model is forced to compromise on what is
"informative".

### Projector (`retro_profile.models.heads.MLPHead` with `l2_normalize=True`)

Projects the pooled speaker embedding into a fixed-dim, L2-normalized space.
Two purposes:

  1. **OrdCon target**: the projector output is what the OrdCon loss
     organizes — pushing similar-trait speakers close and dissimilar-trait
     speakers apart, with directional alignment to the trait axis.
  2. **FAISS query space**: kNN is run in this space, so cosine similarity
     is well-defined.

The projector is **shared across targets** — there is one projected embedding
per speaker, and different targets read different label columns out of the
same set of retrieved neighbours. This is what gives RETRO-Profile its
forensic explanation property: the same K speakers explain both age and
height predictions.

### Residual retrieval head (`retro_profile.models.heads.ResidualRetrievalHead`)

For each target, predict:

```
ŷ = clamp(label_min, label_max,  anchor(z) + Δ(z))
anchor(z) = Σᵢ softmax(sim(z, zᵢ) / τ) · yᵢ      (over kNN of z)
Δ(z)      = delta_clip · tanh(MLP(z))
```

The residual is hard-clipped to ±delta_clip (typically 5 cm or 6 years), so
predictions cannot wander far from the retrieval anchor. This converts an
open-ended regression problem into bounded interpolation between real
observed humans.

When the datastore is empty (e.g. early in training), the head degrades to
plain MLP regression on the projector output — same parameters get used, no
extra logic needed.

### FAISS datastore (`retro_profile.retrieval.RetrievalDatastore`)

One entry per training speaker. Keyed by an L2-normalised, projector-output
embedding; carries that speaker's labels. Built post-training by
`scripts/build_datastore.py` and attached to retrieval-aware heads via
`retrieval.attach_datastore`.

## Loss

`MultitaskLoss = regression_weight * Σ MaskedRegression(target)
                 + ordcon_weight * OrdCon(projector_out)`

  * `MaskedRegression`: smooth-L1 by default; ignores NaN labels (HeightCeleb
    has only height; we want to be able to fine-tune cross-corpus without
    losing rows).
  * `OrdCon`: order-enhanced contrastive loss over the projected embedding.
    Two terms — a margin-ranking direction loss along a learnable per-target
    unit vector, and a magnitude-alignment term that ties pairwise embedding
    distances to absolute label gaps.

## Training loop notes

  * Targets are z-scored by the train-set mean and std before the loss; we
    de-normalize predictions for metric reporting in real units (years, cm).
  * The retrieval datastore can be **rebuilt mid-training** via the
    `PeriodicDatastoreRebuild` callback, but disabled by default — early
    epochs produce noisy embeddings, so building once after training is
    usually enough.
  * Gradient checkpointing for WavLM is *not* enabled by default; with the
    encoder frozen, memory is dominated by the attention pooling + heads,
    which are small.

## Reproducing the proposal numbers

Recommended order:
  1. `python scripts/train.py experiment=no_mil` — confirms mean pooling baseline.
  2. `python scripts/train.py experiment=no_ordcon` — pure regression w/ MIL.
  3. `python scripts/train.py experiment=no_knn` — MIL + OrdCon, no retrieval.
  4. `python scripts/train.py experiment=retro_profile_full` — full model.

Then:
  5. `python scripts/build_datastore.py --ckpt ... --output ... --config-dir ...`
  6. `python scripts/evaluate.py --ckpt ... --datastore ... --eval timit nisp heightceleb`

The cross-corpus evaluation in step 6 is the headline result — RETRO-Profile
should show a markedly smaller drop in MAE going from TIMIT-test to NISP/
HeightCeleb than any of the ablations.
