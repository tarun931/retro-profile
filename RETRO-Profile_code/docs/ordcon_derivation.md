# Derivation of the Order-Enhanced Contrastive (OrdCon) Loss

*A first-principles derivation of the loss used as Component 2 of RETRO-Profile,
taking the retrieval requirement as the starting axiom and arriving at the exact
equations implemented in `src/retro_profile/losses/ordcon.py`. All equations are
LaTeX-ready.*

---

## 1. Motivation: why the embedding geometry must be *engineered*

RETRO-Profile predicts a speaker's trait not by direct regression but by
**retrieval**: a query speaker's embedding $z_q$ is compared against a datastore
of training speakers, and the prediction is a similarity-weighted average of the
$k$ nearest neighbours' labels, plus a small learned residual,

$$
\hat{y}_q \;=\; \underbrace{\sum_{i \in \mathcal{N}_k(q)} w_{qi}\, y_i}_{\text{kNN anchor}} \;+\; \underbrace{\delta_\phi(z_q)}_{\text{residual}},
\qquad
w_{qi} = \frac{\exp(\cos(z_q, z_i)/\tau)}{\sum_{i'} \exp(\cos(z_q, z_{i'})/\tau)} .
$$

The quality of $\hat{y}_q$ therefore depends **entirely on the geometry of the
embedding space**: retrieval is only meaningful if *nearest in embedding space*
implies *nearest in label*. A standard self-supervised or regression-trained
embedding does **not** guarantee this — it is free to organise itself by any
nuisance factor (accent, recording channel, phonetic content, speaker identity)
so long as a downstream head can still decode the target. Interpolation over
such a space retrieves acoustically similar but **label-irrelevant** neighbours.

**We must therefore impose the geometry we need.** The OrdCon loss is derived to
enforce, directly on the projected embedding $z = g_\theta(\cdot)$, the two
geometric properties that make retrieval work.

---

## 2. Notation and setup

| Symbol | Meaning |
|---|---|
| $z_i \in \mathbb{R}^D$ | Projected, **$\ell_2$-normalised** speaker embedding ($\lVert z_i\rVert_2 = 1$) |
| $y_i^{t} \in \mathbb{R}$ | Ground-truth label of speaker $i$ for target $t \in \{\text{age}, \text{height}\}$ |
| $m_i^{t} \in \{0,1\}$ | Label mask (0 when target $t$ is missing for speaker $i$, e.g. age on HeightCeleb) |
| $u_t \in \mathbb{R}^D$ | **Learnable unit direction** for target $t$, $\lVert u_t\rVert_2 = 1$ |
| $\mathcal{B}$ | A mini-batch of speakers |
| $m, s, g$ | Margin, magnitude-scale, magnitude-gain hyperparameters |

The embedding is $\ell_2$-normalised because the datastore uses **cosine**
similarity; normalisation makes cosine retrieval well-defined and gives the
margin a consistent scale. (Section 9 discusses the consequence of this choice.)

---

## 3. Design principles (the two axioms)

We formalise "a geometry good for retrieval" as two properties that must hold for
each target $t$:

> **(P1) Order preservation.** If speaker $i$ has a larger label than $j$, then
> $z_i$ must lie *further along the target axis* than $z_j$. The embedding must be
> **monotonic** in the label.

> **(P2) Magnitude alignment.** The embedding distance between two speakers must
> **grow with their label gap**. Speakers far apart in label must be far apart in
> embedding space (so distinct labels never collapse to the same neighbourhood).

(P1) makes the retrieved neighbours *ordinally* correct; (P2) makes the retrieval
*distance* a proxy for label distance, so the similarity weights $w_{qi}$ are
meaningful. We now derive one loss term from each.

---

## 4. Deriving the **direction** (order) term from (P1)

**Step 4.1 — a target axis.** Assume that, locally, the label increases along a
single direction in embedding space — a unit vector $u_t$. This is the minimal
modelling assumption that gives (P1) a concrete meaning. Define the scalar
**position of speaker $i$ along the axis** as the projection

$$
s_i \;=\; \langle z_i,\, u_t\rangle .
$$

**Step 4.2 — the order condition.** Property (P1) says the ordering of positions
must match the ordering of labels:

$$
y_i > y_j \;\;\Longrightarrow\;\; s_i > s_j
\qquad\Longleftrightarrow\qquad
\operatorname{sign}(y_i - y_j)\,\big(s_i - s_j\big) > 0 .
$$

Because $s_i - s_j = \langle z_i - z_j,\, u_t\rangle$, the condition is

$$
\operatorname{sign}(y_i - y_j)\,\langle z_i - z_j,\, u_t\rangle \;>\; 0 .
$$

**Step 4.3 — add a margin.** A strict inequality $>0$ is satisfiable trivially
(e.g. by shrinking all embeddings) and gives no gradient once barely met. We
require the ordering to hold by a **margin** $m > 0$, which enforces a minimum
separation along the axis and prevents collapse:

$$
\operatorname{sign}(y_i - y_j)\,\langle z_i - z_j,\, u_t\rangle \;\ge\; m .
$$

**Step 4.4 — hinge relaxation.** Turn the constraint into a differentiable
penalty that is zero when satisfied and linear in the violation:

$$
\ell^{\text{dir}}_{ij} \;=\; \operatorname{ReLU}\!\Big( m - \operatorname{sign}(y_i - y_j)\,\langle z_i - z_j,\, u_t\rangle \Big).
$$

**Step 4.5 — why $u_t$ is learnable and unit-norm.** We do not know a priori
*which* direction of the $D$-dimensional space encodes the target, so $u_t$ is a
free parameter learned jointly with the projector. Constraining $\lVert u_t\rVert = 1$
(implemented as $u_t = w_t / \lVert w_t\rVert$) keeps the margin $m$ on a fixed
scale; without it the network could satisfy the margin by simply growing
$\lVert u_t\rVert$.

**Step 4.6 — batch average.** Averaging over the ordered pairs with a defined
sign ($y_i \ne y_j$) gives the per-target direction loss:

$$
\boxed{\;\mathcal{L}^{t}_{\text{dir}}
= \frac{1}{|P_t|}\sum_{\substack{(i,j)\in\mathcal{B}^2 \\ y_i \ne y_j}}
\operatorname{ReLU}\!\Big( m - \operatorname{sign}(y_i - y_j)\,\langle z_i - z_j,\, u_t\rangle \Big)\;}
$$

where $P_t$ is the set of such pairs.

---

## 5. Deriving the **magnitude** term from (P2)

**Step 5.1 — the target distance.** Property (P2) asks the embedding distance to
increase with the label gap. Take the simplest such relation — proportionality —
and normalise the label gap into embedding-distance units by a per-target scale
$s$ (the *magnitude scale*):

$$
d^{\star}_{ij} \;=\; \frac{\lvert y_i - y_j\rvert}{s}.
$$

The scale $s$ harmonises targets that live on different numeric ranges (years vs
centimetres), so a single loss can balance age and height.

**Step 5.2 — a lower-bound constraint.** We require the (gain-scaled) embedding
distance to be **at least** the target distance:

$$
g\,\lVert z_i - z_j\rVert_2 \;\ge\; d^{\star}_{ij}
\qquad\Longleftrightarrow\qquad
g\,\lVert z_i - z_j\rVert_2 \;\ge\; \frac{\lvert y_i - y_j\rvert}{s},
$$

with $g$ (the *magnitude gain*) controlling how tightly distance must track the
gap.

**Step 5.3 — one-sided hinge.** We penalise only the "too close" direction:

$$
\ell^{\text{mag}}_{ij} \;=\; \operatorname{ReLU}\!\Big( \tfrac{\lvert y_i - y_j\rvert}{s} - g\,\lVert z_i - z_j\rVert_2 \Big).
$$

The constraint is deliberately **one-sided** (a lower bound, not an equality). We
want to prevent distinct labels from *collapsing*; we do **not** want to punish a
pair for being *too far apart*, because an equality constraint would fight the
direction term (which is already spreading embeddings along $u_t$) and would cap
the useful spread of the space.

**Step 5.4 — exclude self-pairs and average.** The diagonal $i = j$ carries no
information ($\ell^{\text{mag}}_{ii}=0$ but it dilutes the mean), so we exclude it
and average over the $n(n-1)$ off-diagonal pairs (with $n = |\mathcal{B}|$ valid
samples):

$$
\boxed{\;\mathcal{L}^{t}_{\text{mag}}
= \frac{1}{n(n-1)}\sum_{\substack{(i,j)\in\mathcal{B}^2 \\ i \ne j}}
\operatorname{ReLU}\!\Big( \tfrac{\lvert y_i - y_j\rvert}{s} - g\,\lVert z_i - z_j\rVert_2 \Big)\;}
$$

(Pairs with equal labels contribute $\operatorname{ReLU}(0 - g\lVert\cdot\rVert)=0$, so they are harmless.)

---

## 6. Aggregation and the total training objective

**Per target:** the two terms are combined with equal weight (both are already
$O(1)$ hinges):

$$
\mathcal{L}^{t}_{\text{ordcon}} = \mathcal{L}^{t}_{\text{dir}} + \mathcal{L}^{t}_{\text{mag}}.
$$

**Across targets:** we sum, and — crucially — only over targets that are
**present** in the batch. A target with fewer than two labelled speakers
(e.g. age on a HeightCeleb-only batch) is skipped, so a single model trains
across corpora with different available labels:

$$
\mathcal{L}_{\text{ordcon}} = \sum_{t\,:\,\sum_i m_i^t \ge 2} \mathcal{L}^{t}_{\text{ordcon}} .
$$

**Total loss:** OrdCon is a regulariser on the geometry, added to the masked
regression loss with weight $\lambda$ (`ordcon_weight`):

$$
\boxed{\;\mathcal{L} = \mathcal{L}_{\text{reg}} + \lambda\,\mathcal{L}_{\text{ordcon}}\;}
$$

---

## 7. Gradient analysis — *why the terms do the right thing*

**Direction term.** For a violated pair ($\ell^{\text{dir}}_{ij} > 0$) with
$\sigma = \operatorname{sign}(y_i - y_j)$:

$$
\frac{\partial \ell^{\text{dir}}_{ij}}{\partial z_i} = -\sigma\, u_t,
\qquad
\frac{\partial \ell^{\text{dir}}_{ij}}{\partial z_j} = +\sigma\, u_t,
\qquad
\frac{\partial \ell^{\text{dir}}_{ij}}{\partial u_t} = -\sigma\,(z_i - z_j).
$$

A gradient-descent step therefore moves $z_i \leftarrow z_i + \eta\,\sigma\, u_t$
and $z_j \leftarrow z_j - \eta\,\sigma\, u_t$: if $y_i > y_j$ ($\sigma=+1$) the
higher-label embedding slides in $+u_t$ and the lower-label one in $-u_t$ — they
**separate in the correct order**. Meanwhile $u_t \leftarrow u_t + \eta\,\sigma\,(z_i - z_j)$
rotates the axis toward the direction pointing from low-label to high-label
embeddings — $u_t$ **learns the data's true label-increasing direction.**

**Magnitude term.** For a violated pair:

$$
\frac{\partial \ell^{\text{mag}}_{ij}}{\partial z_i} = -\,g\,\frac{z_i - z_j}{\lVert z_i - z_j\rVert}.
$$

The step $z_i \leftarrow z_i + \eta\,g\,\frac{z_i - z_j}{\lVert z_i - z_j\rVert}$
pushes $z_i$ **away** from $z_j$ — increasing the distance of pairs that are too
close given their label gap, exactly as (P2) demands.

---

## 8. Worked example (illustrating both terms)

Three speakers, ages $y = (25, 30, 40)$, and (for illustration) current axis
positions $s = \langle z, u_t\rangle = (0.2,\, 0.1,\, 0.5)$, with margin $m = 0.1$.

- **Pair (1,2):** $y_1 < y_2$ but $s_1 > s_2$ — an **order violation**.
  $\sigma = \operatorname{sign}(25-30) = -1$, $\;\langle z_1-z_2,u_t\rangle = s_1 - s_2 = 0.1$.
  $\ell^{\text{dir}}_{12} = \operatorname{ReLU}(0.1 - (-1)(0.1)) = \operatorname{ReLU}(0.2) = 0.2 > 0$ → **penalised**. ✔
- **Pair (2,3):** $y_2 < y_3$ and $s_2 < s_3$ — order correct.
  $\sigma=-1,\;\langle z_2-z_3,u_t\rangle = 0.1-0.5=-0.4$.
  $\ell^{\text{dir}}_{23} = \operatorname{ReLU}(0.1 - (-1)(-0.4)) = \operatorname{ReLU}(-0.3)=0$ → **satisfied**. ✔

Magnitude, pair (1,3): $\lvert\Delta y\rvert = 15$, $s = 5 \Rightarrow d^\star = 3$.
If $\lVert z_1 - z_3\rVert = 0.8$, $g=1$: $\ell^{\text{mag}}_{13} = \operatorname{ReLU}(3 - 0.8) = 2.2 > 0$ → the pair is pushed apart. (See §9 for why $d^\star = 3$ here is unreachable.)

---

## 9. Assumptions, design choices, and limitations (stated honestly)

1. **$\ell_2$-normalisation bounds distances, which saturates the magnitude term.**
   Because all $z_i$ lie on the unit sphere, $\lVert z_i - z_j\rVert \in [0, 2]$.
   But $d^\star_{ij} = \lvert\Delta y\rvert / s$ can exceed 2 (e.g. an age gap of
   $50$ with $s=5$ gives $d^\star = 10$). For any pair with a large label gap the
   constraint $g\lVert\cdot\rVert \ge d^\star$ is **unsatisfiable**, so
   $\ell^{\text{mag}}$ saturates to a near-constant and provides little useful
   gradient there. The magnitude term therefore acts mostly on **small-gap**
   pairs. *Fix:* replace $d^\star$ with a bounded target, e.g.
   $d^\star = 2\tanh(\lvert\Delta y\rvert / s)$, or use an angular/cosine distance
   consistent with the retrieval metric.
2. **Single linear direction $u_t$.** (P1) is enforced along one global axis,
   assuming the label manifold is locally linear. This is simple and sufficient
   for *local* kNN retrieval, but a curved label manifold is not fully captured.
3. **Shared scale $s$.** A single $s = 5$ approximates the per-target label
   spread. A more principled choice sets $s$ to each target's label standard
   deviation (age std $\approx 5$–6 yr, height std $\approx 9$ cm).
4. **Margin-ranking vs. contrastive.** This formulation is a *margin-ranking*
   relaxation chosen for stability under small mini-batches and continuous
   labels; it is not the InfoNCE-style softmax-contrastive form of the original
   OrdCon paper (see §10).

---

## 10. Relation to prior work

The two governing principles — **order preservation** and **magnitude
alignment** — are those of Wang et al. (2025, IEEE TIFS), *Order-Enhanced
Contrastive Learning for Face Age Estimation*, which shapes an ordinal embedding
for age. Our contribution here is the **adaptation of those principles to (a)
scalar, continuous speech-trait targets and (b) a multi-target, partially-labelled
setting**, recast as differentiable margin constraints with a learnable per-target
axis. It differs from the triplet loss used by Kaushik et al. (2021) for the same
task, which treats age/height as **discrete classes** and thereby discards the
ordinal structure that (P1)–(P2) explicitly preserve.

---

## 11. Hyperparameters as used

| Symbol | Config key | Default | Role |
|---|---|---:|---|
| $m$ | `ordcon_kwargs.margin` | $0.1$ | Order margin along $u_t$ |
| $g$ | `ordcon_kwargs.magnitude_gain` | $1.0$ | Distance-vs-gap tightness |
| $s$ | `ordcon_kwargs.magnitude_scale` | $5.0$ | Label-gap normaliser (units) |
| $\lambda$ | `loss.ordcon_weight` | $0.3$ | Weight of OrdCon vs. regression |
| $D$ | `projector.output_dim` | $256$ | Embedding / direction dimension |

---

## 12. Algorithm (matches the implementation)

```
Input: batch embeddings Z = [z_1..z_n] (L2-normalised), labels {y^t}, masks {m^t}
for each target t with >= 2 valid labels:
    z, y  <- rows where m^t = 1
    u     <- normalize(direction[t])                 # learnable unit vector
    dz    <- z_i - z_j    for all i,j                 # (n,n,D)
    dy    <- y_i - y_j    for all i,j                 # (n,n)
    # direction term
    proj  <- <dz, u>                                  # (n,n)
    Ldir  <- mean over {(i,j): dy!=0} of ReLU(m - sign(dy) * proj)
    # magnitude term
    dist  <- ||dz||                                   # (n,n)
    Lmag  <- mean over {i!=j} of ReLU(|dy|/s - dist * g)
    L_t   <- Ldir + Lmag
L_ordcon <- sum_t L_t
return L_ordcon
```

Total objective: $\mathcal{L} = \mathcal{L}_{\text{reg}} + \lambda\,\mathcal{L}_{\text{ordcon}}$.
