"""Training on several corpora at once (HeightCeleb + NISP, tested on TIMIT) and the two runs' configs.

The combined set must be the union of each corpus's own split (each corpus read by its own loader),
validation must be HeightCeleb dev only, speaker IDs must not repeat across corpora, and the two
experiments must differ from the base setting in their training data alone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from retro_profile.data import MultiCorpusDataModule
from retro_profile.utils.config import instantiate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _manifests(root, splits, **cols):
    (root / "manifests").mkdir(parents=True)
    for split, speakers in splits.items():
        rows = [{"speaker_id": s, "utt_id": f"{s}_{i}", "audio_path": f"{s}_{i}.wav",
                 "age": 30.0, "height": 170.0, "gender": "F", **cols} for s in speakers for i in range(2)]
        pd.DataFrame(rows).to_csv(root / "manifests" / f"{split}.csv", index=False)


@pytest.fixture
def roots(tmp_path):
    hc, nisp = tmp_path / "heightceleb", tmp_path / "nisp"
    _manifests(hc, {"train": ["id10001", "id10002"], "dev": ["id10003"], "test": ["id10041"]},
               video_id="vidAAAAAAAA")
    _manifests(nisp, {"train": ["Hin_0001"], "dev": ["Hin_0002"], "test": ["Tam_0001"]},
               language="English")
    return hc, nisp


def _dm(roots, **overrides):
    """configs/data/heightceleb_nisp.yaml, instantiated the way train.py does, on tiny manifests."""
    cfg = yaml.safe_load((CONFIGS / "data" / "heightceleb_nisp.yaml").read_text())
    cfg["corpora"]["heightceleb"]["root"], cfg["corpora"]["nisp"]["root"] = map(str, roots)
    cfg.update(num_workers=0, **overrides)
    return instantiate(cfg)


def test_union_of_each_corpus_split_and_validation_is_heightceleb_dev(roots):
    dm = _dm(roots)
    assert isinstance(dm, MultiCorpusDataModule)
    dm.setup("fit")
    assert dm.train_set.speakers == ["Hin_0001", "Tam_0001", "id10001", "id10002", "id10041"]
    assert dm.val_set.speakers == ["id10003"]
    assert set(dm.train_set.manifest.corpus) == {"heightceleb", "nisp"}
    assert dm.train_set.manifest.audio_path.iloc[0] == str(roots[0] / "id10001_0.wav")
    assert dm.label_stats["height"]["count"] == 5 and dm.train_set.mode == "bag"
    dm.setup("test")
    assert dm.test_set.speakers == ["id10003"]


def test_utterance_loader_roles(roots):
    dm = _dm(roots)
    assert len(dm.utterance_loader("train").dataset) == 10
    assert len(dm.utterance_loader("dev").dataset) == len(dm.utterance_loader("val").dataset) == 2
    with pytest.raises(ValueError, match="split must be one of"):
        dm.utterance_loader("holdout")


def test_nisp_options_pass_through(roots):
    cfg = {"heightceleb": {"root": str(roots[0]), "splits": {"train": "train", "val": "dev"}},
           "nisp": {"root": str(roots[1]), "languages": ["klingon"], "splits": {"train": "train"}}}
    with pytest.raises(ValueError, match="matched none"):
        MultiCorpusDataModule(corpora=cfg, num_workers=0).setup("fit")


def test_repeated_speaker_across_corpora_is_refused(tmp_path):
    for name in ("a", "b"):
        _manifests(tmp_path / name, {"train": ["id10001"], "dev": ["id10009"], "test": []})
    corpora = {"heightceleb": {"root": str(tmp_path / "a"), "splits": {"train": "train", "val": "dev"}},
               "timit": {"root": str(tmp_path / "b"), "splits": {"train": "train"}}}
    with pytest.raises(ValueError, match="appears in both"):
        MultiCorpusDataModule(corpora=corpora, num_workers=0).setup("fit")


def test_unknown_corpus_and_empty_role_are_refused(roots):
    with pytest.raises(ValueError, match="unknown corpora"):
        MultiCorpusDataModule(corpora={"voxceleb2": {"root": "x", "splits": {"train": "train"}}})
    dm = MultiCorpusDataModule(corpora={"heightceleb": {"root": str(roots[0]),
                                                        "splits": {"train": "train"}}}, num_workers=0)
    with pytest.raises(ValueError, match="no corpus supplies the 'val' role"):
        dm.setup("fit")


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out.update(_flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def _cfg(experiment):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=os.fspath(CONFIGS), version_base=None):
        c = compose(config_name="default", overrides=[f"experiment={experiment}"])
    return _flat(OmegaConf.to_container(c, resolve=True))


def test_run_a_changes_only_the_training_data():
    base, a = _cfg("retro_ordcon_fixed_l100_s5"), _cfg("heightceleb_full_l100_s5")
    changed = {k for k in set(base) | set(a) if base.get(k) != a.get(k)}
    assert changed == {"data._target_", "data.name", "data.root", "data.splits.train",
                       "data.splits.test", "run_name", "output_dir"}
    assert a["data.root"] == "data/heightceleb" and a["data.splits.train"] == "train+test"
    assert a["data.splits.val"] == "dev"


def test_run_b_adds_nisp_and_nothing_else():
    base, a, b = (_cfg("retro_ordcon_fixed_l100_s5"), _cfg("heightceleb_full_l100_s5"),
                  _cfg("heightceleb_nisp_l100_s5"))
    changed = {k for k in set(base) | set(b) if base.get(k) != b.get(k)}
    assert all(k.startswith("data.") or k in ("run_name", "output_dir") for k in changed)
    for k in ("batch_size", "bag_size", "max_audio_seconds", "min_audio_seconds", "sample_rate",
              "augmentation.enabled", "targets", "normalize_targets"):
        assert b[f"data.{k}"] == base[f"data.{k}"] == a[f"data.{k}"], k
    assert b["data.max_audio_seconds"] == 8.0 and b["data.batch_size"] == 4
    assert b["data.corpora.heightceleb.root"] == a["data.root"]
    assert b["data.corpora.heightceleb.splits.train"] == a["data.splits.train"] == "train+test"
    assert b["data.corpora.heightceleb.splits.val"] == a["data.splits.val"] == "dev"
    assert b["data.corpora.nisp.splits.train"] == _cfg("nisp_full_l100_s5")["data.splits.train"]
    assert b["data.corpora.nisp.root"] == "data/nisp" and b["data.corpora.nisp.splits.val"] is None


# ---- joint TIMIT + HeightCeleb + NISP -> TIMIT (plan step 3b, 2026-10-12) ---------------------------

def test_joint_data_validates_on_timit_dev_only(roots, tmp_path):
    timit = tmp_path / "timit"
    _manifests(timit, {"train": ["FAKS0", "MABC0"], "dev": ["FDEV0"], "test": ["MTST0"]})
    cfg = yaml.safe_load((CONFIGS / "data" / "timit_heightceleb_nisp.yaml").read_text())
    cfg["corpora"]["timit"]["root"] = str(timit)
    cfg["corpora"]["heightceleb"]["root"], cfg["corpora"]["nisp"]["root"] = map(str, roots)
    cfg.update(num_workers=0)
    dm = instantiate(cfg)
    dm.setup("fit")
    dm.setup("test")
    assert dm.val_set.speakers == ["FDEV0"] and dm.test_set.speakers == ["MTST0"]
    assert dm.train_set.speakers == ["FAKS0", "Hin_0001", "MABC0", "Tam_0001", "id10001", "id10002", "id10041"]


def test_joint_runs_add_timit_to_run_b_and_change_only_the_data():
    base, joint, b = (_cfg("retro_ordcon_fixed_l100_s5"), _cfg("timit_hc_nisp_l100_s5"),
                      _cfg("heightceleb_nisp_l100_s5"))
    changed = {k for k in set(base) | set(joint) if base.get(k) != joint.get(k)}
    assert all(k.startswith("data.") or k in ("run_name", "output_dir") for k in changed)
    for k in ("batch_size", "bag_size", "max_audio_seconds", "augmentation.enabled", "targets"):
        assert joint[f"data.{k}"] == base[f"data.{k}"], k
    for k in [k for k in b if k.startswith("data.corpora.")]:                 # HeightCeleb + NISP as in Run B ...
        if ".splits.val" not in k and ".splits.test" not in k:
            assert joint[k] == b[k], k
    assert joint["data.corpora.timit.root"] == base["data.root"] == "data/timit"   # ... plus TIMIT's own split
    assert joint["data.corpora.timit.splits.val"] == base["data.splits.val"] == "dev"
    assert joint["data.corpora.heightceleb.splits.val"] is None and joint["data.corpora.nisp.splits.val"] is None
    for name, timit_cfg in (("timit_hc_nisp_gupta_wavlm", "gupta2022_finetune_wavlm"), ("timit_hc_nisp_jaid", "jaid2023")):
        new, old = _cfg(name), _cfg(timit_cfg)
        assert all(k.startswith("data.") or k in ("run_name", "output_dir")
                   for k in set(new) | set(old) if new.get(k) != old.get(k)), name
        assert new["data.utts_per_speaker"] == 8 and new["data.train_unit"] == old["data.train_unit"]
        assert {k: v for k, v in new.items() if k.startswith("data.corpora.")} == \
            {k: v for k, v in joint.items() if k.startswith("data.corpora.")}


def test_attention_runs_change_only_the_attention_on_run_b():
    """heightceleb_nisp_hier / _temporal (13 Oct 2026): Run B plus frame attention (own LR); temporal also
    averages clips instead of MIL."""
    b = _cfg("heightceleb_nisp_l100_s5")
    expected = {"model.encoder.frame_pooling", "model.encoder.frame_attention_hidden",
                "model.encoder.frame_attention_dropout", "training.frame_attention_lr", "run_name", "output_dir"}
    hier, temporal = _cfg("heightceleb_nisp_hier_l100_s5"), _cfg("heightceleb_nisp_temporal_l100_s5")
    assert {k for k in set(b) | set(hier) if b.get(k) != hier.get(k)} == expected
    assert {k for k in set(b) | set(temporal) if b.get(k) != temporal.get(k)} == expected | {"model.pooling._target_"}
    assert hier["model.encoder.frame_pooling"] == "attention" and hier["training.frame_attention_lr"] == 1e-3
    assert temporal["model.pooling._target_"] == "retro_profile.models.MeanPooling"


@pytest.mark.parametrize("experiment, key, value", [
    ("heightceleb_nisp_enc1e6_l100_s5", "training.encoder_lr", 1e-6),
    ("heightceleb_nisp_enc3e6_l100_s5", "training.encoder_lr", 3e-6),
    ("heightceleb_nisp_enc3e5_l100_s5", "training.encoder_lr", 3e-5),
    ("heightceleb_nisp_l030_s5", "training.loss.ordcon_weight", 0.3),
    ("heightceleb_nisp_l300_s5", "training.loss.ordcon_weight", 3.0),
    ("heightceleb_nisp_l1000_s5", "training.loss.ordcon_weight", 10.0),
])
def test_tuning_runs_change_one_value_of_run_b(experiment, key, value):
    """14 Oct 2026 tuning grid: each run is Run B with exactly one value changed (plus its own run name)."""
    b, new = _cfg("heightceleb_nisp_l100_s5"), _cfg(experiment)
    assert {k for k in set(b) | set(new) if b.get(k) != new.get(k)} == {key, "run_name", "output_dir"}
    assert new[key] == value and b[key] in (1e-5, 1.0)
    assert new["run_name"] == f"{experiment}_heightceleb_nisp"   # slurm sets <experiment>_timit_s42


def test_combined_tuning_run_changes_exactly_lr_and_lambda():
    """14 Oct 2026: the combined run = Run B with the winning WavLM LR (1e-6) AND OrdCon lambda (3.0), nothing else."""
    b, new = _cfg("heightceleb_nisp_l100_s5"), _cfg("heightceleb_nisp_enc1e6_l300_s5")
    assert {k for k in set(b) | set(new) if b.get(k) != new.get(k)} == {
        "training.encoder_lr", "training.loss.ordcon_weight", "run_name", "output_dir"}
    assert new["training.encoder_lr"] == 1e-6 and new["training.loss.ordcon_weight"] == 3.0


def test_timit_lambda3_and_hierarchical_change_only_their_part_of_the_base():
    """15 Oct 2026: TIMIT base + lambda 3.0; TIMIT base + frame attention with its own LR (as E34)."""
    base = _cfg("retro_ordcon_fixed_l100_s5")
    l3, hier = _cfg("retro_ordcon_fixed_l300_s5"), _cfg("retro_hier_falr1e3_l100_s5")
    assert {k for k in set(base) | set(l3) if base.get(k) != l3.get(k)} == {
        "training.loss.ordcon_weight", "run_name", "output_dir"}
    assert l3["training.loss.ordcon_weight"] == 3.0
    assert {k for k in set(base) | set(hier) if base.get(k) != hier.get(k)} == {
        "model.encoder.frame_pooling", "model.encoder.frame_attention_hidden", "model.encoder.frame_attention_dropout",
        "training.frame_attention_lr", "run_name", "output_dir"}
    assert hier["model.encoder.frame_pooling"] == "attention" and hier["training.frame_attention_lr"] == 1e-3
    assert base["data.name"] == l3["data.name"] == hier["data.name"] == "timit"


def test_fairness_checks_change_only_clips_per_group_or_gupta_lr():
    """15 Oct 2026: RETRO on groups of all 10 TIMIT clips; Gupta at flat LR 3e-6 / 1e-5 (paper: 1e-6)."""
    base, b10 = _cfg("retro_ordcon_fixed_l100_s5"), _cfg("retro_ordcon_fixed_l100_s5_b10")
    assert {k for k in set(base) | set(b10) if base.get(k) != b10.get(k)} == {"data.bag_size", "run_name", "output_dir"}
    assert (base["data.bag_size"], b10["data.bag_size"]) == (8, 10)
    g = _cfg("gupta2022_finetune_wavlm")
    for tag, lr in (("3e6", 3e-6), ("1e5", 1e-5)):
        new = _cfg(f"gupta2022_finetune_wavlm_lr{tag}")
        assert {k for k in set(g) | set(new) if g.get(k) != new.get(k)} == {
            "training.gupta.encoder_lr", "training.gupta.head_lr", "run_name", "output_dir"}
        assert new["training.gupta.encoder_lr"] == new["training.gupta.head_lr"] == lr
    assert g["training.gupta.encoder_lr"] == g["training.gupta.head_lr"] == 1e-6


def test_gupta_recipe_b_takes_retros_recipe_and_nothing_else():
    """16 Oct 2026: Gupta's model with RETRO's final TIMIT recipe (LRs, 40 clips per step, 60 epochs)."""
    a, b = _cfg("gupta2022_finetune_wavlm"), _cfg("gupta2022_finetune_wavlm_retrorecipe")
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "data.bag_size", "training.accumulate_grad_batches", "training.max_epochs", "training.gupta.encoder_lr",
        "training.gupta.head_lr", "run_name", "output_dir"}
    retro = _cfg("retro_ordcon_fixed_l100_s5_b10")
    assert b["data.batch_size"] * b["training.accumulate_grad_batches"] == \
        retro["data.batch_size"] * retro["data.bag_size"] * retro["training.accumulate_grad_batches"] == 40
    assert b["data.batch_size"] == 8                     # 40 clips at once ran out of GPU memory (job 61051631)
    assert b["training.max_epochs"] == retro["training.max_epochs"] == 60
    assert (b["training.gupta.encoder_lr"], b["training.gupta.head_lr"]) == (
        retro["training.encoder_lr"], retro["training.optimizer.lr"])
    assert b["data.targets"] == ["age", "height", "gender"] and b["training.early_stopping.enabled"] is False


def test_jaid_recipes_change_only_learning_rate_or_take_retros_recipe():
    """16 Oct 2026: Jaid recipe A at LR 3e-4 / 3e-3 (paper 1e-3); recipe B = RETRO's recipe."""
    a = _cfg("jaid2023")
    assert (a["training.baseline.head_lr"], a["training.max_epochs"], a["training.early_stopping.enabled"]) == (1e-3, 200, True)
    for tag, lr in (("lr3e4", 3e-4), ("lr3e3", 3e-3), ("lr1e2", 1e-2)):
        new = _cfg(f"jaid2023_{tag}")
        assert {k for k in set(a) | set(new) if a.get(k) != new.get(k)} == {
            "training.baseline.head_lr", "training.baseline.encoder_lr", "run_name", "output_dir"}
        assert new["training.baseline.head_lr"] == new["training.baseline.encoder_lr"] == lr
    b, retro = _cfg("jaid2023_retrorecipe"), _cfg("retro_ordcon_fixed_l100_s5_b10")
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "data.batch_size", "training.max_epochs", "training.early_stopping.enabled", "training.baseline.head_lr",
        "training.baseline.encoder_lr", "training.baseline.weight_decay", "run_name", "output_dir"}
    assert b["data.batch_size"] == retro["data.batch_size"] * retro["data.bag_size"] == 40
    assert (b["training.max_epochs"], b["training.early_stopping.enabled"]) == (60, False)
    assert b["training.baseline.head_lr"] == retro["training.optimizer.lr"] == 1e-4
    assert b["training.baseline.weight_decay"] == retro["training.optimizer.weight_decay"] == 0.01


def test_wavlm_mlp_baseline_is_final_retro_minus_its_three_parts():
    """16 Oct 2026: WavLM + MLP = RETRO's final setting (groups of 10) without MIL, OrdCon and kNN; LR check 3e-6 / 3e-5."""
    retro, mlp = _cfg("retro_ordcon_fixed_l100_s5_b10"), _cfg("wavlm_direct_budget_b10")
    assert {k for k in set(retro) | set(mlp) if retro.get(k) != mlp.get(k)} == {
        "model.pooling._target_", "training.loss.ordcon_weight", "run_name", "output_dir"}
    assert mlp["model.pooling._target_"] == "retro_profile.models.MeanPooling" and mlp["training.loss.ordcon_weight"] == 0.0
    assert mlp["data.bag_size"] == 10 and mlp["model.heads.age.use_retrieval"] is False
    for tag, lr in (("enc3e6", 3e-6), ("enc3e5", 3e-5)):
        new = _cfg(f"wavlm_direct_budget_b10_{tag}")
        assert {k for k in set(mlp) | set(new) if mlp.get(k) != new.get(k)} == {"training.encoder_lr", "run_name", "output_dir"}
        assert new["training.encoder_lr"] == lr


def test_table2_ablations_remove_exactly_one_part_of_final_retro():
    """16 Oct 2026: '- MIL' = plain average over clips; '- OrdCon' = OrdCon weight 0; nothing else changes."""
    full = _cfg("retro_ordcon_fixed_l100_s5_b10")
    nomil, noord = _cfg("retro_ordcon_fixed_l100_s5_b10_nomil"), _cfg("retro_ordcon_fixed_l100_s5_b10_noordcon")
    assert {k for k in set(full) | set(nomil) if full.get(k) != nomil.get(k)} == {
        "model.pooling._target_", "run_name", "output_dir"}
    assert nomil["model.pooling._target_"] == "retro_profile.models.MeanPooling"
    assert {k for k in set(full) | set(noord) if full.get(k) != noord.get(k)} == {
        "training.loss.ordcon_weight", "run_name", "output_dir"}
    assert noord["training.loss.ordcon_weight"] == 0.0 and full["training.loss.ordcon_weight"] == 1.0


def test_wavlm_mlp_convergence_check_changes_only_the_epoch_count():
    """16 Oct 2026: WavLM + MLP at 100 epochs (its 60-epoch validation curve was still falling)."""
    base, e100 = _cfg("wavlm_direct_budget_b10"), _cfg("wavlm_direct_budget_b10_e100")
    assert {k for k in set(base) | set(e100) if base.get(k) != e100.get(k)} == {"training.max_epochs", "run_name", "output_dir"}
    assert (base["training.max_epochs"], e100["training.max_epochs"]) == (60, 100)


@pytest.mark.parametrize("new, table1", [
    ("heightceleb_nisp_b10_l100_s5", "retro_ordcon_fixed_l100_s5_b10"),
    ("heightceleb_nisp_mlp_b10_e100", "wavlm_direct_budget_b10_e100"),
    ("heightceleb_nisp_gupta_lr3e6", "gupta2022_finetune_wavlm_lr3e6"),
    ("heightceleb_nisp_jaid_lr3e3", "jaid2023_lr3e3"),
])
def test_table5_models_keep_their_table1_settings_and_change_only_the_data(new, table1):
    """16 Oct 2026: Table 5 trains each final Table 1 model on HeightCeleb + NISP; nothing outside data.* changes."""
    a, b = _cfg(new), _cfg(table1)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert all(k.startswith("data.") or k in ("run_name", "output_dir") for k in changed), changed
    assert a["data._target_"] == "retro_profile.data.MultiCorpusDataModule"
    assert a["data.corpora.heightceleb.splits.train"] == "train+test" and a["data.corpora.nisp.splits.train"] == "train+test"
    assert a["data.batch_size"] == b["data.batch_size"] and a.get("data.train_unit", "bag") == b.get("data.train_unit", "bag")
    if b.get("data.train_unit") == "utterance":
        assert a["data.utts_per_speaker"] == 10                    # same audio per epoch as RETRO's groups of 10
    else:
        assert a["data.bag_size"] == 10


# ---- TABLE 4: TIMIT + NISP (4a) / + age-balanced HeightCeleb sample (4b) -> HeightCeleb split2 (2026-10-17) ----------

def test_table4_data_trains_on_timit_and_nisp_and_validates_on_timit_dev(roots, tmp_path):
    timit, sample = tmp_path / "timit", tmp_path / "heightceleb_split2_age_sample"
    _manifests(timit, {"train": ["FAKS0", "MABC0"], "dev": ["FDEV0"], "test": ["MTST0"]})
    _manifests(sample, {"train": ["id10002"]}, video_id="vidAAAAAAAA")
    for name, extra in (("timit_nisp", []), ("timit_nisp_hcsample", ["id10002"])):
        cfg = yaml.safe_load((CONFIGS / "data" / f"{name}.yaml").read_text())
        cfg["corpora"]["timit"]["root"], cfg["corpora"]["nisp"]["root"] = str(timit), str(roots[1])
        if "heightceleb" in cfg["corpora"]:
            cfg["corpora"]["heightceleb"]["root"] = str(sample)
        dm = instantiate({**cfg, "num_workers": 0})
        dm.setup("fit")
        dm.setup("test")
        assert dm.train_set.speakers == sorted(["FAKS0", "MABC0", "Hin_0001", "Tam_0001"] + extra), name
        assert dm.val_set.speakers == ["FDEV0"] and dm.test_set.speakers == ["FDEV0"], name   # no test speaker touched


@pytest.mark.parametrize("new, table1", [
    ("timit_nisp_b10_l100_s5", "retro_ordcon_fixed_l100_s5_b10"),
    ("timit_nisp_mlp_b10_e100", "wavlm_direct_budget_b10_e100"),
    ("timit_nisp_gupta_lr3e6", "gupta2022_finetune_wavlm_lr3e6"),
    ("timit_nisp_jaid_lr3e3", "jaid2023_lr3e3"),
])
def test_table4_models_keep_their_table1_settings_and_change_only_the_data(new, table1):
    """17 Oct 2026: Table 4 trains each final Table 1 model on TIMIT + NISP (4a) and on that plus the HeightCeleb sample
    (4b); nothing outside data.* changes, and 4b differs from 4a only in the HeightCeleb corpus entry."""
    a, b = _cfg(new), _cfg(table1)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert all(k.startswith("data.") or k in ("run_name", "output_dir") for k in changed), changed
    assert a["data._target_"] == "retro_profile.data.MultiCorpusDataModule" and a["data.name"] == "timit_nisp"
    assert a["data.corpora.timit.splits.train"] == "train" and a["data.corpora.timit.splits.val"] == "dev"
    assert a["data.corpora.nisp.splits.train"] == "train+test" and a["data.corpora.nisp.splits.val"] is None
    for k in ("batch_size", "max_audio_seconds", "augmentation.enabled", "targets", "normalize_targets"):
        assert a[f"data.{k}"] == b[f"data.{k}"], k
    assert a.get("data.train_unit", "bag") == b.get("data.train_unit", "bag")
    if b.get("data.train_unit") == "utterance":
        assert a["data.utts_per_speaker"] == 10                    # same audio per epoch as RETRO's groups of 10
    else:
        assert a["data.bag_size"] == 10
    hcs = _cfg(new.replace("timit_nisp_", "timit_nisp_hcs_"))
    diff = {k for k in set(a) | set(hcs) if a.get(k) != hcs.get(k)}
    assert diff == {"data.name", "run_name", "output_dir", "data.corpora.heightceleb.root",
                    "data.corpora.heightceleb.splits.train"}, diff          # val / test null: no HeightCeleb role
    assert hcs["data.corpora.heightceleb.root"] == "data/heightceleb_split2_age_sample"
    assert hcs["data.corpora.heightceleb.splits.train"] == "train"
    assert hcs["data.corpora.heightceleb.splits.val"] is None and hcs["data.corpora.heightceleb.splits.test"] is None
