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


# ---- joint TIMIT + HeightCeleb + NISP -> TIMIT (plan step 3b, 2026-10-12) ---------------------------

def test_wavlm_mlp_baseline_is_final_retro_minus_its_three_parts():
    """16 Oct 2026: WavLM + MLP = RETRO's final setting (groups of 10) without MIL, OrdCon and kNN."""
    retro, mlp = _cfg("retro_ordcon_fixed_l100_s5_b10"), _cfg("wavlm_direct_budget_b10")
    assert {k for k in set(retro) | set(mlp) if retro.get(k) != mlp.get(k)} == {
        "model.pooling._target_", "training.loss.ordcon_weight", "run_name", "output_dir"}
    assert mlp["model.pooling._target_"] == "retro_profile.models.MeanPooling" and mlp["training.loss.ordcon_weight"] == 0.0
    assert mlp["data.bag_size"] == 10 and mlp["model.heads.age.use_retrieval"] is False


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
