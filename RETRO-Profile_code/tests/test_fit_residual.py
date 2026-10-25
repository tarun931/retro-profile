"""scripts/fit_residual.py: the second stage that trains ONLY each trait's residual correction on top of kNN.

Pins: only the residual's weights move; epoch 0 is kNN alone, so validation can never get worse; the script's
formula equals the head's own forward with the datastore attached (what evaluate.py runs); training estimates leave
each speaker out of its own lookup; k is chosen on validation; the reproduction check stops on a mismatch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("faiss")

from retro_profile.models.heads.mlp import MLPHead  # noqa: E402
from retro_profile.models.heads.residual import ResidualRetrievalHead  # noqa: E402
from retro_profile.models.pooling import MILAttentionPooling  # noqa: E402
from retro_profile.models.retro_profile import RetroProfile  # noqa: E402
from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model import TinyRandomEncoder  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fit_residual", _ROOT / "scripts" / "fit_residual.py")
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

D = 6


def _unit(rng, n):
    v = rng.randn(n, D).astype(np.float32)
    return torch.from_numpy(v / np.linalg.norm(v, axis=1, keepdims=True))


def _datastore(n=60, seed=0):
    rng = np.random.RandomState(seed)
    ds = RetrievalDatastore(embedding_dim=D, metric="cosine", temperature=0.1)
    q = _unit(rng, n)
    for i in range(n):
        ds.add(DatastoreEntry(speaker_id=f"T{i}", embedding=q[i].numpy(), labels={"age": 20.0 + i * 0.5}))
    ds.build()
    return ds, q


def _head():
    torch.manual_seed(0)
    return ResidualRetrievalHead(input_dim=D, hidden_dim=16, label_min=18.0, label_max=80.0, delta_clip=6.0)


def _biased(n, seed):
    """z, a kNN estimate that reads 4 yr too old when z[:, 0] > 0 and 4 too young otherwise, and the truth."""
    rng = np.random.RandomState(seed)
    z = _unit(rng, n)
    y = torch.from_numpy(rng.uniform(25, 60, n).astype(np.float32))
    a = y + 4.0 * torch.sign(z[:, 0])
    return z, a, y


def test_only_the_residual_is_trained_and_it_removes_a_learnable_bias():
    head = _head()
    direct_before = [p.clone() for p in head.direct.parameters()]
    res_before = [p.clone() for p in head.residual.parameters()]
    (z_tr, a_tr, y_tr), (z_va, a_va, y_va) = _biased(300, 1), _biased(70, 2)
    fit = fr.train_residual(head, z_tr, a_tr, y_tr, z_va, a_va, y_va, y_scale=10.0, epochs=150, lr=3e-3,
                            patience=150)
    assert all(torch.equal(a, b) for a, b in zip(direct_before, head.direct.parameters(), strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(res_before, head.residual.parameters(), strict=True))
    assert fit["knn_val_mae"] == pytest.approx(4.0, abs=1e-4)
    assert fit["best_epoch"] > 0 and fit["best_val_mae"] < 2.0
    assert not any(p.requires_grad for p in head.residual.parameters())          # frozen again afterwards


def test_epoch_zero_is_knn_alone_so_validation_never_gets_worse():
    head = _head()
    z, a, y = _biased(50, 3)
    fit = fr.train_residual(head, z, a, y, z, a, a.clone(), y_scale=10.0, epochs=20)   # val truth = estimate
    assert fit["best_epoch"] == 0 and fit["best_val_mae"] == 0.0
    assert torch.equal(fr.residual_predict(head, z, a), a.clamp(18.0, 80.0))       # correction exactly 0


def test_script_formula_equals_the_heads_own_forward_with_the_datastore():
    ds, _ = _datastore()
    head = _head()
    with torch.no_grad():
        head.residual[-1].weight.normal_()                                        # a non-zero correction
    rng = np.random.RandomState(5)
    split = {"speaker_ids": [f"V{i}" for i in range(9)], "query": _unit(rng, 9)}
    z = _unit(rng, 9)
    head.set_datastore(ds, "age")
    head.knn_k = 7
    head.eval()                                                                   # scoring is always eval mode
    with torch.no_grad():
        want = head(z, retrieval_emb=split["query"], speaker_ids=split["speaker_ids"])
        got = fr.residual_predict(head, z, fr.knn_estimate(ds, split, "age", 7))
    assert torch.allclose(got, want, atol=1e-5)


def test_training_estimates_leave_each_speaker_out():
    ds, q = _datastore()
    own = {"speaker_ids": [f"T{i}" for i in range(60)], "query": q}
    stranger = {"speaker_ids": [f"X{i}" for i in range(60)], "query": q}
    labels = torch.tensor([20.0 + i * 0.5 for i in range(60)])
    assert torch.allclose(fr.knn_estimate(ds, stranger, "age", 1), labels)        # finds itself when not excluded
    assert not torch.allclose(fr.knn_estimate(ds, own, "age", 1), labels)          # never its own label


def test_k_is_chosen_on_validation_with_ties_to_the_smaller_k():
    ds, q = _datastore()
    val = {"speaker_ids": [f"X{i}" for i in range(60)], "query": q,
           "truth": {"age": torch.tensor([20.0 + i * 0.5 for i in range(60)])}}
    k, table = fr.choose_k(ds, val, "age", (18.0, 80.0), k_grid=(1, 5, 300))
    assert k == 1 and table[1] == pytest.approx(0.0, abs=1e-5) and sorted(table) == [1, 5, 60]


def test_embed_returns_head_inputs_query_and_direct_predictions(fake_bag_batch):
    torch.manual_seed(0)
    model = RetroProfile(
        encoder=TinyRandomEncoder(output_dim=32),
        pooling=MILAttentionPooling(input_dim=32, hidden_dim=16, num_heads=2, target_names=["age", "height"]),
        heads={"age": ResidualRetrievalHead(input_dim=16, hidden_dim=8, label_min=18.0, label_max=80.0),
               "height": ResidualRetrievalHead(input_dim=16, hidden_dim=8, label_min=140.0, label_max=210.0)},
        projector=MLPHead(input_dim=32, hidden_dim=32, output_dim=16, l2_normalize=True)).eval()
    out = fr.embed(model, [fake_bag_batch], "cpu")
    assert out["query"].shape == (3, 16) and out["z"]["age"].shape == (3, 16)
    with torch.no_grad():
        want = model(fake_bag_batch)
    assert torch.allclose(out["direct"]["age"], want["age"]) and torch.allclose(out["direct"]["height"], want["height"])
    assert torch.equal(out["truth"]["age"], fake_bag_batch.labels["age"].float())


def test_reproduction_check_stops_on_a_mismatch(tmp_path):
    ours = pd.DataFrame({"speaker_id": ["A", "B"], "target": "age", "prediction": [30.0, 40.0]})
    ref = tmp_path / "predictions_timit.csv"
    ours.assign(prediction=[30.01, 40.0]).to_csv(ref, index=False)
    assert fr.check_reproduces(ours, ref) == pytest.approx(0.01)
    ours.assign(prediction=[31.0, 40.0]).to_csv(ref, index=False)
    with pytest.raises(SystemExit, match="differ"):
        fr.check_reproduces(ours, ref)


def _loo_mae(pred, y):
    out = []
    for i in range(len(pred)):
        keep = np.arange(len(pred)) != i
        a, b = np.polyfit(pred[keep], y[keep], 1)
        out.append(abs(a * pred[i] + b - y[i]))
    return float(np.mean(out))


def test_calibrated_mae_is_leave_one_out_after_the_two_number_fit():
    """--calibrated (Table 5): a model that ranks well but is offset and stretched scores well once calibrated."""
    rng = np.random.default_rng(0)
    y = rng.uniform(20, 40, 50)
    pred = 1.5 * y + 12 + rng.normal(0, 0.5, 50)
    raw = fr._mae(torch.tensor(pred), torch.tensor(y))
    cal = fr._mae(torch.tensor(pred), torch.tensor(y), calibrated=True)
    assert raw > 20 and cal < 0.6 and cal == pytest.approx(_loo_mae(pred, y), rel=1e-6)


def test_calibrated_k_choice_scores_each_k_after_calibration():
    ds, q = _datastore()
    rng = np.random.RandomState(3)
    val = {"speaker_ids": [f"X{i}" for i in range(25)], "query": _unit(rng, 25), "truth": {}}
    val["truth"]["age"] = fr.knn_estimate(ds, val, "age", 5) * 1.7 - 30 + torch.from_numpy(rng.randn(25).astype(np.float32))
    k, table = fr.choose_k(ds, val, "age", (-1e9, 1e9), k_grid=(1, 5, 20), calibrated=True)
    for kk, v in table.items():
        est = fr.knn_estimate(ds, val, "age", kk).numpy().astype(float)
        assert v == pytest.approx(_loo_mae(est, val["truth"]["age"].numpy().astype(float)), rel=1e-5)
    assert k == min(table, key=table.get)


def test_calibrated_frames_fit_on_validation_and_apply_to_test():
    rng = np.random.default_rng(1)
    yv, yt = rng.uniform(20, 40, 30), rng.uniform(20, 40, 12)
    val = pd.DataFrame({"speaker_id": [f"v{i}" for i in range(30)], "target": "age", "prediction": 2 * yv + 5 + rng.normal(0, 1, 30), "ground_truth": yv})
    test = pd.DataFrame({"speaker_id": [f"t{i}" for i in range(12)], "target": "age", "prediction": 2 * yt + 5, "ground_truth": yt})
    vcal, tcal = fr.calibrated_frames(val, test)
    a, b = np.polyfit(val.prediction, val.ground_truth, 1)
    assert np.allclose(tcal.prediction, a * test.prediction + b)
    assert np.mean(np.abs(vcal.prediction - vcal.ground_truth)) == pytest.approx(_loo_mae(val.prediction.to_numpy(), yv), rel=1e-6)


def test_calibrated_residual_training_never_worse_than_calibrated_knn_alone():
    head = _head()
    z, a, y = _biased(60, 4)
    fit = fr.train_residual(head, z, a, y, z, a * 1.3 + 7, y, y_scale=10.0, epochs=30, calibrated=True)
    assert fit["best_val_mae"] <= fit["knn_val_mae"]
    assert fit["knn_val_mae"] == pytest.approx(fr._mae((a * 1.3 + 7).clamp(18, 80), y, calibrated=True), rel=1e-6)


def test_missing_ages_are_skipped_in_k_choice_residual_and_calibration():
    """Table 4 (17 Oct 2026): HeightCeleb split2 has an age for under half its dev / test speakers (112 / 251,
    118 / 249). k, the residual epoch and the 2-number calibration use only speakers with an age; speakers without
    one still get a prediction (their height counts), and no score turns NaN."""
    ds, _ = _datastore()
    rng = np.random.RandomState(7)
    val = {"speaker_ids": [f"X{i}" for i in range(30)], "query": _unit(rng, 30), "truth": {}}
    est = fr.knn_estimate(ds, val, "age", 5)
    truth = est * 1.4 - 12 + torch.from_numpy(rng.randn(30).astype(np.float32))
    truth[::2] = float("nan")                                                       # half without an age
    val["truth"]["age"] = truth
    aged = ~torch.isnan(truth)
    k, table = fr.choose_k(ds, val, "age", (-1e9, 1e9), k_grid=(1, 5, 20), calibrated=True)
    for kk, v in table.items():
        e = fr.knn_estimate(ds, val, "age", kk)[aged].numpy().astype(float)
        assert np.isfinite(v) and v == pytest.approx(_loo_mae(e, truth[aged].numpy().astype(float)), rel=1e-5)
    head = _head()
    z, a, y = _biased(40, 8)
    y_tr = y.clone()
    y_tr[:10] = float("nan")
    fit = fr.train_residual(head, z, a, y_tr, z[:30], est, truth, y_scale=10.0, epochs=5, calibrated=True)
    assert np.isfinite(fit["knn_val_mae"]) and np.isfinite(fit["best_val_mae"])
    assert fit["knn_val_mae"] == pytest.approx(fr._mae(est.clamp(18, 80)[aged], truth[aged], calibrated=True), rel=1e-6)

    gt = rng.uniform(20, 60, 12)
    gt[:5] = np.nan
    vdf = pd.DataFrame({"speaker_id": [f"v{i}" for i in range(30)], "target": "age", "prediction": est.numpy(),
                        "ground_truth": truth.numpy()})
    tdf = pd.DataFrame({"speaker_id": [f"t{i}" for i in range(12)], "target": "age",
                        "prediction": rng.uniform(20, 60, 12), "ground_truth": gt})
    vcal, tcal = fr.calibrated_frames(vdf, tdf)
    assert len(vcal) == int(aged.sum()) and vcal["prediction"].notna().all()       # validation: aged rows only
    assert len(tcal) == 12 and tcal["prediction"].notna().all()                    # test: every speaker predicted
    b, c = np.polyfit(vdf.dropna().prediction, vdf.dropna().ground_truth, 1)
    assert np.allclose(tcal["prediction"], b * tdf["prediction"] + c)               # fit on aged validation only
