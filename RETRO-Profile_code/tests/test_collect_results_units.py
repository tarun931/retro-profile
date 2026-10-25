"""`collect_results` must surface every protocol, not just the eval's own unit.

`evaluate.py --unit utterance` derives per-SPEAKER metrics from the same
predictions and writes both under `<corpus>__breakdown`. Those derived speaker
numbers are what the RQ1 table is built on: Gupta predicts each clip and
averages, which is the aggregation RETRO's `eval_spk` (MIL pooling) has to be
compared against.

Reading only the top-level `<corpus>` block reported Gupta's per-UTTERANCE
figure (4.19) where the table needs its per-SPEAKER one (3.66) -- silently
comparing two different protocols, which rule R6 exists to prevent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_results import group_name, read_all_units  # noqa: E402


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(payload))
    return p


def test_both_protocols_are_returned(tmp_path):
    summary = _write(tmp_path, {
        "timit": {"age/mae": 4.19, "height/mae": 5.50},
        "timit__unit": "utterance",
        "timit__breakdown": {
            "utterance": {"overall": {"age/mae": 4.19, "height/mae": 5.50}},
            "speaker": {"overall": {"age/mae": 3.66, "height/mae": 5.13}},
        },
    })
    units = read_all_units(summary)
    assert set(units) == {"utterance", "speaker"}
    assert units["speaker"]["age/mae"] == 3.66
    assert units["utterance"]["age/mae"] == 4.19


def test_single_target_run_has_no_height(tmp_path):
    """The age-only run must not fabricate height metrics."""
    summary = _write(tmp_path, {
        "timit": {"age/mae": 3.49},
        "timit__breakdown": {"speaker": {"overall": {"age/mae": 3.49}}},
    })
    units = read_all_units(summary)
    assert units["speaker"]["age/mae"] == 3.49
    assert "height/mae" not in units["speaker"]


def test_falls_back_when_there_is_no_breakdown(tmp_path):
    """Older runs predate the breakdown block and must still be readable."""
    summary = _write(tmp_path, {"timit": {"age/mae": 5.58, "height/mae": 5.35}})
    units = read_all_units(summary)
    assert len(units) == 1
    assert next(iter(units.values()))["age/mae"] == 5.58


def test_seed_suffix_is_stripped_for_grouping():
    """Seeds of one config must collapse into a single mean +/- std row."""
    names = ["retro_noknn_budget_timit_s42", "retro_noknn_budget_timit_s1",
             "retro_noknn_budget_timit_s2"]
    assert {group_name(n) for n in names} == {"retro_noknn_budget_timit"}
