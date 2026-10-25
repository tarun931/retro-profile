"""VoxCeleb1 age labels (heightceleb/build_vox1_ages.py), without touching the network.

The script must reproduce Hechmi et al.'s recording-year rule exactly, use Wikidata's
preferred birth date, keep a birth year only when Wikidata and DBpedia agree, reject a
same-name person of the other gender, and pick each speaker's age deterministically.
"""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parent.parent / "heightceleb" / "build_vox1_ages.py"
_SPEC = importlib.util.spec_from_file_location("build_vox1_ages", _PATH)
va = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(va)


def test_recording_year_rule_is_hechmis_substring_test():
    assert va.year_rules("2014", "Interview 2014", "Recorded at Cannes 2014") == (True, True)
    assert va.year_rules("2014", "Interview 2014", "no year here") == (True, False)
    assert va.year_rules("2014", "Interview", None) == (False, False)
    assert va.year_rules("", "Interview 2014", "2014") == (False, False)


def test_display_name():
    assert va.display_name("A.J._Buckley") == "A.J. Buckley"


def _stmt(value, rank="normal"):
    return {"rank": rank, "mainsnak": {"datavalue": {"value": value}}}


def _claims(human=True, born="+1975-06-12T00:00:00Z", sex="Q6581097"):
    c = {}
    if human:
        c["P31"] = [_stmt({"id": "Q5"})]
    if born:
        c["P569"] = [_stmt({"time": born})]
    if sex:
        c["P21"] = [_stmt({"id": sex})]
    return c


def test_preferred_birth_date_wins_and_deprecated_is_ignored():
    claims = {"P569": [_stmt({"time": "+1966-01-06T00:00:00Z"}),
                       _stmt({"time": "+1967-01-06T00:00:00Z"}, rank="preferred")]}
    assert va.best_claim(claims, "P569")["time"].startswith("+1967")
    claims = {"P569": [_stmt({"time": "+1900-01-01T00:00:00Z"}, rank="deprecated"),
                       _stmt({"time": "+1975-06-12T00:00:00Z"})]}
    assert va.best_claim(claims, "P569")["time"].startswith("+1975")
    assert va.best_claim({}, "P569") is None


def test_wikidata_claims_parse():
    assert va.wikidata_year_gender(_claims()) == (1975, "m", True)
    assert va.wikidata_year_gender(_claims(sex="Q6581072")) == (1975, "f", True)
    assert va.wikidata_year_gender(_claims(human=False, born=None, sex=None)) == (None, None, False)


def test_pick_person_takes_the_first_human_with_a_birth_year():
    ents = {"Q1": {"human": False, "birth_year": 1975},       # e.g. a film with the name
            "Q2": {"human": True, "birth_year": None},          # a person with no date
            "Q3": {"human": True, "birth_year": 1977, "label": "A. J. Buckley"}}
    assert va.pick_person(["Q1", "Q2", "Q3"], ents) == ("Q3", ents["Q3"])
    assert va.pick_person(["Q1"], ents) == (None, {})


def test_dbpedia_iri():
    assert va.dbpedia_iri("Edge (wrestler)") == "<http://dbpedia.org/resource/Edge_(wrestler)>"
    assert va.dbpedia_iri("Anahí de Cárdenas") == "<http://dbpedia.org/resource/Anahí_de_Cárdenas>"
    assert va.dbpedia_iri('Who?') == "<http://dbpedia.org/resource/Who%3F>"


@pytest.fixture
def files(tmp_path, monkeypatch):
    for attr, name in (("META", "vox1_meta.csv"), ("VIDEOS", "vox1_videos.csv"), ("LOOKUP", "lookup.csv"),
                       ("SEARCH_CACHE", "search.json"), ("PER_VIDEO", "per_video.csv"),
                       ("PER_SPEAKER", "per_speaker.csv"), ("HEIGHTS", "height_celeb.csv")):
        monkeypatch.setattr(va, attr, tmp_path / name)
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    pd.DataFrame({"VoxCeleb1 ID": ["id10001", "id10002", "id10003"],
                  "VGGFace1 ID": ["A.J._Buckley", "B_Person", "C_Person"],
                  "Gender": ["m", "f", "m"], "Nationality": ["x"] * 3, "Set": ["dev"] * 3}) \
        .to_csv(tmp_path / "vox1_meta.csv", sep="\t", index=False)
    pd.DataFrame({"VoxCeleb1 ID": ["id10001", "id10002", "id10003"], "height": [180, 165, 175],
                  "sex": ["M", "F", "M"], "split": ["TRAIN", "TRAIN", "TEST"]}) \
        .to_csv(tmp_path / "height_celeb.csv", index=False)
    pd.DataFrame([
        ("id10001", "vB", "t 2010", "2010-01-01", "2010", True, True),
        ("id10001", "vA", "t 2012", "2012-01-01", "2012", True, False),    # title only
        ("id10001", "vC", "t", "2013-01-01", "2013", False, False),        # no rule: unusable
        ("id10002", "vD", "t 2016", "2016-01-01", "2016", True, True),
        ("id10003", "vE", "t 2015", "2015-01-01", "2015", True, True),
    ], columns=["speaker_id", "video_id", "title", "publishing_date", "upload_year", "year_in_title",
                "year_in_description"]).to_csv(tmp_path / "vox1_videos.csv", index=False)
    return tmp_path


def _fake_wikidata(calls):
    people = {"A.J. Buckley": ["Q10", "Q11"], "B Person": ["Q20"], "C Person": []}
    entities = {
        "Q10": {"claims": _claims(human=False)},
        "Q11": {"claims": _claims(born="+1977-02-09T00:00:00Z"), "labels": {"en": {"value": "A. J. Buckley"}},
                "sitelinks": {"enwiki": {"title": "A. J. Buckley"}}},
        "Q20": {"claims": _claims(born="+1990-01-01T00:00:00Z", sex="Q6581097"),   # wrong gender
                "labels": {"en": {"value": "B Person"}}, "sitelinks": {"enwiki": {"title": "B Person"}}},
    }

    def get(url):
        calls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if q["action"] == ["wbsearchentities"]:
            return {"search": [{"id": i} for i in people[q["search"][0]]]}
        return {"entities": {i: entities[i] for i in q["ids"][0].split("|")}}
    return get


def _fake_dbpedia(births):
    def get(url):
        sparql = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["query"][0]
        rows = [{"p": {"value": f"http://dbpedia.org/resource/{t.replace(' ', '_')}"}, "birth": {"value": b}}
                for t, bs in births.items() for b in bs if t.replace(" ", "_") in sparql]
        return {"results": {"bindings": rows}}
    return get


def test_lookup_end_to_end(files):
    calls = []
    df = va.run_lookup(get_json=_fake_wikidata(calls),
                       get_sparql=_fake_dbpedia({"A. J. Buckley": ["1977-02-09"], "B Person": ["1990-01-01"]}),
                       pause=0)
    rows = df.set_index("voxceleb_id")
    assert rows.loc["id10001", "wd_qid"] == "Q11" and rows.loc["id10001", "wd_birth_year"] == 1977
    assert rows.loc["id10001", "db_birth_year"] == 1977 and rows.loc["id10001", "db_status"] == "found"
    assert pd.isna(rows.loc["id10003", "wd_qid"]) and rows.loc["id10003", "db_status"] == "not_found"
    assert va.birth_consensus(rows.loc["id10001"].to_dict()) == (1977, "consensus")
    assert va.birth_consensus(rows.loc["id10002"].to_dict()) == (None, "gender_mismatch")
    # the name search is cached: a second run makes no search requests
    n_search = sum("wbsearchentities" in u for u in calls)
    va.run_lookup(get_json=_fake_wikidata(calls), get_sparql=_fake_dbpedia({}), pause=0)
    assert sum("wbsearchentities" in u for u in calls) == n_search
    assert json.loads(va.SEARCH_CACHE.read_text())["A.J. Buckley"] == ["Q10", "Q11"]


def test_dbpedia_disagreeing_years_are_ambiguous(files):
    births = va.dbpedia_births(["X Y"], get=_fake_dbpedia({"X Y": ["1970-01-01", "1971-05-05"]}), pause=0)
    assert births == {"X Y": {1970, 1971}}


@pytest.mark.parametrize("row,expected", [
    ({"wd_birth_year": 1975, "db_birth_year": 1975, "wd_gender": "m", "vox_gender": "m"}, (1975, "consensus")),
    ({"wd_birth_year": 1975, "db_birth_year": 1976, "wd_gender": "m", "vox_gender": "m"}, (None, "sources_disagree")),
    ({"wd_birth_year": 1975, "db_birth_year": float("nan"), "wd_gender": "m", "vox_gender": "m"}, (None, "missing_source")),
    ({"wd_birth_year": 1975, "db_birth_year": 1975, "wd_gender": "f", "vox_gender": "m"}, (None, "gender_mismatch")),
])
def test_birth_year_consensus(row, expected):
    assert va.birth_consensus(row) == expected


def test_ages_per_video_and_one_per_speaker(files):
    pd.DataFrame([
        {"voxceleb_id": "id10001", "name": "A.J. Buckley", "vox_gender": "m", "wd_birth_year": 1975,
         "wd_gender": "m", "db_birth_year": 1975},
        {"voxceleb_id": "id10002", "name": "B Person", "vox_gender": "f", "wd_birth_year": 1990,
         "wd_gender": "f", "db_birth_year": 1991},                                          # disagree
        {"voxceleb_id": "id10003", "name": "C Person", "vox_gender": "m", "wd_birth_year": 2010,
         "wd_gender": "m", "db_birth_year": 2010},                                          # age 5: dropped
    ]).to_csv(va.LOOKUP, index=False)
    per_video, per_speaker = va.build_ages()                       # strict: title AND description only
    assert list(per_video.video_id) == ["vB"]
    row = per_speaker.iloc[0]
    assert list(per_speaker.speaker_id) == ["id10001"] and row.video_id == "vB" and row.age == 35
    assert row.height == 180 and row.age_spread == 0 and row.n_dated_videos == 1

    per_video, per_speaker = va.build_ages("loose")                # also title-only videos
    assert sorted(per_video.video_id) == ["vA", "vB"]
    assert dict(zip(per_video.video_id, per_video.age, strict=True)) == {"vA": 37, "vB": 35}
    row = per_speaker.iloc[0]
    assert row.video_id == "vB" and row.rule == "title_and_description"   # main rule preferred
    assert row.age == 35 and row.age_spread == 2 and row.n_dated_videos == 2


def test_http_429_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    replies = [urllib.error.HTTPError("u", 429, "Too Many Requests", {"Retry-After": "1"}, None)]

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        if replies:
            raise replies.pop()
        assert req.get_header("Accept") == va.SPARQL_JSON
        return Resp(b'{"ok": 1}')
    monkeypatch.setattr(va.urllib.request, "urlopen", urlopen)
    assert va._get_sparql("https://dbpedia.org/sparql?q") == {"ok": 1}
