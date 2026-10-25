"""Age labels for HeightCeleb's VoxCeleb1 speakers, by the Hechmi et al. (ASRU 2021) method.

HeightCeleb gives a height for every VoxCeleb1 speaker; the Hechmi age enrichment
covers VoxCeleb2 only (voxceleb_age_enrichment.pdf, section 6). This rebuilds ages for
VoxCeleb1 the same way:

    recording year  = a video's YouTube upload year, if that year appears in its title
                      AND description (their main rule) -- or in the title (looser rule)
    birth year      = Wikidata and DBpedia must agree on the year
    age             = recording year - birth year, per video

Two deliberate differences from their method, both recorded here:
  * Two sources, not three: Google Knowledge Graph needs an API key.
  * The person is identified ONCE, on Wikidata (first search hit that is a human with a
    birth date), and DBpedia is read for that person's English Wikipedia article. They
    matched the name separately in each source; doing so on DBpedia missed well-known
    people (Adam Brody, Abigail Spencer) whose foaf:name differs. The birth DATE is still
    compared between two separately maintained values (the Wikidata statement and the
    Wikipedia infobox DBpedia extracts).
Extra checks, not in their method: the Wikidata gender must match VoxCeleb's gender (a
same-name person of the other gender is rejected); Wikidata's PREFERRED birth date is
used when it lists several; ages outside 10-100 are dropped.

Three steps, each writing into this folder:

    python heightceleb/build_vox1_ages.py videos --hechmi-repo DIR   # offline, once
    python heightceleb/build_vox1_ages.py lookup [--limit N]         # online, resumable
    python heightceleb/build_vox1_ages.py ages [--rule loose]        # offline; strict by default

`videos` needs the Hechmi repository's dataset/ files (vox1_meta.csv,
voxceleb_person_videos.json, yt_metadata.json). `lookup` sends only celebrity names and
Wikidata / Wikipedia identifiers to the public Wikidata API and DBpedia SPARQL endpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
META = HERE / "vox1_meta.csv"
VIDEOS = HERE / "vox1_videos.csv"
SEARCH_CACHE = HERE / "vox1_wikidata_search.json"
LOOKUP = HERE / "vox1_birth_lookup.csv"
PER_VIDEO = HERE / "vox1_height_age_videos.csv"
PER_SPEAKER = HERE / "vox1_height_age.csv"
HEIGHTS = HERE / "height_celeb.csv"

USER_AGENT = "RETRO-Profile-thesis/0.1 (VoxCeleb1 age labels)"
PAUSE_S = 1.0
BATCH = 50                                           # Wikidata wbgetentities limit; also DBpedia VALUES size
SPARQL_JSON = "application/sparql-results+json"      # DBpedia answers 406 to plain application/json
RETRIES = 6
AGE_RANGE = (10, 100)
WD_GENDER = {"Q6581097": "m", "Q6581072": "f"}


# ---- step 1: videos (offline) ---------------------------------------------------------
def year_rules(upload_year: str, title, description) -> tuple[bool, bool]:
    """Hechmi's test, exactly: the upload year as a substring of the title / description."""
    if len(upload_year) < 4:
        return False, False
    in_title = isinstance(title, str) and upload_year in title
    in_desc = isinstance(description, str) and upload_year in description
    return in_title, in_desc


def build_videos(repo: Path) -> pd.DataFrame:
    ds = repo / "dataset"
    shutil.copyfile(ds / "vox1_meta.csv", META)
    person_videos = json.loads((ds / "voxceleb_person_videos.json").read_text())
    pairs = pd.DataFrame([(s, v) for s, vs in person_videos.items() for v in vs],
                         columns=["speaker_id", "video_id"])
    yt = pd.DataFrame.from_records(list(json.loads((ds / "yt_metadata.json").read_text())["_default"].values()))
    yt = yt[yt.video_id.isin(set(pairs.video_id))]
    df = pairs.merge(yt[["video_id", "title", "publishing_date", "description"]], on="video_id", how="left")
    df["publishing_date"] = df["publishing_date"].fillna("")
    df["upload_year"] = df["publishing_date"].str[:4]
    flags = [year_rules(y, t, d) for y, t, d in zip(df.upload_year, df.title, df.description, strict=True)]
    df["year_in_title"] = [f[0] for f in flags]
    df["year_in_description"] = [f[1] for f in flags]
    df = df.drop(columns=["description"]).sort_values(["speaker_id", "video_id"])
    df.to_csv(VIDEOS, index=False)
    return df


# ---- step 2: birth years (online) -----------------------------------------------------
def display_name(vgg_name: str) -> str:
    """'A.J._Buckley' -> 'A.J. Buckley'."""
    return str(vgg_name).replace("_", " ").strip()


def _get_json(url: str, accept: str = "application/json") -> dict:
    """GET with polite retries: on 429 / 5xx / network errors wait (Retry-After, else 5, 10, 20 s ...)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == RETRIES - 1:
                raise
            wait = exc.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else 5 * 2 ** attempt)
        except urllib.error.URLError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def _get_sparql(url: str) -> dict:
    return _get_json(url, accept=SPARQL_JSON)


def best_claim(claims: dict, prop: str):
    """Value of the preferred statement for `prop`, else the first normal one; never deprecated."""
    usable = [c for c in claims.get(prop, []) if c.get("rank") != "deprecated"]
    preferred = [c for c in usable if c.get("rank") == "preferred"]
    for c in preferred or usable:
        value = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if value is not None:
            return value
    return None


def wikidata_year_gender(claims: dict) -> tuple[int | None, str | None, bool]:
    """(birth year, 'm'/'f'/'other', is_human) from a Wikidata claims dict."""
    human = any(c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q5"
                for c in claims.get("P31", []))
    born = best_claim(claims, "P569")
    year = int(born["time"][1:5]) if born and re.match(r"^\+\d{4}-", born.get("time", "")) else None
    sex = best_claim(claims, "P21")
    gender = WD_GENDER.get(sex["id"], "other") if sex else None
    return year, gender, human


def search_wikidata(names: list[str], get=_get_json, pause: float = PAUSE_S) -> dict[str, list[str]]:
    """name -> up to 5 candidate QIDs; cached on disk so an interrupted run resumes."""
    cache = json.loads(SEARCH_CACHE.read_text()) if SEARCH_CACHE.exists() else {}
    todo = [n for n in dict.fromkeys(names) if n not in cache]
    for i, name in enumerate(todo, 1):
        q = urllib.parse.urlencode({"action": "wbsearchentities", "search": name, "language": "en",
                                    "type": "item", "limit": 5, "format": "json"})
        cache[name] = [h["id"] for h in get(f"https://www.wikidata.org/w/api.php?{q}").get("search", [])]
        if i % 25 == 0 or i == len(todo):
            SEARCH_CACHE.write_text(json.dumps(cache))
            print(f"  wikidata search {i}/{len(todo)}", flush=True)
        time.sleep(pause)
    return cache


def fetch_entities(qids: list[str], get=_get_json, pause: float = PAUSE_S) -> dict[str, dict]:
    """QID -> {label, birth_year, gender, human, enwiki}, 50 entities per request."""
    out: dict[str, dict] = {}
    uniq = list(dict.fromkeys(qids))
    for i in range(0, len(uniq), BATCH):
        q = urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(uniq[i:i + BATCH]),
                                    "props": "claims|labels|sitelinks", "languages": "en",
                                    "sitefilter": "enwiki", "format": "json"})
        for qid, e in get(f"https://www.wikidata.org/w/api.php?{q}").get("entities", {}).items():
            year, gender, human = wikidata_year_gender(e.get("claims", {}))
            out[qid] = {"label": e.get("labels", {}).get("en", {}).get("value"), "birth_year": year,
                        "gender": gender, "human": human,
                        "enwiki": e.get("sitelinks", {}).get("enwiki", {}).get("title")}
        time.sleep(pause)
    return out


def pick_person(candidates: list[str], entities: dict[str, dict]) -> tuple[str | None, dict]:
    """First candidate that is a human with a birth year (Hechmi's 'first plausible entity')."""
    for qid in candidates:
        e = entities.get(qid, {})
        if e.get("human") and e.get("birth_year") is not None:
            return qid, e
    return None, {}


def dbpedia_iri(title: str) -> str:
    """Wikipedia title -> DBpedia resource IRI (spaces to underscores; IRI-unsafe chars escaped)."""
    t = title.replace(" ", "_")
    return "<http://dbpedia.org/resource/" + "".join(
        urllib.parse.quote(ch) if ch in '<>"{}|^`\\?#' else ch for ch in t) + ">"


def dbpedia_births(titles: list[str], get=_get_sparql, pause: float = PAUSE_S) -> dict[str, set[int]]:
    """Wikipedia title -> birth years DBpedia holds for that article (empty set if none)."""
    out: dict[str, set[int]] = {t: set() for t in titles}
    uniq = list(dict.fromkeys(titles))
    for i in range(0, len(uniq), BATCH):
        chunk = uniq[i:i + BATCH]
        iris = {dbpedia_iri(t)[1:-1]: t for t in chunk}
        sparql = ("SELECT ?p ?birth WHERE { VALUES ?p { " + " ".join(f"<{iri}>" for iri in iris)
                  + " } ?p dbo:birthDate ?birth . }")
        q = urllib.parse.urlencode({"query": sparql, "format": SPARQL_JSON})
        for r in get(f"https://dbpedia.org/sparql?{q}").get("results", {}).get("bindings", []):
            title = iris.get(r["p"]["value"])
            if title is not None and re.match(r"^\d{4}", r["birth"]["value"]):
                out[title].add(int(r["birth"]["value"][:4]))
        time.sleep(pause)
    return out


def run_lookup(limit: int | None = None, get_json=_get_json, get_sparql=_get_sparql,
               pause: float = PAUSE_S) -> pd.DataFrame:
    """Birth year from Wikidata and from DBpedia for every speaker; writes LOOKUP."""
    meta = pd.read_csv(META, sep="\t")
    if limit is not None:
        meta = meta.head(limit)
    names = [display_name(n) for n in meta["VGGFace1 ID"]]
    candidates = search_wikidata(names, get=get_json, pause=pause)
    entities = fetch_entities([q for n in names for q in candidates.get(n, [])], get=get_json, pause=pause)
    rows = []
    for vid, name, vox_gender in zip(meta["VoxCeleb1 ID"], names, meta["Gender"], strict=True):
        qid, e = pick_person(candidates.get(name, []), entities)
        rows.append({"voxceleb_id": vid, "name": name, "vox_gender": str(vox_gender).strip().lower()[:1],
                     "wd_qid": qid, "wd_label": e.get("label"), "wd_birth_year": e.get("birth_year"),
                     "wd_gender": e.get("gender"), "enwiki_title": e.get("enwiki")})
    df = pd.DataFrame(rows)
    births = dbpedia_births([t for t in df.enwiki_title.dropna()], get=get_sparql, pause=pause)

    def db(title):
        years = births.get(title, set()) if isinstance(title, str) else set()
        if len(years) == 1:
            return next(iter(years)), "found"
        return None, ("ambiguous" if years else "not_found")

    df[["db_birth_year", "db_status"]] = [db(t) for t in df.enwiki_title]
    df.to_csv(LOOKUP, index=False)
    return df


# ---- step 3: ages (offline) -----------------------------------------------------------
def birth_consensus(row) -> tuple[int | None, str]:
    """Birth year kept only if Wikidata and DBpedia agree and the gender matches VoxCeleb."""
    wd, db = row.get("wd_birth_year"), row.get("db_birth_year")
    if pd.isna(wd) or pd.isna(db):
        return None, "missing_source"
    if int(wd) != int(db):
        return None, "sources_disagree"
    if row.get("wd_gender") not in (None, "") and not pd.isna(row.get("wd_gender")) \
            and row["wd_gender"] != row.get("vox_gender"):
        return None, "gender_mismatch"
    return int(wd), "consensus"


RULES = {"strict": ("title_and_description",), "loose": ("title_and_description", "title_only")}


def build_ages(rule: str = "strict") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-video and per-speaker age tables. rule="strict" (the agreed default, 2026-10-12) keeps only
    videos whose upload year is in the title AND the description; "loose" also accepts title only."""
    heights = pd.read_csv(HEIGHTS).rename(columns={"VoxCeleb1 ID": "speaker_id"})
    look = pd.read_csv(LOOKUP, dtype={"voxceleb_id": str})
    cons = look.apply(birth_consensus, axis=1, result_type="expand")
    look["birth_year"], look["birth_status"] = cons[0], cons[1]
    vids = pd.read_csv(VIDEOS, dtype={"upload_year": str})
    vids = vids[vids.year_in_title]                         # every usable video passes the title test
    vids["rule"] = vids.year_in_description.map({True: "title_and_description", False: "title_only"})
    vids = vids[vids.rule.isin(RULES[rule])]
    vids["recording_year"] = pd.to_numeric(vids.upload_year)
    df = (vids.merge(look[["voxceleb_id", "name", "birth_year", "birth_status"]],
                     left_on="speaker_id", right_on="voxceleb_id")
              .merge(heights, on="speaker_id"))
    df = df[df.birth_year.notna()].copy()
    df["age"] = df.recording_year - df.birth_year
    df = df[df.age.between(*AGE_RANGE)]
    cols = ["speaker_id", "name", "sex", "split", "height", "birth_year", "video_id", "title",
            "recording_year", "rule", "age"]
    per_video = df[cols].sort_values(["speaker_id", "rule", "video_id"])
    per_video.to_csv(PER_VIDEO, index=False)

    # One age per speaker, as Hechmi did -- but deterministic: prefer the main rule, then
    # the earliest video id. The chosen video's clips make that speaker's bag.
    ranked = per_video.assign(_main=per_video.rule.eq("title_and_description")) \
                      .sort_values(["speaker_id", "_main", "video_id"], ascending=[True, False, True])
    spread = per_video.groupby("speaker_id").age.agg(lambda a: a.max() - a.min()).rename("age_spread")
    n = per_video.groupby("speaker_id").video_id.nunique().rename("n_dated_videos")
    per_speaker = ranked.drop_duplicates("speaker_id").drop(columns="_main").merge(spread, on="speaker_id") \
                        .merge(n, on="speaker_id")
    per_speaker.to_csv(PER_SPEAKER, index=False)
    return per_video, per_speaker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("videos")
    v.add_argument("--hechmi-repo", type=Path, required=True)
    lk = sub.add_parser("lookup")
    lk.add_argument("--limit", type=int, default=None)
    ag = sub.add_parser("ages")
    ag.add_argument("--rule", choices=sorted(RULES), default="strict")
    args = ap.parse_args()
    if args.cmd == "videos":
        df = build_videos(args.hechmi_repo)
        print(f"Wrote {VIDEOS}: {len(df):,} (speaker, video) rows, {df.speaker_id.nunique()} speakers; "
              f"title+description {int((df.year_in_title & df.year_in_description).sum()):,}, "
              f"title {int(df.year_in_title.sum()):,}")
    elif args.cmd == "lookup":
        out = run_lookup(args.limit)
        print(f"Wrote {LOOKUP}: {len(out)} speakers")
    else:
        pv, ps = build_ages(args.rule)
        print(f"Wrote {PER_VIDEO} ({len(pv):,} rows) and {PER_SPEAKER} ({len(ps)} speakers)")
        print(ps.rule.value_counts().to_dict())


if __name__ == "__main__":
    main()
