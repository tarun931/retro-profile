"""Build heightceleb/voxceleb_age_enrichment.pdf -- the VoxCeleb age enrichment, and what it gives us.

Paper: Hechmi, Trong, Hautamaki, Kinnunen, "VoxCeleb enrichment for age and gender
recognition", ASRU 2021 (arXiv:2109.13510v2). Repository: https://git.io/JYKN4 ->
github.com/hechmik/voxceleb_enrichment_age_gender (branch main), read file by file
on 2026-10-11.

Statistics are COMPUTED from the repository's files and saved to
voxceleb_age_enrichment_stats.json; the PDF is built from that JSON. To recompute,
download the repository's dataset/ and notebooks/ files (dataset/yt_metadata.json is
101 MB, final_dataframe_extended.csv 88 MB) and pass the folder:

    python heightceleb/build_age_enrichment.py --repo /path/to/download   # recompute + build
    python heightceleb/build_age_enrichment.py                            # build from the JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
STATS = HERE / "voxceleb_age_enrichment_stats.json"
OUT = HERE / "voxceleb_age_enrichment.pdf"
HEIGHTCELEB = HERE / "height_celeb.csv"
ACCENT = colors.HexColor("#1f3a63")
GREY = colors.HexColor("#f2f4f7")
LINE = colors.HexColor("#c9d1dc")


# ---- statistics from the repository's files ------------------------------------------
def compute(repo: Path) -> dict:
    ds = repo / "dataset"
    enr = pd.read_csv(ds / "final_dataframe_extended.csv", low_memory=False)
    ids = enr["VoxCeleb_ID"].astype(str)
    is_v1, is_v2 = ids.str.match(r"^id1\d{4}$"), ids.str.match(r"^id0\d{4}$")
    aged = enr[enr.speaker_age.notna()]
    g = aged.groupby("VoxCeleb_ID").speaker_age
    spread = g.max() - g.min()
    tr, te = pd.read_csv(ds / "age-train.txt"), pd.read_csv(ds / "age-test.txt")

    v1pv = json.loads((ds / "voxceleb_person_videos.json").read_text())
    v2vp = json.loads((ds / "voxceleb2_video_people.json").read_text())
    v1 = pd.DataFrame([(s, v) for s, vs in v1pv.items() for v in vs], columns=["spk", "video_id"])
    yt = pd.DataFrame.from_records(list(json.loads((ds / "yt_metadata.json").read_text())["_default"].values()))
    m = v1.merge(yt, on="video_id", how="left")
    m["year"] = m["publishing_date"].fillna("").str[:4]
    has_meta = m["year"] != ""

    def in_col(col):
        return pd.Series([y != "" and isinstance(t, str) and y in t for y, t in zip(m.year, m[col], strict=True)],
                         index=m.index)

    in_title, in_desc = in_col("title"), in_col("description")
    strict = in_title & in_desc
    hc = pd.read_csv(HEIGHTCELEB)
    hc_test = set(hc[hc.split == "TEST"]["VoxCeleb1 ID"])

    import re
    import unicodedata

    def norm(s):
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z]", "", s.lower())

    vox1_meta = pd.read_csv(ds / "vox1_meta.csv", sep="\t")
    names1 = {norm(x) for x in vox1_meta["VGGFace1 ID"]}
    enr_names = enr.assign(n=enr["Name"].map(norm))
    shared = names1 & set(enr_names["n"])

    return {
        "table_rows": int(len(enr)), "table_columns": list(enr.columns),
        "table_vox1_ids": int(ids[is_v1].nunique()), "table_vox2_ids": int(ids[is_v2].nunique()),
        "with_gender_speakers": int(enr[enr.gender.notna()].VoxCeleb_ID.nunique()),
        "with_birth_year_speakers": int(enr[enr.birth_year.notna()].VoxCeleb_ID.nunique()),
        "with_age_speakers": int(g.ngroups),
        "with_age_title_only_speakers": int(enr[enr.speaker_age_title_only.notna()].VoxCeleb_ID.nunique()),
        "age_rows": int(len(aged)), "age_speaker_age_pairs": int(aged.drop_duplicates(["VoxCeleb_ID", "speaker_age"]).shape[0]),
        "age_min": float(aged.speaker_age.min()), "age_max": float(aged.speaker_age.max()),
        "age_median": float(aged.speaker_age.median()),
        "multi_age_speakers": int((g.nunique() > 1).sum()), "age_spread_median": float(spread.median()),
        "age_spread_p90": float(spread.quantile(0.9)), "age_spread_max": float(spread.max()),
        "split_train_speakers": int(tr.VoxCeleb_ID.nunique()), "split_test_speakers": int(te.VoxCeleb_ID.nunique()),
        "split_overlap": len(set(tr.VoxCeleb_ID) & set(te.VoxCeleb_ID)),
        "split_vox1_ids": int(sum(str(x).startswith("id1") for x in pd.concat([tr.VoxCeleb_ID, te.VoxCeleb_ID]))),
        "test_age_mean": round(float(te.speaker_age.mean()), 1), "test_age_median": float(te.speaker_age.median()),
        "yt_videos": int(len(yt)), "yt_with_date": int((yt.publishing_date != "").sum()),
        "v1_speakers": int(v1.spk.nunique()), "v1_pairs": int(len(v1)), "v1_videos": int(v1.video_id.nunique()),
        "v1_videos_also_in_v2": len(set(v1.video_id) & set(v2vp)),
        "v1_videos_with_meta": int(m[has_meta].video_id.nunique()),
        "v1_videos_strict": int(m[strict].video_id.nunique()), "v1_speakers_strict": int(m[strict].spk.nunique()),
        "v1_videos_title": int(m[in_title].video_id.nunique()), "v1_speakers_title": int(m[in_title].spk.nunique()),
        "hc_test_strict": len(set(m[strict].spk) & hc_test), "hc_test_title": len(set(m[in_title].spk) & hc_test),
        "v1_strict_years": {k: int(v) for k, v in m[strict].drop_duplicates("video_id").year.value_counts().sort_index().items()},
        "name_overlap_v1_v2": len(shared),
        "name_overlap_with_age": len(shared & set(enr_names[enr_names.speaker_age.notna()]["n"])),
    }


# ---- PDF ------------------------------------------------------------------------------
def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("H1", parent=s["Heading1"], fontSize=15, spaceBefore=12, spaceAfter=6, textColor=ACCENT))
    s.add(ParagraphStyle("B", parent=s["BodyText"], fontSize=9.2, leading=13, alignment=TA_LEFT, spaceAfter=5))
    s.add(ParagraphStyle("C", parent=s["BodyText"], fontSize=7.8, leading=10.2, spaceAfter=0))
    s.add(ParagraphStyle("CB", parent=s["C"], fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("S", parent=s["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#55606e")))
    return s


def table(data, widths, st, header=True):
    rows = [[Paragraph(str(v), st["CB"] if header and r == 0 else st["C"]) for v in row]
            for r, row in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        cmds += [("BACKGROUND", (0, r), (-1, r), GREY) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(cmds))
    return t


def build(k: dict) -> None:
    st = styles()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="VoxCeleb Age Enrichment",
                            author="RETRO-Profile project")
    avail = doc.width

    def para(text, style="B"):
        return Paragraph(text, st[style])

    s = []
    s.append(para("VoxCeleb age enrichment (Hechmi et al., ASRU 2021) -- what it contains, and what it gives "
                  "HeightCeleb", "H1"))
    s.append(para("Written 2026-10-11 for the RETRO-Profile project. Sources: the paper (arXiv 2109.13510v2) and "
                  "its repository (https://git.io/JYKN4 = github.com/hechmik/voxceleb_enrichment_age_gender), read "
                  "file by file, including the 88 MB label table and the 101 MB YouTube metadata. Numbers in "
                  "sections 5-7 are computed from those files (voxceleb_age_enrichment_stats.json); paper "
                  "numbers are transcribed. Companion: HeightCeleb_Summary.pdf in this folder.", "S"))
    s.append(Spacer(1, 4))
    facts = [["Quick facts (for future sessions)", ""],
             ["What it is", "Per-video speaker AGE (and a cross-checked gender) for VoxCeleb celebrities, from public "
                            "databases plus YouTube upload years. Code (MIT), label tables and three trained models"],
             ["Paper", "K. Hechmi, T. Ngo Trong, V. Hautamaki, T. Kinnunen (Univ. of Eastern Finland), 'VoxCeleb "
                       "enrichment for age and gender recognition', ASRU 2021; arXiv:2109.13510v2 (20 Dec 2021)"],
             ["Who has an age", f"<b>VoxCeleb2 only</b>: {k['with_age_speakers']:,} speakers by their main rule "
                                f"({k['with_age_title_only_speakers']:,} with a looser rule). <b>No VoxCeleb1 speaker "
                                f"-- so no HeightCeleb speaker -- has an age</b> ({k['table_vox1_ids']} VoxCeleb1 IDs in "
                                f"the {k['table_rows']:,}-row table)"],
             ["Why VoxCeleb1 is missing", "A bug in their enrichment notebook drops every VoxCeleb1 speaker at the "
                                          "ID-merge step (section 6). The paper then describes VoxCeleb1 as the "
                                          "unlabelled part, used only to train feature extractors"],
             ["What exists for VoxCeleb1", f"Names, speaker-to-video maps, and YouTube metadata for "
                                           f"{k['v1_videos_with_meta']:,} of {k['v1_videos']:,} videos. Their recording-year "
                                           f"rule finds a year for {k['v1_videos_strict']:,} videos of {k['v1_speakers_strict']} "
                                           f"speakers. Birth years are NOT in the repository"],
             ["Ages", f"{k['age_min']:.0f}-{k['age_max']:.0f} years, median {k['age_median']:.0f}; estimated error up to "
                      f"~2 years; ages are per video, and {k['multi_age_speakers']:,} speakers have more than one"],
             ["Their best result", "Age MAE 9.443 years on in-the-wild VoxCeleb2 (i-vectors + linear regression, "
                                   "or an MFCC CNN); gender F1 0.983"]]
    s.append(table(facts, [36 * mm, avail - 36 * mm], st))

    s.append(para("1. What the paper set out to do", "H1"))
    s.append(para("VoxCeleb ships speaker identity, gender and nationality but no age. The authors (1) derive age "
                  "and an independent gender label for VoxCeleb speakers by cross-checking public databases, and "
                  "(2) build first age and gender recognisers on this 'found', in-the-wild speech, where earlier "
                  "age work used controlled telephone corpora (NIST SRE 2008/2010, mostly students; i-vector MAEs "
                  "4.7-6.9 years)."))

    s.append(para("2. How the labels were made", "H1"))
    s.append(para("<b>People.</b> Each celebrity name was looked up in three sources: Wikidata (search API, first "
                  "plausible entity), DBpedia (SPARQL on foaf:name) and Google Knowledge Graph (which needs an API "
                  "key; it is used to find the Wikipedia page, whose Wikidata entry is then read). A gender or "
                  "birth year is kept only when <b>all three agree</b>. Paper counts: of 7,365 people, 5,536 have a "
                  "consistent gender (3,437 male, 2,098 female, 1 transgender female) and 5,157 a consistent birth "
                  "year. Against VoxCeleb's own gender labels they agree 99.2 %; of the 42 disagreements, 32 "
                  "VoxCeleb 'male' labels and 9 'female' ones were flipped, and the authors treat their labels as "
                  "correct."))
    s.append(para("<b>Videos.</b> The YouTube Data API returned title, description and upload date for 88.5 % of "
                  "videos (the rest were deleted or geo-blocked). Recording dates are unknown, so they work in "
                  "years: if the <b>upload year appears in both the title and the description</b>, it is taken as "
                  "the recording year. A looser 'title only' variant was used to add training data (1,106 more "
                  "celebrities, never used in testing). <b>Age = recording year - birth year</b>, accurate to "
                  "about +/-2 years. Ages are therefore per video: the paper's example is Tom Cruise, recorded in "
                  "three different years. For modelling they keep one randomly chosen age per speaker. Paper "
                  "totals: ages for 2,817 speakers, 6,601 (video, speaker, age) triplets, 4,821 (speaker, age) "
                  "pairs."))
    dist = [["Age (years)", "(0,10]", "(10,20]", "(20,30]", "(30,40]", "(40,50]", "(50,60]", "(60,70]", "(70,80]", "(80,92]"],
            ["Speakers (paper Table 1, per (speaker, age) pair)", "1", "112", "1,204", "1,287", "926", "702", "457", "122", "10"]]
    s.append(table(dist, [52 * mm] + [(avail - 52 * mm) / 9] * 9, st))

    s.append(para("3. Their experiments and results", "H1"))
    s.append(para("<b>Features.</b> i-vectors and x-vectors from ASVTorch (Kaldi + PyTorch), trained for speaker "
                  "recognition on <b>VoxCeleb1</b> (the repository's notes: training on VoxCeleb1, 'trial' data = "
                  "all of VoxCeleb2); also MFCC and log-mel features (25 ms frames, 30 mel filters, 24 MFCCs), "
                  "with non-speech removed and a random 2-second (200-frame) crop, plus MUSAN-augmented copies. "
                  "<b>Models.</b> Gender: logistic regression and 2- or 4-layer networks. Age: linear, LASSO and "
                  "ridge regression, small dense networks, and 1-D / 2-D CNNs (one variant also predicts an age "
                  "band, loss 0.1 x MSE + 10 x cross-entropy). <b>Protocol.</b> Labelled speakers split 60 / 40 "
                  "into train and test with no shared speakers (the repository's lists: "
                  f"{k['split_train_speakers']:,} / {k['split_test_speakers']:,} speakers, all VoxCeleb2); 5-fold "
                  "cross-validation for model choice; each speaker contributes the same number of recordings to "
                  "the score."))
    res = [["Result (MAE in years unless stated)", "i-vector", "x-vector"],
           ["Gender F1 -- logistic regression (best)", "0.9829", "0.9769"],
           ["Gender F1 -- 2-layer / 4-layer network", "0.9823 / 0.9805", "0.9776 / 0.9776"],
           ["Age, balanced training -- linear / ridge / LASSO", "10.332 / 10.402 / 10.126", "11.095 / 9.777 / 9.843"],
           ["Age, all training data -- linear / ridge / LASSO", "<b>9.443</b> / 9.539 / 9.516", "9.962 / 9.768 / 9.789"],
           ["Age, 1-D CNN on MFCC / log-mel (augmented, title-only data)", "9.443 / 16.770", ""],
           ["Age, 1-D CNN with age-band output, MFCC / log-mel", "9.510 / 13.326", ""],
           ["Reference: random draw from the test ages", "16.330", ""],
           ["Reference: uniform random over the age range", "23.212", ""],
           ["Reference: always predict 39 (best constant)", "12.243", ""]]
    s.append(table(res, [86 * mm, (avail - 86 * mm) / 2, (avail - 86 * mm) / 2], st))
    s.append(Spacer(1, 4))
    s.append(para("Errors depend strongly on age: lowest (about 5.5-6 years) around 40, where most data sits, and "
                  "predictions stay within 19-68 years -- young and old speakers are pulled to the middle. The "
                  "authors conclude that age from in-the-wild speech is hard and suggest better augmentation, "
                  "self-supervised features or an age prior."))

    s.append(PageBreak())
    s.append(para("4. The repository", "H1"))
    files = [["Path", "What it is (verified 2026-10-11)"],
             ["dataset/final_dataframe_extended.csv (88 MB)", f"The label table: {k['table_rows']:,} rows (one per video "
              "and speaker), 23 columns -- name; gender, birth date and nationality from each of Wikidata, DBpedia "
              "and Google; video id, title, upload date, description; year in title; VoxCeleb_ID; consensus gender "
              "and birth_year; upload year; recording_year and recording_year_title_only; speaker_age and "
              "speaker_age_title_only"],
             ["dataset/age-train.txt, age-test.txt", f"The age split: one row per speaker (age, birth year, name, ID, "
              f"video); {k['split_train_speakers']:,} / {k['split_test_speakers']:,} speakers, {k['split_overlap']} shared, "
              f"{k['split_vox1_ids']} from VoxCeleb1"],
             ["dataset/gender-train_set.txt, gender-test_set.txt", "Lists of speaker IDs for the gender task (VoxCeleb2)"],
             ["dataset/vox1_meta.csv, vox2_meta.csv", "VoxCeleb's own metadata: ID, name, gender, nationality / VGGFace2 "
              "ID, dev/test set"],
             ["dataset/voxceleb_person_videos.json, voxceleb_video_people.json", f"VoxCeleb1 speaker <-> YouTube video "
              f"maps: {k['v1_speakers']:,} speakers, {k['v1_videos']:,} videos, {k['v1_pairs']:,} pairs"],
             ["dataset/voxceleb2_person_videos.json, voxceleb2_video_people.json", "The same for VoxCeleb2"],
             ["dataset/yt_metadata.json (101 MB)", f"YouTube title, description and upload date for {k['yt_videos']:,} "
              f"videos of both corpora ({k['yt_with_date']:,} with a date)"],
             ["notebooks/01-Enrich_VoxCeleb_Dataset.ipynb", "Builds the label table (database queries, consensus, "
              "recording year, age)"],
             ["notebooks/01.1, 02.x, 03.x", "Age statistics; gender and age experiments on i-vectors, x-vectors, "
              "MFCC, log-mel"],
             ["notebooks/src/", "Database and YouTube query code (enrich/), model and training code"],
             ["best_models/", "The three best models: MFCC CNN (Keras) and i-vector linear regression (pickle) for "
              "age, MAE 9.443 each; i-vector logistic regression (PyTorch) for gender, F1 0.9829"]]
    s.append(table(files, [62 * mm, avail - 62 * mm], st))

    s.append(para("5. What the data actually contains (computed)", "H1"))
    s.append(para(f"The label table has {k['table_vox2_ids']:,} distinct VoxCeleb2 IDs and <b>{k['table_vox1_ids']} "
                  f"VoxCeleb1 IDs</b>. Consensus gender exists for {k['with_gender_speakers']:,} speakers, consensus "
                  f"birth year for {k['with_birth_year_speakers']:,}, an age (title and description rule) for "
                  f"{k['with_age_speakers']:,} and an age by the title-only rule for "
                  f"{k['with_age_title_only_speakers']:,}. The {k['age_rows']:,} aged rows give "
                  f"{k['age_speaker_age_pairs']:,} (speaker, age) pairs -- both match the paper. Ages run "
                  f"{k['age_min']:.0f}-{k['age_max']:.0f}, median {k['age_median']:.0f}; the test speakers average "
                  f"{k['test_age_mean']} (median {k['test_age_median']:.0f}). <b>Within a speaker:</b> "
                  f"{k['multi_age_speakers']:,} of {k['with_age_speakers']:,} speakers have more than one age across "
                  f"videos; the spread is a median {k['age_spread_median']:.0f}, 90th percentile "
                  f"{k['age_spread_p90']:.0f} and at most {k['age_spread_max']:.0f} years."))

    s.append(para("6. Why VoxCeleb1 has no ages: a bug, not a choice", "H1"))
    s.append(para("The enrichment notebook does try to include VoxCeleb1: it loads vox1_meta.csv beside "
                  "vox2_meta.csv (7,365 names in total), merges both speaker-video maps (170,678 videos) and queries "
                  "the databases for every name. It then builds one ID column with <font face='Courier'>if "
                  "row['VoxCeleb2 ID'] != np.nan: use the VoxCeleb2 ID, else the VoxCeleb1 ID</font>. In Python that "
                  "test is always true, because NaN never equals anything, itself included. So VoxCeleb1 speakers "
                  "receive the empty VoxCeleb2 ID, and the later merge of videos with people on that column drops "
                  "every VoxCeleb1 video. The table keeps only VoxCeleb2 rows (149,350, matching the notebook's "
                  "printed shape). The paper's description of VoxCeleb1 as the data 'where these attributes are "
                  "absent', used for extractor training, is consistent with this outcome. Whether the authors "
                  "noticed is not stated anywhere in the repository."))

    s.append(para("7. What exists for VoxCeleb1 (and so for HeightCeleb) in the repository", "H1"))
    v1 = [["Item (computed)", "Count"],
          ["VoxCeleb1 speakers / videos / (speaker, video) pairs in the maps",
           f"{k['v1_speakers']:,} / {k['v1_videos']:,} / {k['v1_pairs']:,}"],
          ["VoxCeleb1 video IDs that also appear in VoxCeleb2", f"{k['v1_videos_also_in_v2']:,}"],
          ["VoxCeleb1 videos with YouTube metadata (title, description, date)",
           f"{k['v1_videos_with_meta']:,} ({k['v1_videos_with_meta'] / k['v1_videos']:.1%})"],
          ["... upload year in title AND description (their recording-year rule)",
           f"{k['v1_videos_strict']:,} videos, {k['v1_speakers_strict']} speakers"],
          ["... upload year in the title (looser rule)", f"{k['v1_videos_title']:,} videos, {k['v1_speakers_title']} speakers"],
          ["HeightCeleb TEST speakers (40) covered: main rule / looser rule", f"{k['hc_test_strict']} / {k['hc_test_title']}"],
          ["VoxCeleb1 names also among the VoxCeleb2 names (normalised) / of those with an age",
           f"{k['name_overlap_v1_v2']} / {k['name_overlap_with_age']}"],
          ["Birth years for VoxCeleb1 names", "<b>Not in the repository</b> (the per-source query results were not saved)"]]
    s.append(table(v1, [118 * mm, avail - 118 * mm], st))
    yrs = k["v1_strict_years"]
    s.append(Spacer(1, 4))
    s.append(para("Recording years found by the main rule for VoxCeleb1 videos: "
                  + ", ".join(f"{y}: {n}" for y, n in yrs.items()) + ". VoxCeleb1 and VoxCeleb2 are effectively "
                  "different people: only a handful of names occur in both, so ages cannot be borrowed across them."))

    s.append(para("8. What this means for RETRO-Profile", "H1"))
    s.append(para("<b>HeightCeleb speakers get no age from this repository.</b> Three ways forward, none chosen yet:"))
    opts = [["Option", "What it takes", "For", "Against"],
            ["A. Age from NISP only (HeightCeleb rows age = NaN)", "Nothing new; the NaN-aware loss and kNN fallback "
             "already handle it", "Simplest; clean labels", "NISP ages 18-47.5 only: on 2026-10-11 a NISP-trained model "
             "was 14-24 years too young for TIMIT speakers over 40"],
            ["B. Add VoxCeleb2 age-labelled speakers as an age-only corpus", f"VoxCeleb2 audio for the labelled videos "
             f"({k['age_rows']:,} videos, {k['with_age_speakers']:,} speakers); a manifest; height = NaN",
             f"Ages {k['age_min']:.0f}-{k['age_max']:.0f}, median {k['age_median']:.0f} -- covers TIMIT's 21-76",
             "VoxCeleb audio is no longer distributed officially; labels +/-2 years and per video; in-the-wild"],
            ["C. Derive VoxCeleb1 ages with their method", f"Birth years for 1,251 names (Wikidata / DBpedia / Google, "
             f"consensus); recording years from yt_metadata.json ({k['v1_speakers_strict']} speakers by the main rule)",
             "HeightCeleb speakers would carry height AND age (roughly half of them)", "Our own labelling effort; a "
             "Google Knowledge Graph key; the same noise as B; only about half the speakers"]]
    s.append(table(opts, [38 * mm, 50 * mm, 38 * mm, avail - 126 * mm], st))
    s.append(Spacer(1, 4))
    s.append(para("<b>Integration notes.</b> (1) Height is per speaker but these ages are per video: a speaker "
                  "with several ages needs either one chosen video (its clips form the bag) or one chosen age, as "
                  "the authors did. (2) Their MAE of 9.4 years is per recording, on VoxCeleb2's wide age range and "
                  "noisy audio; it is not comparable with our TIMIT per-speaker age MAE (3.7). (3) Label noise of "
                  "+/-2 years is a floor on what such labels can teach. (4) Selection must stay on validation "
                  "speakers, never on TIMIT."))

    s.append(para("9. Not verified / open", "H1"))
    for item in ("Whether a VoxCeleb2 audio copy is obtainable for the labelled videos (the official site no longer "
                 "distributes VoxCeleb audio -- see HeightCeleb_Summary.pdf).",
                 "How many VoxCeleb1 names would get a consensus birth year (in VoxCeleb2 it was 5,157 of 6,112).",
                 "Age-VoxCeleb (Tawara et al., ICASSP 2021; github.com/nttcslab-sp/agevoxceleb), which estimated ages "
                 "from faces, is mentioned by the authors; its coverage of VoxCeleb1 has not been checked.",
                 "The repository's code was read but not run; the trained models were not loaded."):
        s.append(para("- " + item))

    s.append(para("10. Sources", "H1"))
    for src in ("K. Hechmi, T. Ngo Trong, V. Hautamaki, T. Kinnunen, 'VoxCeleb enrichment for age and gender "
                "recognition', ASRU 2021; arXiv:2109.13510v2.",
                "github.com/hechmik/voxceleb_enrichment_age_gender (via https://git.io/JYKN4), branch main, last "
                "change 18 Dec 2021, MIT licence: README.md, dataset/README.md, notebooks/README.md, "
                "best_models/README.md, all dataset/ files, notebooks 01, 01.1 and 03.1, notebooks/src/enrich/*.py, "
                "notebooks/src/training_setup.py (read 2026-10-11).",
                "heightceleb/height_celeb.csv (HeightCeleb labels) for the TEST-speaker coverage."):
        s.append(para("- " + src, "S"))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#7a838f"))
        canvas.drawString(18 * mm, 10 * mm, "VoxCeleb age enrichment -- RETRO-Profile")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, default=None,
                    help="folder holding the downloaded repository files; recomputes the statistics")
    args = ap.parse_args()
    if args.repo is not None:
        STATS.write_text(json.dumps(compute(args.repo), indent=1))
        print(f"Wrote {STATS}")
    build(json.loads(STATS.read_text()))


if __name__ == "__main__":
    main()
