from pathlib import Path
import json
import pandas as pd
import numpy as np
import calendar
import re
from bs4 import BeautifulSoup
import os

RUN_YEAR =  "1997"
BASE_DIR = "congressional-record/output" #download data using Congressional Record: https://github.com/unitedstates/congressional-record/

INPUT_DIR = f"{BASE_DIR}/{RUN_YEAR}"
print("RUN_YEAR =", RUN_YEAR)
print("INPUT_DIR =", INPUT_DIR)

def load_json(path):
    """Robust JSON loader."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def rows_from_doc(doc: dict, source_path: str, include_header_row: bool, skip_kinds: set | None):
    """
    Yield flattened rows from a single document.
    - Header fields are copied to every content row.
    - Optionally emit a single header row with kind='header'.
    - Keeps original key names, for example 'kind'.
    """
    header = doc.get("header", {}) or {}
    top_level = {
        "id": doc.get("id"),
        "doc_title": doc.get("doc_title"),
        "title": doc.get("title"),
        "source_path": source_path,
    }

    if include_header_row:
        header_row = {**header, **top_level}
        header_row["kind"] = "header"
        yield header_row

    for item in doc.get("content", []) or []:
        if skip_kinds and item.get("kind") in skip_kinds:
            continue
        row = {**header, **top_level, **item}
        yield row

def json_tree_to_dataframe(
    root_dir: str | Path,
    pattern: str = "*.json",
    include_header_row: bool = True,
    skip_kinds: set[str] | None = {"linebreak"},
) -> pd.DataFrame:
    """
    Recursively find JSON files under root_dir, flatten each into rows, and return a single DataFrame.
    - include_header_row=True adds one header row per file with kind='header'
    - skip_kinds controls which content kinds are skipped, set to None to keep all
    """
    root = Path(root_dir)
    all_rows = []

    for path in root.rglob(pattern):
        try:
            doc = load_json(path)
        except Exception as e:

            print(f"Skipping {path} due to error: {e}")
            continue

        for row in rows_from_doc(
            doc,
            source_path=str(path),
            include_header_row=include_header_row,
            skip_kinds=skip_kinds,
        ):
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    preferred_first = [
        "kind", "speaker", "text", "turn", "speaker_bioguide", "itemno",
        "id", "doc_title", "title",

        "vol", "num", "wkday", "month", "day", "year", "chamber", "pages", "extension",
        "source_path",
    ]
    ordered_cols = [c for c in preferred_first if c in df.columns] + [c for c in df.columns if c not in preferred_first]
    df = df[ordered_cols]

    return df

if __name__ == "__main__":

    df = json_tree_to_dataframe(
        root_dir=INPUT_DIR,
        include_header_row=True,
        skip_kinds={"linebreak"},
    )

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "all_rows.parquet", index=False)

df.head()

df.shape

df = df[df["kind"] == "speech"]
df.shape

df.shape

df.columns

d = pd.to_datetime(
    df["year"].astype(str) + " " + df["month"].astype(str) + " " + df["day"].astype(str),
    format="%Y %B %d",
    errors="coerce",
)

cong = pd.Series(np.nan, index=df.index, dtype="float")

m1 = (d >= pd.Timestamp("1789-03-04")) & (d < pd.Timestamp("1933-03-04"))
y1 = d.dt.year
odd1 = y1 % 2 == 1
pre_mar4 = odd1 & ((d.dt.month < 3) | ((d.dt.month == 3) & (d.dt.day < 4)))
start_odd1 = np.where(odd1, np.where(pre_mar4, y1 - 2, y1), y1 - 1)
vals1 = pd.Series(((start_odd1 - 1789) // 2) + 1, index=df.index, dtype="float")
cong.loc[m1] = vals1.loc[m1]

m2 = (d >= pd.Timestamp("1933-03-04")) & (d < pd.Timestamp("1935-01-03"))
cong.loc[m2] = 73

m3 = d >= pd.Timestamp("1935-01-03")
y3 = d.dt.year
odd3 = y3 % 2 == 1
pre_jan3 = odd3 & (d.dt.month.eq(1) & d.dt.day.lt(3))
start_odd3 = np.where(odd3, np.where(pre_jan3, y3 - 2, y3), y3 - 1)
vals3 = pd.Series(((start_odd3 - 1935) // 2) + 74, index=df.index, dtype="float")
cong.loc[m3] = vals3.loc[m3]
df["congress"] = np.floor(cong).astype("float64")

def build_origin_html(doc_id: str) -> str | None:
    """Return govinfo HTML origin URL from a CREC id, 
    where the pkg part only contains CREC-YYYY-MM-DD."""
    if not doc_id:
        return None

    parts = doc_id.split("-")
    base_id = "-".join(parts[:4])

    return f"https://www.govinfo.gov/content/pkg/{base_id}/html/{doc_id}.htm"

df["origin_url"] = df["id"].apply(build_origin_html)

df.head()

df["the_order"] = df.groupby("id").cumcount() 
df.head()

df = df.rename(columns={"text": "speaking"})
df.head()

df = df.rename(columns={"speaker": "speaker_raw"})
df.head()

df = df.rename(columns={"num": "number"})

df = df.rename(columns={"vol": "volume"})

df.head()

month_map = {m: i for i, m in enumerate(calendar.month_name) if m}

df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" +
    df["month"].map(month_map).astype(str) + "-" +
    df["day"].astype(str),
    errors="coerce"
).dt.strftime("%Y-%m-%d")

df.head()

d = pd.to_datetime(
    df["year"].astype(str) + " " + df["month"].astype(str) + " " + df["day"].astype(str),
    format="%Y %B %d",
    errors="coerce",
)

session = pd.Series(pd.NA, index=df.index, dtype="Int64")

m1 = (d >= pd.Timestamp("1789-03-04")) & (d < pd.Timestamp("1933-03-04"))

s1_phase1 = m1 & (d.dt.year % 2 == 1) & (
    (d.dt.month > 3) | ((d.dt.month == 3) & (d.dt.day >= 4))
)
s2_phase1 = m1 & (
    ((d.dt.year % 2 == 0)) |
    ((d.dt.year % 2 == 1) & ((d.dt.month < 3) | ((d.dt.month == 3) & (d.dt.day < 4))))
)

session.loc[s1_phase1] = 1
session.loc[s2_phase1] = 2

m2 = (d >= pd.Timestamp("1933-03-04")) & (d < pd.Timestamp("1935-01-03"))
session.loc[m2 & (d.dt.year == 1933)] = 1
session.loc[m2 & (d.dt.year >= 1934)] = 2

m3 = d >= pd.Timestamp("1935-01-03")

s1_phase3 = m3 & (d.dt.year % 2 == 1) & (
    (d.dt.month > 1) | ((d.dt.month == 1) & (d.dt.day >= 3))
)
s2_phase3 = m3 & (
    ((d.dt.year % 2 == 0)) |
    ((d.dt.year % 2 == 1) & ((d.dt.month == 1) & (d.dt.day < 3)))
)

session.loc[s1_phase3] = 1
session.loc[s2_phase3] = 2

df["session"] = session

df = df.rename(columns={"speaker_bioguide": "bioguide_id"})
df.columns

df["origin_id"] = df["id"].astype(str) + ".chunk" + df["the_order"].astype(str)

df.head()

bio = pd.read_csv("BioID/bioguide_profiles.csv", sep=",", dtype=str)

bio = bio.rename(columns={
    "usCongressBioId": "bioguide_id",
    "givenName": "speaker_first",
    "familyName": "speaker_last"
})[["bioguide_id", "speaker_first", "speaker_last"]]

df = df.merge(bio, on="bioguide_id", how="left")

df.head()

df.columns

bio = pd.read_csv("BioID/bioguide_profiles.csv", sep=",", dtype=str)

def _split_pipes(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split("|")]

def _align_to_n(lst, n):
    """Return list of length n.
    - If empty, fill with None.
    - If length 1, broadcast that value to length n.
    - If shorter than n, repeat last value to reach n.
    - If longer than n, truncate.
    """
    lst = lst or []
    if n <= 0:
        return []
    if len(lst) == 0:
        return [None] * n
    if len(lst) == 1 and n > 1:
        return lst * n
    if len(lst) < n:
        return lst + [lst[-1]] * (n - len(lst))
    return lst[:n]

def build_bio_long(bio: pd.DataFrame) -> pd.DataFrame:
    b = bio.copy()
    if "bioguide_id" not in b.columns:
        b = b.rename(columns={"usCongressBioId": "bioguide_id"})

    b["cong_list"]  = b["congressNumber"].map(_split_pipes)
    b["party_list"] = b["party"].map(_split_pipes)
    b["state_list"] = b["regionCode"].map(_split_pipes)

    n = b["cong_list"].str.len().fillna(0).astype(int)
    b["party_list"] = [ _align_to_n(p, nn) for p, nn in zip(b["party_list"], n) ]
    b["state_list"] = [ _align_to_n(s, nn) for s, nn in zip(b["state_list"], n) ]

    b = b[["bioguide_id", "cong_list", "party_list", "state_list"]].explode(
        ["cong_list", "party_list", "state_list"], ignore_index=True
    )

    b["congress"] = pd.to_numeric(b["cong_list"], errors="coerce")
    b = b.rename(columns={
        "party_list": "speaker_party",
        "state_list": "speaker_state"
    })[["bioguide_id", "congress", "speaker_party", "speaker_state"]]

    return b

bio_long = build_bio_long(bio)
df = df.merge(bio_long, on=["bioguide_id", "congress"], how="left")

df["speaker_party"] = df["speaker_party"].replace({"Democrat": "D", "Republican": "R", "Independent": "I"})

df.head()

df.columns

df = df.rename(columns={"id": "doc_id"})

ROOT_HTML_DIR = Path(INPUT_DIR)

def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _expand_range(prefix: str, start: int, end: int) -> list[str]:
    """Inclusive numeric expansion: prefix + number for start..end."""
    if end < start:
        start, end = end, start
    return [f"{prefix} {i}" for i in range(start, end + 1)]

def _extract_category(raw: str, pair_patterns: list[re.Pattern], single_patterns: list[re.Pattern]) -> list[str]:
    """
    Extract items for a category.
    - pair_patterns capture (prefix, start, end)
    - single_patterns capture (prefix, num)
    """
    results = []

    consumed_spans = []
    for pat in pair_patterns:
        for m in pat.finditer(raw):
            prefix, a, b = m.group(1), m.group(2), m.group(3)
            consumed_spans.append(m.span())
            try:
                results.extend(_expand_range(prefix, int(a), int(b)))
            except ValueError:
                continue

    def _not_in_consumed(span):
        s, e = span
        for cs, ce in consumed_spans:
            if s >= cs and e <= ce:
                return False
        return True

    for pat in single_patterns:
        for m in pat.finditer(raw):
            if not _not_in_consumed(m.span()):
                continue
            prefix, n = m.group(1), m.group(2)
            results.append(f"{prefix} {int(n)}")

    return _dedupe_preserve_order(results)

BILLS_RANGE  = re.compile(r"\b((?:H\.?\s*R\.?|S\.))\s*(\d+)\s*-\s*(\d+)\b", re.IGNORECASE)
BILLS_SINGLE = re.compile(r"\b((?:H\.?\s*R\.?|S\.))\s*(\d+)\b", re.IGNORECASE)

JRES_RANGE   = re.compile(r"\b((?:H\.?\s*J\.?\s*Res\.?|S\.?\s*J\.?\s*Res\.?))\s*(\d+)\s*-\s*(\d+)\b", re.IGNORECASE)
JRES_SINGLE  = re.compile(r"\b((?:H\.?\s*J\.?\s*Res\.?|S\.?\s*J\.?\s*Res\.?))\s*(\d+)\b", re.IGNORECASE)

CRES_RANGE   = re.compile(r"\b((?:H\.?\s*Con\.?\s*Res\.?|S\.?\s*Con\.?\s*Res\.?))\s*(\d+)\s*-\s*(\d+)\b", re.IGNORECASE)
CRES_SINGLE  = re.compile(r"\b((?:H\.?\s*Con\.?\s*Res\.?|S\.?\s*Con\.?\s*Res\.?))\s*(\d+)\b", re.IGNORECASE)

SRES_RANGE   = re.compile(r"\b((?:H\.?\s*Res\.?|S\.?\s*Res\.?))\s*(\d+)\s*-\s*(\d+)\b", re.IGNORECASE)
SRES_SINGLE  = re.compile(r"\b((?:H\.?\s*Res\.?|S\.?\s*Res\.?))\s*(\d+)\b", re.IGNORECASE)

EXT_RANGE    = re.compile(r"\b(E)\s*(\d+)\s*-\s*(\d+)\b", re.IGNORECASE)
EXT_SINGLE   = re.compile(r"\b(E)\s*(\d+)\b", re.IGNORECASE)

def extract_citations_from_html(html_path: Path) -> dict:
    """
    Parse HTML and return:
    {
      'bills': 'H.R. 100, S. 200',
      'joint_resolution': 'H.J.Res. 10',
      'concurrent_resolution': 'S.Con.Res. 5',
      'simple_resolution': 'S.Res. 583, H.Res. 22',
      'extension_of_remarks': 'E1234, E1235'
    }
    """
    try:
        text = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {
            "bills": "",
            "joint_resolution": "",
            "concurrent_resolution": "",
            "simple_resolution": "",
            "extension_of_remarks": "",
        }

    soup = BeautifulSoup(text, "html.parser")
    raw = soup.get_text(" ", strip=True)

    def _normalise_prefixes(s: str) -> str:

        s = re.sub(r"\.\s+", ". ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s
    raw = _normalise_prefixes(raw)

    bills_list = _extract_category(raw, [BILLS_RANGE], [BILLS_SINGLE])
    jres_list  = _extract_category(raw, [JRES_RANGE],  [JRES_SINGLE])
    cres_list  = _extract_category(raw, [CRES_RANGE],  [CRES_SINGLE])
    sres_list  = _extract_category(raw, [SRES_RANGE],  [SRES_SINGLE])
    ext_list   = _extract_category(raw, [EXT_RANGE],   [EXT_SINGLE])

    ext_list   = [x.replace("E ", "E") for x in ext_list]

    return {
        "bills": ", ".join(bills_list),
        "joint_resolution": ", ".join(jres_list),
        "concurrent_resolution": ", ".join(cres_list),
        "simple_resolution": ", ".join(sres_list),
        "extension_of_remarks": ", ".join(ext_list),
    }

html_index = {}
for ext in ("*.htm", "*.html"):
    for p in ROOT_HTML_DIR.rglob(ext):
        key = p.stem
        if key not in html_index:
            html_index[key] = p

unique_ids = pd.Series(df["doc_id"].astype(str).unique())

records = []
for doc_id in unique_ids:
    html_path = html_index.get(doc_id)
    if html_path is None:
        rec = {
            "doc_id": doc_id,
            "bills": "",
            "joint_resolution": "",
            "concurrent_resolution": "",
            "simple_resolution": "",
            "extension_of_remarks": "",
        }
    else:
        rec = {"doc_id": doc_id}
        rec.update(extract_citations_from_html(html_path))
    records.append(rec)

found_df = pd.DataFrame.from_records(records)

df = df.merge(found_df, on="doc_id", how="left")

df.head()

df["pdf_url"] = df["origin_url"].str.replace("/html/", "/pdf/").str.replace(".htm", ".pdf")

df.head()

def infer_gender_from_raw(name: str):
    """Infer gender from speaker_raw prefix."""
    if pd.isna(name):
        return None
    n = name.strip().lower()
    if n.startswith("mr"):
        return "M"
    if n.startswith("mrs") or n.startswith("ms"):
        return "F"
    return None

df["speaker_gender"] = df["speaker_raw"].map(infer_gender_from_raw)

df.head()

df["bioguide_url"] = "https://bioguide.congress.gov/search/bio/" + df["bioguide_id"].astype(str)
df.head()

df = df.loc[:, ~df.columns.duplicated(keep="first")]

cols = [

    "speaker_state",
    "congress",
    "origin_url",
    "pdf_url",

    "chamber",
    "speaking",
    "speaker_party",
    "bills",
    "joint_resolution",
    "concurrent_resolution",
    "simple_resolution",
    "pages",
    "speaker_raw",
    "speaker_first",
    "speaker_last",

    "speaker_gender",      

    "volume",
    "session",

    "date",
    "bioguide_id",
    "bioguide_url",
    "origin_id"
]

other_cols = [c for c in df.columns if c not in cols]

df = df[cols + other_cols]

df["congress"] = np.floor(cong).astype("Int64")

df["origin_id"] = df["origin_id"].str.replace(r"\.chunk\d+$", "", regex=True)

df = df.reset_index(drop=True)

d = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce")

cong = pd.Series(np.nan, index=df.index, dtype="float")

m1 = (d >= pd.Timestamp("1789-03-04")) & (d < pd.Timestamp("1933-03-04"))
y1 = d.dt.year
odd1 = y1 % 2 == 1
pre_mar4 = odd1 & ((d.dt.month < 3) | ((d.dt.month == 3) & (d.dt.day < 4)))
start_odd1 = np.where(odd1, np.where(pre_mar4, y1 - 2, y1), y1 - 1)
cong.loc[m1] = (((start_odd1 - 1789) // 2) + 1)[m1]

m2 = (d >= pd.Timestamp("1933-03-04")) & (d < pd.Timestamp("1935-01-03"))
cong.loc[m2] = 73

m3 = d >= pd.Timestamp("1935-01-03")
y3 = d.dt.year
odd3 = y3 % 2 == 1
pre_jan3 = odd3 & (d.dt.month.eq(1) & d.dt.day.lt(3))
start_odd3 = np.where(odd3, np.where(pre_jan3, y3 - 2, y3), y3 - 1)
cong.loc[m3] = (((start_odd3 - 1935) // 2) + 74)[m3]

df["congress"] = np.floor(cong).astype("Int64")

df["year"] = pd.to_numeric(df["year"], errors="coerce")

year_value = int(df.loc[df.index[0], "year"])

output_folder = "Raw_CSVs"
os.makedirs(output_folder, exist_ok=True)

filename = f"Raw_speeches_{year_value}.csv"
filepath = os.path.join(output_folder, filename)

df.to_csv(filepath, index=False, encoding="utf-8")

print(f"Saved file: {filepath}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["year"] = pd.to_numeric(df["year"], errors="coerce")

df["row_no"] = df.groupby("year").cumcount().add(1).astype(str).str.zfill(2)

df["id"] = df["year"].astype(str) + df["row_no"]

df.drop(columns=["row_no"], inplace=True)
df.head()

df.shape

df["year"] = pd.to_numeric(df["year"], errors="coerce")

year_value = int(df.loc[df.index[0], "year"])

output_folder = "jsonl"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, f"speeches_{year_value}.jsonl")

cols = [
    "id",
    "speaker_state",
    "congress",
    "origin_url",
    "pdf_url",
    "chamber",
    "speaking",                
    "speaker_party",
    "bills",
    "joint_resolution",
    "concurrent_resolution",
    "simple_resolution",
    "pages",                    
    "speaker_raw",
    "speaker_first",
    "speaker_last",
    "speaker_gender",           
    "volume",                   
    "session",
    "date",
    "bioguide_id",
    "bioguide_url",
    "origin_id"                 
]

if "gender" in df.columns and "speaker_gender" not in df.columns:
    df["speaker_gender"] = df["gender"]

for col in cols:
    if col not in df.columns:
        df[col] = None

df_out = df[cols]

for c in df_out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
    df_out[c] = df_out[c].dt.strftime("%Y-%m-%d")

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df_out.iterrows():
        record = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        json_str = json.dumps(record, ensure_ascii=False)
        json_str = json_str.replace("\\/", "/")
        f.write(json_str + "\n")

print(f"JSONL saved to: {output_file}")

