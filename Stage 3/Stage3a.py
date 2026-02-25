import os
import sys
import pandas as pd
from datetime import datetime
import glob
import re
import numpy as np

INPUT_DIR = "Data" # raw data from Standford (https://data.stanford.edu/congress_text)
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

def iter_lines_any_encoding(path):
    """
    Yield decoded text lines from a file opened in binary mode.
    Each raw line is decoded with a cascade of encodings, so we never crash.
    """
    with open(path, "rb") as fb:
        for bline in fb:
            line = None
            for enc in ENCODINGS:
                try:
                    line = bline.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if line is None:

                line = bline.decode("utf-8", errors="replace")
            yield line

def normalize_date(s):
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def safe_split_first(line):

    line = line.replace("\ufeff", "")
    i = line.find("|")
    if i == -1:
        return None, None
    return line[:i].strip(), line[i+1:].rstrip("\r\n")

def safe_split_exact(line, expected_cols):
    """
    Split by '|' into exactly expected_cols pieces.
    Extra pieces are merged into the last column,
    missing pieces are padded with blanks.
    """
    parts = line.rstrip("\r\n").replace("\ufeff", "").split("|")
    if len(parts) < expected_cols:
        parts += [""] * (expected_cols - len(parts))
    elif len(parts) > expected_cols:
        head = parts[:expected_cols-1]
        tail = "|".join(parts[expected_cols-1:])
        parts = head + [tail]
    return parts

def coalesce(*vals):
    for v in vals:
        if v is not None:
            v = str(v).strip()
            if v and v.lower() != "unknown":
                return v
    return ""

speech_files, speaker_files, descr_files = [], [], []
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".txt"):
        continue
    path = os.path.join(INPUT_DIR, fname)
    if fname.startswith("speeches_"):
        speech_files.append(path)
    elif fname.endswith("SpeakerMap.txt"):
        speaker_files.append(path)
    elif fname.startswith("descr_"):
        descr_files.append(path)

if not speech_files:
    print("No speeches_*.txt files found", file=sys.stderr)
if not speaker_files:
    print("No *SpeakerMap.txt files found", file=sys.stderr)
if not descr_files:
    print("No descr_*.txt files found", file=sys.stderr)

speech_rows = []
bad_speech_lines = 0
for path in speech_files:
    for ln, raw in enumerate(iter_lines_any_encoding(path), 1):
        raw = raw.rstrip("\r\n")
        if not raw.strip():
            continue
        sid, text = safe_split_first(raw)
        if sid is None:
            bad_speech_lines += 1
            continue
        speech_rows.append((sid, text))

speeches_df = pd.DataFrame(speech_rows, columns=["speech_id", "text"], dtype=str)

speeches_df = speeches_df.drop_duplicates(subset=["speech_id"], keep="first")

speaker_cols = ["speakerid","speech_id","lastname","firstname","chamber","state","gender","party","district","nonvoting"]
speaker_rows = []
for path in speaker_files:
    for ln, raw in enumerate(iter_lines_any_encoding(path), 1):
        raw = raw.rstrip("\r\n")
        if not raw.strip():
            continue
        parts = safe_split_exact(raw, len(speaker_cols))
        if len(parts) != len(speaker_cols):
            continue
        speaker_rows.append(parts)

speakers_df = pd.DataFrame(speaker_rows, columns=speaker_cols, dtype=str)
speakers_df.rename(columns={"speakerid": "bioguide_id"}, inplace=True)

descr_cols = [
    "speech_id","chamber","date","number_within_file","speaker","first_name",
    "last_name","state","gender","line_start","line_end","file","char_count","word_count"
]
descr_rows = []
for path in descr_files:
    for ln, raw in enumerate(iter_lines_any_encoding(path), 1):
        raw = raw.rstrip("\r\n")
        if not raw.strip():
            continue
        parts = safe_split_exact(raw, len(descr_cols))
        if len(parts) != len(descr_cols):
            continue
        descr_rows.append(parts)

descr_df = pd.DataFrame(descr_rows, columns=descr_cols, dtype=str)

merged = descr_df.merge(speeches_df, on="speech_id", how="left")
merged = merged.merge(speakers_df, on="speech_id", how="left", suffixes=("", "_spk"))

merged["first_name_final"] = merged.apply(lambda r: coalesce(r.get("first_name"), r.get("firstname")), axis=1)
merged["last_name_final"]  = merged.apply(lambda r: coalesce(r.get("last_name"),  r.get("lastname")),  axis=1)
merged["gender_final"]     = merged.apply(lambda r: coalesce(r.get("gender"),     r.get("gender_spk")), axis=1)
merged["party_final"]      = merged.apply(lambda r: coalesce(r.get("party"),      r.get("party_spk")),  axis=1)
merged["chamber_final"]    = merged.apply(lambda r: coalesce(r.get("chamber"),    r.get("chamber_spk")),axis=1)

merged["date"] = merged["date"].map(normalize_date)
merged["year"] = merged["date"].str[:4].fillna("")

for c in ["text","speaker","bioguide_id","first_name_final","last_name_final","gender_final","party_final","chamber_final"]:
    if c not in merged.columns:
        merged[c] = ""
    else:
        merged[c] = merged[c].fillna("")

merged["middle_name"] = ""

final = merged.rename(columns={
    "first_name_final": "first_name",
    "last_name_final":  "last_name",
    "gender_final":     "gender",
    "party_final":      "party",
    "chamber_final":    "chamber"
})[[
    "speech_id","text","speaker","bioguide_id","date",
    "last_name","first_name","middle_name","gender","party","chamber","year"
]]

written = 0
for year, grp in final.groupby("year", dropna=True):
    year_str = str(year)
    if not year_str.isdigit() or len(year_str) != 4:
        continue
    out_path = os.path.join(OUTPUT_DIR, f"{year_str}.csv")
    grp.to_csv(out_path, index=False, encoding="utf-8")
    written += 1
    print(f"Wrote {out_path} , rows={len(grp)}")

print(
    f"Summary, speeches={len(speeches_df)}, speaker_map_rows={len(speakers_df)}, "
    f"descr_rows={len(descr_df)}, bad_speech_lines={bad_speech_lines}, csv_written={written}"
)

IN_DIR = "Output"
OUT_DIR = "Output_Clean"
os.makedirs(OUT_DIR, exist_ok=True)

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

def read_csv_any_encoding(path):
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    base_order, groups = [], {}
    for c in cols:
        base = re.sub(r"\.\d+$", "", c)
        if base not in groups:
            groups[base] = []
            base_order.append(base)
        groups[base].append(c)

    out = pd.DataFrame(index=df.index)
    for base in base_order:
        same = groups[base]
        sub = df[same].astype(str)
        sub = sub.replace({"nan": "", "NaN": "", "None": "", "none": "", "NULL": "", "null": ""})
        sub = sub.replace("", np.nan)
        merged = sub.bfill(axis=1).iloc[:, 0].fillna("")
        out[base] = merged
    return out

def canonicalise_party(series: pd.Series) -> pd.Series:
    """
    Map many variants to 'Republican' or 'Democrat'. Otherwise empty.
    Examples handled: R, D, Rep, Dem, Republican, Democrat, GOP, Democratic,
    tokens like 'R-ME' or 'Dem.,' etc.
    """
    if series is None:
        return pd.Series([], dtype=str)

    s = series.fillna("").astype(str).str.strip()

    token = (
        s.str.replace(r"[^A-Za-z\-]+", "", regex=True)
         .str.split("-", n=1, expand=False)
         .str[0]
         .str.lower()
    )

    fallback = s.str.lower().str.replace(r"[^a-z]+", "", regex=True)

    base = token.where(token.ne(""), fallback)

    rep_set = {"r", "rep", "repub", "republican", "gop"}
    dem_set = {"d", "dem", "democrat", "democratic"}

    mapped = np.where(
        base.isin(rep_set), "Republican",
        np.where(base.isin(dem_set), "Democrat", "")
    )
    return pd.Series(mapped, index=series.index, dtype=str)

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:

    df = collapse_duplicate_columns(df)

    if "gender" in df.columns:
        df = df[~df["gender"].fillna("").str.strip().str.casefold().eq("special")]

    if "party" in df.columns:
        df["party"] = canonicalise_party(df["party"])
        df = df[df["party"].isin(["Republican", "Democrat"])]

    if "bioguide_id" not in df.columns:
        df["bioguide_id"] = ""
    else:
        df["bioguide_id"] = ""

    df = df.replace({np.nan: ""})
    df = df.replace(["None", "none", "NaN", "nan", "NULL", "null"], "", regex=False)

    return df

def process_all():
    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not paths:
        print(f"No CSV files found in {IN_DIR}")
        return

    for path in paths:
        try:
            df = read_csv_any_encoding(path)
            cleaned = clean_frame(df)
            out_path = os.path.join(OUT_DIR, os.path.basename(path))
            cleaned.to_csv(out_path, index=False, encoding="utf-8")
            print(
                f"Cleaned, {os.path.basename(path)} , "
                f"rows in={len(df)} , rows out={len(cleaned)} , cols={len(cleaned.columns)}"
            )
        except Exception as e:
            print(f"Failed, {os.path.basename(path)} , reason={e}")

if __name__ == "__main__":
    process_all()

