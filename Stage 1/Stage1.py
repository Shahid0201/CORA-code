import pandas as pd
import re
from bs4 import BeautifulSoup
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
from itertools import zip_longest
import json


run_year = "1994"
run_year = str(run_year)


Input_dir = "3-Garcia_et_al_2025_Extracted Data/" #Extracted data from Garacia et al 2025 repository

filename = Input_dir + run_year + ".csv"

df = pd.read_csv(filename)
df.head()

df.shape

df = df.rename(columns={
    "text": "speaking",
    "speaker": "speaker_raw",
    "last_name": "speaker_last",
    "first_name": "speaker_first",
    "middle_name": "speaker_middle",
    "party": "speaker_party"
})
df.head()

df['gender'].unique().tolist()

cols = ["gender"]

df[cols] = (
    df[cols]
    .apply(lambda col: col.where(~col.astype(str).str.lower().isin(["unknown", "special"]), ""))
)

df['gender'].unique().tolist()

df['speaker_party'].unique().tolist()

df['speaker_party'] = df['speaker_party'].str.lower().map({"democrat": "D", "republican": "R", "independent": "I"})
df['speaker_party'].unique().tolist()

df['chamber'].unique().tolist()

df['chamber'] = df['chamber'].replace({"H": "House", "S": "Senate", "E": "Extension"})
df['chamber'].unique().tolist()

section_map = {"House": "house-section",
               "Senate": "senate-section",
               "Extension": "extensions-of-remarks-section"}

df["origin_url"] = (
    "https://www.congress.gov/bound-congressional-record/"
    + pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y/%m/%d")
    + "/"
    + df["chamber"].map(section_map)
)
df['origin_url']

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

def _normalise_prefixes(s: str) -> str:

    s = re.sub(r"\.\s+", ". ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def extract_citations_from_text(raw_text: str) -> dict:
    """
    Parse plain or HTML-ish text and return five fields:
      - 'bills'
      - 'joint_resolution'
      - 'concurrent_resolution'
      - 'simple_resolution'
      - 'extension_of_remarks'
    """
    if not isinstance(raw_text, str):
        raw_text = "" if pd.isna(raw_text) else str(raw_text)

    try:
        raw = BeautifulSoup(raw_text, "html.parser").get_text(" ", strip=True)
    except Exception:
        raw = raw_text

    raw = _normalise_prefixes(raw)

    bills_list = _extract_category(raw, [BILLS_RANGE], [BILLS_SINGLE])
    jres_list  = _extract_category(raw, [JRES_RANGE],  [JRES_SINGLE])
    cres_list  = _extract_category(raw, [CRES_RANGE],  [CRES_SINGLE])
    sres_list  = _extract_category(raw, [SRES_RANGE],  [SRES_SINGLE])
    ext_list   = _extract_category(raw, [EXT_RANGE],   [EXT_SINGLE])

    ext_list = [x.replace("E ", "E") for x in ext_list]

    return {
        "bills": ", ".join(bills_list),
        "joint_resolution": ", ".join(jres_list),
        "concurrent_resolution": ", ".join(cres_list),
        "simple_resolution": ", ".join(sres_list),
        "extension_of_remarks": ", ".join(ext_list),
    }

parsed = df["speaking"].apply(extract_citations_from_text).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

df.head()

df['bills'].head(100).unique().tolist()

cols = ["speaker_first", "speaker_last", "speaker_middle"]

df[cols] = (
    df[cols]
    .apply(lambda col: col.where(
        col.astype(str).str.lower() != "unknown", np.nan
    ))
)

df.head()

df.columns

df.shape

d = pd.to_datetime(df["date"], errors="coerce")

congress = pd.Series(np.nan, index=df.index, dtype="float")

m1 = (d >= pd.Timestamp("1879-01-01")) & (d < pd.Timestamp("1933-03-04"))
base1 = 46 + ((d.dt.year - 1879) // 2)
pre_start1 = (d.dt.year % 2 == 1) & ((d.dt.month < 3) | ((d.dt.month == 3) & (d.dt.day < 4)))
congress.loc[m1] = (base1 - pre_start1.astype(int)).loc[m1]
congress.loc[(d < pd.Timestamp("1879-03-04")) & (d.dt.year == 1879)] = 45

m2 = (d >= pd.Timestamp("1933-03-04")) & (d < pd.Timestamp("1935-01-03"))
congress.loc[m2] = 73

m3 = (d >= pd.Timestamp("1935-01-03"))
base3 = 74 + ((d.dt.year - 1935) // 2)
pre_start3 = (d.dt.year % 2 == 1) & ((d.dt.month == 1) & (d.dt.day < 3))
congress.loc[m3] = (base3 - pre_start3.astype(int)).loc[m3]

df["congress"] = congress.astype("Int64")

df.head()

d = pd.to_datetime(df["date"], errors="coerce")
session = pd.Series(pd.NA, index=df.index, dtype="Int64")

m1 = (d >= pd.Timestamp("1879-03-04")) & (d < pd.Timestamp("1933-03-04"))
odd_year  = d.dt.year % 2 == 1
even_year = ~odd_year

s1_m1 = m1 & odd_year & ((d.dt.month > 3) | ((d.dt.month == 3) & (d.dt.day >= 4)))
s2_m1 = m1 & even_year
session.loc[s1_m1] = 1
session.loc[s2_m1] = 2

session.loc[(d < pd.Timestamp("1879-03-04")) & (d.dt.year == 1879)] = 2

m2 = (d >= pd.Timestamp("1933-03-04")) & (d < pd.Timestamp("1935-01-03"))
session.loc[m2 & (d.dt.year == 1933)] = 1
session.loc[m2 & (d.dt.year >= 1934)] = 2

m3 = (d >= pd.Timestamp("1935-01-03"))
session.loc[m3 & odd_year]  = 1
session.loc[m3 & even_year] = 2

window = (d >= pd.Timestamp("1879-01-01")) & (d < pd.Timestamp("1995-01-03"))
session = session.where(window)

df["session"] = session

df.head(1000).to_csv('test_for_TR.csv', index=False)

df['origin_url'].nunique()

BIOGUIDE_PATH = Path("BioID/bioguide_profiles.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
SCORE_THRESHOLD = 0.80

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^a-z0-9 ]")
_apos = re.compile(r"[’'`]")
TITLES = re.compile(
    r"^(mr|mrs|ms|miss|dr|rep|representative|sen|senator|del|delegate|resident commissioner|speaker)\.?\s+",
    flags=re.I,
)

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _ws.sub(" ", s.strip())

def norm_name(s: str) -> str:
    s = clean_text(s).lower()
    s = s.replace(".", " ")
    s = _apos.sub("", s)
    s = _punct.sub(" ", s)
    return _ws.sub(" ", s).strip()

def strip_titles(s: str) -> str:
    s = clean_text(s)
    s = s.replace(".", " ")
    s = TITLES.sub("", s).strip()
    return _ws.sub(" ", s)

def split_pipe_field(s: str) -> List[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    return [part.strip() for part in str(s).split("|")]

def to_int_or_none(s: str) -> Optional[int]:
    try:
        return int(str(s).strip())
    except Exception:
        return None

def normalise_party(p: str) -> str:
    if pd.isna(p):
        return ""
    p = str(p).strip()
    mapping = {
        "D": "Democrat",
        "R": "Republican",
        "I": "Independent",
        "Democratic": "Democrat",
        "Democratic Republican": "Democratic Republican",
        "Crawford Republican": "Crawford Republican",
        "Republican": "Republican",
        "Democrat": "Democrat",
        "Whig": "Whig",
        "Independent": "Independent",
    }
    return mapping.get(p, p)

def build_bio_variants(given: str, middle: str, family: str, nick: str) -> List[str]:
    g = norm_name(given)
    m = norm_name(middle)
    f = norm_name(family)
    n = norm_name(nick)
    variants = set()
    if g and f:
        variants.add(f"{g} {f}")
    if g and m and f:
        variants.add(f"{g} {m} {f}")
    if n and f:
        variants.add(f"{n} {f}")
    return [v for v in variants if v]

def build_query_variants(first: str, last: str, raw: str) -> List[str]:
    out = set()
    f = norm_name(first)
    l = norm_name(last)
    if f and l:
        out.add(f"{f} {l}")
    raw_clean = norm_name(strip_titles(raw))
    if raw_clean:
        out.add(raw_clean)
    return [x for x in out if x]

bio = pd.read_csv(BIOGUIDE_PATH, dtype=str, keep_default_na=False)

rows = []
for _, r in bio.iterrows():
    congress_list = split_pipe_field(r.get("congressNumber", ""))
    party_list = [normalise_party(x) for x in split_pipe_field(r.get("party", ""))] or [""]
    if not congress_list:
        congress_list = [""]

    for c in congress_list:
        for p in party_list:
            rows.append({
                "bioguide_id": (r.get("usCongressBioId", "") or "").strip(),
                "given": r.get("givenName", "") or "",
                "middle": r.get("middleName", "") or "",
                "family": r.get("familyName", "") or "",
                "nick": r.get("nickName", "") or "",
                "congress": to_int_or_none(c),
                "party_norm": p,
                "gender": (r.get("gender", "") or "").strip(),
                "last_norm": norm_name(r.get("familyName", "") or ""),
            })

bio_flat = pd.DataFrame(rows)
bio_flat["variants"] = bio_flat.apply(
    lambda r: build_bio_variants(r["given"], r["middle"], r["family"], r["nick"]), axis=1
)
bio_flat = bio_flat.explode("variants").dropna(subset=["variants"]).reset_index(drop=True)

model = SentenceTransformer(MODEL_NAME)

@lru_cache(maxsize=None)
def embed_cached(text: str) -> np.ndarray:
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

def best_similarity(query_variants: List[str], cand_variants: List[str]) -> float:
    if not query_variants or not cand_variants:
        return 0.0
    q_embs = np.vstack([embed_cached(q) for q in query_variants])
    c_embs = np.vstack([embed_cached(c) for c in cand_variants])
    sims = cosine_similarity(q_embs, c_embs)
    return float(np.max(sims))

@lru_cache(maxsize=None)
def bucket_candidates(congress: int, party_norm: str) -> pd.DataFrame:
    """Filter by congress first, then party if present."""
    sub = bio_flat[bio_flat["congress"] == congress]
    if party_norm:
        sub_party = sub[sub["party_norm"] == party_norm]
        if not sub_party.empty:
            sub = sub_party
    return sub.reset_index(drop=True)

memo_best_for_key = {}

def fill_bioguide_ids_inplace(df: pd.DataFrame,
                              score_threshold: float = SCORE_THRESHOLD) -> None:
    """Mutates df in place to fill bioguide_id and bioguide_id_below_threshold."""

    if "bioguide_id" not in df.columns:
        df["bioguide_id"] = ""

    df["congress"] = df["congress"].apply(lambda x: to_int_or_none(x))
    df["party_norm"] = df.get("speaker_party", "").map(normalise_party)
    df["gender_norm"] = df.get("gender", "").astype(str).str.strip()

    df["first_norm"] = df.get("speaker_first", "").map(norm_name)
    df["last_norm"] = df.get("speaker_last", "").map(norm_name)
    df["raw_norm"] = df.get("speaker_raw", "").map(strip_titles).map(norm_name)

    if "bioguide_id_below_threshold" not in df.columns:
        df["bioguide_id_below_threshold"] = ""
    if "match_score" not in df.columns:
        df["match_score"] = ""

    df["bioguide_id"] = df["bioguide_id"].fillna("")
    total_blank_init = int(((df["bioguide_id"].isna()) | (df["bioguide_id"].astype(str).str.strip() == "")).sum())
    found_above = 0
    not_found = 0

    for i, row in df.iterrows():
        if str(row["bioguide_id"]).strip():
            continue

        cong = row["congress"]
        if pd.isna(cong):
            not_found += 1
            continue

        party_norm = row["party_norm"]
        gender_norm = row["gender_norm"]
        key = (int(cong), party_norm, gender_norm[:1].upper() if gender_norm else "", row["first_norm"], row["last_norm"], row["raw_norm"])

        if key in memo_best_for_key:
            best_id, best_score, below_id = memo_best_for_key[key]
            if best_id:
                df.at[i, "bioguide_id"] = best_id
                df.at[i, "match_score"] = round(best_score, 4)
                found_above += 1
            elif below_id:
                df.at[i, "bioguide_id_below_threshold"] = below_id
                df.at[i, "match_score"] = round(best_score, 4)
                not_found += 1
            else:
                not_found += 1
            continue

        cand = bucket_candidates(int(cong), party_norm)
        if cand.empty:
            memo_best_for_key[key] = (None, None, None)
            not_found += 1
            continue

        g = gender_norm
        if g:
            have_gender = cand["gender"].astype(str).str.len() > 0
            if have_gender.any():
                mask = cand["gender"].str[0].str.upper() == g[0].upper()
                cand_g = cand[mask]
                if not cand_g.empty:
                    cand = cand_g

        if row["last_norm"]:
            same_last = cand[cand["last_norm"] == row["last_norm"]]
            if not same_last.empty:
                cand = same_last

        q_variants = build_query_variants(row.get("speaker_first", ""), row.get("speaker_last", ""), row.get("speaker_raw", ""))
        if not q_variants:
            memo_best_for_key[key] = (None, None, None)
            not_found += 1
            continue

        best_id = None
        best_score = -1.0
        for bid, grp in cand.groupby("bioguide_id"):
            score = best_similarity(q_variants, grp["variants"].tolist())
            if score > best_score:
                best_score = score
                best_id = bid

        if best_score >= score_threshold:
            df.at[i, "bioguide_id"] = best_id
            df.at[i, "match_score"] = round(best_score, 4)
            memo_best_for_key[key] = (best_id, best_score, None)
            found_above += 1
        else:
            df.at[i, "bioguide_id_below_threshold"] = best_id or ""
            df.at[i, "match_score"] = "" if best_score < 0 else round(best_score, 4)
            memo_best_for_key[key] = (None, best_score, best_id)
            not_found += 1

    print(f"Total number of blank bioguide_id initially, {total_blank_init}")
    print(f"Total number of bioguide_id found at or above threshold {SCORE_THRESHOLD}, {found_above}")
    print(f"Total number of bioguide_id not found or below threshold, {not_found}")
    return total_blank_init, found_above, not_found

fill_bioguide_ids_inplace(df, score_threshold=SCORE_THRESHOLD)

def is_blank(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip().str.lower()
    return col.isna() | s.isin(["", "nan", "none", "null"])

ms_blank   = is_blank(df["match_score"])
below_blank = is_blank(df["bioguide_id_below_threshold"])
id_blank   = is_blank(df["bioguide_id"])

found_above = int((~ms_blank & below_blank).sum())
found_below = int((~below_blank).sum())
still_blank_now = int((id_blank).sum())

initial_blank = int(((~ms_blank & below_blank) | (~below_blank) | (id_blank)).sum())

not_found_or_below = int(initial_blank - found_above)

out_dir = "Bioguides_found"
os.makedirs(out_dir, exist_ok=True)
year_val = str(df["year"].iloc[0])

summary_df = pd.DataFrame(
    [
        ["Total number of blank bioguide_id initially", initial_blank],
        [f"Total number of bioguide_id found at or above threshold {SCORE_THRESHOLD}", found_above],
        ["Total number of bioguide_id not found or below threshold", not_found_or_below],
    ],
    columns=["description", "value"]
)

out_path = os.path.join(out_dir, f"summary_{year_val}.csv")
summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Summary saved to {out_path}")

bio = pd.read_csv("BioID/bioguide_profiles.csv", dtype=str, keep_default_na=False)

def split_pipe(s):
    return [p.strip() for p in str(s).split("|")] if str(s).strip() else []

rows = []
for _, r in bio.iterrows():
    cs = split_pipe(r.get("congressNumber", ""))
    rs = split_pipe(r.get("regionCode", ""))
    if len(cs) == len(rs) and cs:
        pairs = zip(cs, rs)
    elif cs and len(rs) == 1:
        pairs = [(c, rs[0]) for c in cs]
    elif rs and len(cs) == 1:
        pairs = [(cs[0], s) for s in rs]
    else:
        pairs = []
    for c, s in pairs:
        rows.append({
            "bioguide_id": (r.get("usCongressBioId", "") or "").strip(),
            "congress": int(str(c).strip()),
            "speaker_state": s.strip(),
        })

bio_states = pd.DataFrame(rows)

df["congress"] = pd.to_numeric(df["congress"], errors="coerce").astype("Int64")
df = df.merge(bio_states, how="left", on=["bioguide_id", "congress"])

df.head()

df.to_csv('del1.csv', index=False)

df.columns

    NoSuchElementException,
    TimeoutException,
    WebDriverException,
    StaleElementReferenceException,
)

options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920x1080")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
)

driver = webdriver.Chrome(options=options)

cache = {}
x_anchor = "/html/body/div[2]/div/main/div[2]/div/p/a"

def fetch_one(u, max_retries=3, base_wait=4.0):
    for attempt in range(max_retries + 1):
        try:
            driver.get(u)

            try:
                el = WebDriverWait(driver, 6).until(
                    EC.presence_of_element_located((By.XPATH, x_anchor))
                )
            except TimeoutException:

                try:
                    el = driver.find_element(By.XPATH, x_anchor)
                except NoSuchElementException:
                    el = None

            if el is None:
                return {"pdf_raw_url": None, "vol_raw": None}

            try:
                href_val = el.get_attribute("href")
                text_val = (el.text or "").strip()
            except StaleElementReferenceException:

                try:
                    el = driver.find_element(By.XPATH, x_anchor)
                    href_val = el.get_attribute("href")
                    text_val = (el.text or "").strip()
                except Exception:
                    href_val, text_val = None, None

            return {"pdf_raw_url": href_val, "vol_raw": text_val}

        except WebDriverException as e:

            if attempt < max_retries:
                time.sleep(base_wait * (2 ** attempt))
                continue
            else:
                print(f"Error loading {u}: {e}")
                return {"pdf_raw_url": None, "vol_raw": None}

try:

    unique_urls = (
        df["origin_url"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.startswith(("http://", "https://"))]
        .unique()
    )

    for url in unique_urls:
        res = fetch_one(url)
        cache[url] = res

        time.sleep(0.3)

    df["pdf_raw_url"] = df["origin_url"].map(lambda u: cache.get(u, {}).get("pdf_raw_url"))
    df["vol_raw"]     = df["origin_url"].map(lambda u: cache.get(u, {}).get("vol_raw"))
finally:

    try:
        driver.quit()
    except Exception:
        pass

df.head()

df[df["pdf_raw_url"].notna() & (df["pdf_raw_url"] != None) & (df["pdf_raw_url"] != "")].head()

df.to_csv('del2.csv', index=False)

df["volume"] = df["vol_raw"].str.extract(r"Vol\.(\d+)", expand=False)
df["pages"] = df["vol_raw"].str.extract(r"pages\s+([\d\s\-]+)", expand=False)

df.head()

df["origin_id"] = df["pdf_raw_url"].str.extract(r"/([^/]+)\.pdf$", expand=False)

df.head()

bio_path = "BioID/bioguide_profiles.csv"

bio = pd.read_csv(bio_path, dtype=str).rename(
    columns={
        "usCongressBioId": "bioguide_id",
        "congressNumber": "congress_list",
        "regionCode": "state_list",
        "party": "party_list",
    }
)[["bioguide_id", "congress_list", "state_list", "party_list"]]

def split_pipes(s):
    if pd.isna(s):
        return []

    return [x.strip() for x in re.split(r"\s*\|\s*", str(s)) if x.strip()]

records = []
for _, row in bio.iterrows():
    bid = row["bioguide_id"]
    c_list = split_pipes(row["congress_list"])
    s_list = split_pipes(row["state_list"])
    p_list = split_pipes(row["party_list"])

    def pad_with_last(values, target_len):
        if not values:
            return [None] * target_len
        out = []
        last = values[0]
        for i in range(target_len):
            if i < len(values) and values[i] is not None and values[i] != "":
                last = values[i]
            out.append(last)
        return out

    max_len = max(len(c_list), len(s_list), len(p_list), 1)
    c_list = pad_with_last(c_list, max_len)
    s_list = pad_with_last(s_list, max_len)
    p_list = pad_with_last(p_list, max_len)

    for c, st, pr in zip(c_list, s_list, p_list):

        try:
            c_int = int(str(c).strip())
        except Exception:
            continue
        records.append(
            {
                "bioguide_id": bid,
                "congress": c_int,
                "bio_state": st if st else None,
                "bio_party_full": pr if pr else None,
            }
        )

bio_long = pd.DataFrame.from_records(records)

def party_to_letter(p):
    if p is None:
        return None
    p_low = p.lower()

    if "democrat" in p_low:
        return "D"
    if "republican" in p_low:
        return "R"
    if "independent" in p_low:
        return "I"

    if "whig" in p_low:
        return "W"
    if "federalist" in p_low:
        return "F"
    if "populist" in p_low:
        return "P"
    if "progressive" in p_low:
        return "P"
    if "union" in p_low and "unionist" in p_low:
        return "U"
    if "democratic-republican" in p_low or "democratic republican" in p_low:
        return "DR"

    return p.strip()[0].upper() if p.strip() else None

bio_long["bio_party"] = bio_long["bio_party_full"].map(party_to_letter)

df["congress"] = pd.to_numeric(df["congress"], errors="coerce").astype("Int64")

df = df.merge(
    bio_long[["bioguide_id", "congress", "bio_state", "bio_party"]],
    on=["bioguide_id", "congress"],
    how="left",
)

df["speaker_state_bio"] = df["bio_state"]
df["speaker_party_bio"] = df["bio_party"]

def is_missing(x):
    return pd.isna(x) or str(x).strip() == ""

df["speaker_state"] = df["speaker_state"].where(~df["speaker_state"].map(is_missing), df["speaker_state_bio"])
df["speaker_party"] = df["speaker_party"].where(~df["speaker_party"].map(is_missing), df["speaker_party_bio"])

df.head()

def is_missing(x):
    return pd.isna(x) or str(x).strip() == ""

bio_names = (
    pd.read_csv("BioID/bioguide_profiles.csv", dtype=str)
      .rename(columns={"usCongressBioId": "bioguide_id"})
      [["bioguide_id", "familyName", "givenName", "middleName", "nickName"]]
)

for col in ["familyName", "givenName", "middleName", "nickName"]:
    bio_names[col] = bio_names[col].where(~bio_names[col].map(is_missing), None)

bio_names["given_or_nick"] = bio_names["givenName"].fillna(bio_names["nickName"])

def to_title(s):
    if s is None:
        return None

    s = re.sub(r"\s+", " ", s.strip())

    parts = re.split(r"([\'\-])", s)
    parts = [p.title() if i % 2 == 0 else p for i, p in enumerate(parts)]
    return "".join(parts)

names_fmt = bio_names.assign(
    speaker_last_bio   = bio_names["familyName"].map(lambda s: s.strip().upper() if s else None),
    speaker_first_bio  = bio_names["given_or_nick"].map(to_title),
    speaker_middle_bio = bio_names["middleName"].map(to_title),
)[["bioguide_id", "speaker_last_bio", "speaker_first_bio", "speaker_middle_bio"]].drop_duplicates("bioguide_id")

df = df.merge(names_fmt, on="bioguide_id", how="left")

df["speaker_last"]   = df["speaker_last"].where(~df["speaker_last"].map(is_missing),   df["speaker_last_bio"])
df["speaker_first"]  = df["speaker_first"].where(~df["speaker_first"].map(is_missing), df["speaker_first_bio"])
df["speaker_middle"] = df["speaker_middle"].where(~df["speaker_middle"].map(is_missing), df["speaker_middle_bio"])

df.head()

bio_gender = pd.read_csv(bio_path, dtype=str)[["usCongressBioId", "honorificPrefix"]].rename(
    columns={"usCongressBioId": "bioguide_id"}
)

def honorific_to_gender(honorific):
    if pd.isna(honorific):
        return None
    h = honorific.strip().lower()
    if h.startswith("mr"):
        return "M"
    if h.startswith("ms") or h.startswith("mrs"):
        return "F"
    return None

bio_gender["bio_gender"] = bio_gender["honorificPrefix"].map(honorific_to_gender)

df = df.merge(
    bio_gender[["bioguide_id", "bio_gender"]],
    on="bioguide_id",
    how="left"
)

def is_missing(x):
    return pd.isna(x) or str(x).strip() == ""

df["gender"] = df["gender"].where(~df["gender"].map(is_missing), df["bio_gender"])

df.rename(columns={"pdf_raw_url": "pdf_url"}, inplace=True)

df["bioguide_url"] = np.where(
    df["bioguide_id"].notna() & (df["bioguide_id"].str.strip() != ""),
    "https://bioguide.congress.gov/search/bio/" + df["bioguide_id"].astype(str),
    ""
)

df["year"] = pd.to_numeric(df["year"], errors="coerce")

year_value = int(df.loc[df.index[0], "year"])

output_folder = "Raw_CSVs"
os.makedirs(output_folder, exist_ok=True)

filename = f"Raw_speeches_{year_value}.csv"
filepath = os.path.join(output_folder, filename)

df.to_csv(filepath, index=False, encoding="utf-8")

print(f"Saved file: {filepath}")

df.columns

df.shape

df["year"] = pd.to_numeric(df["year"], errors="coerce")

df["row_no"] = df.groupby("year").cumcount().add(1).astype(str).str.zfill(2)

df["id"] = df["year"].astype(str) + df["row_no"]

df.drop(columns=["row_no"], inplace=True)
df.head()

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

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df_out.iterrows():

        record = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

print(f"JSONL saved to: {output_file}")

