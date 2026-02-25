import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
import multiprocessing
import logging

DATA_DIR = Path(r"data")
EVIDENCE_FILE = Path("evidence.txt")
INTUITION_FILE = Path("intuition.txt")

OUTPUT_EMI_CSV = Path("emi_scores_by_speech.csv")

VECTOR_SIZE = 300
WINDOW = 5
MIN_COUNT = 5
EPOCHS = 5

MIN_WORDS = 30

NUM_LENGTH_BINS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")

def tokenize(text: str) -> List[str]:
    """Lowercase and keep only alphabetic tokens."""
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens

def load_dictionary_words(path: Path) -> List[str]:
    """Load one word per line from a text file."""
    words = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.append(w)
    return words

def iter_speeches_tokens():
    """
    Generator over tokenised speeches that satisfy:
    - speaker_party in {'D', 'R'}
    - more than MIN_WORDS words
    Used as the underlying data source.
    """
    files = sorted(DATA_DIR.glob("speeches_*.jsonl"))
    for fp in files:
        with fp.open("r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                party = obj.get("speaker_party")
                if party not in {"D", "R"}:
                    continue

                speaking = obj.get("speaking", "")
                if not speaking:
                    continue

                tokens = tokenize(speaking)
                if len(tokens) <= MIN_WORDS:
                    continue

                yield tokens

class SpeechCorpus:
    """
    Re iterable wrapper around iter_speeches_tokens,
    so Gensim can do multiple passes.
    """
    def __iter__(self):
        for tokens in iter_speeches_tokens():
            yield tokens

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

def get_year_from_filename(path: Path) -> int:
    """Extract year from filename like speeches_1980.jsonl."""
    m = re.search(r"speeches_(\d{4})\.jsonl", path.name)
    if not m:
        return -1
    return int(m.group(1))

def train_word2vec_model() -> Word2Vec:
    """Train Word2Vec on all valid speeches."""
    logging.info("Training Word2Vec model on speeches")

    corpus = SpeechCorpus()

    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=multiprocessing.cpu_count()
    )

    logging.info("Building vocabulary")
    model.build_vocab(corpus)

    logging.info("Training model")
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=EPOCHS
    )

    logging.info("Word2Vec training complete")
    return model

def build_dictionary_vectors(
    model: Word2Vec,
    evidence_words: List[str],
    intuition_words: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build dictionary vectors by averaging embeddings
    for all dictionary words that are in the model vocabulary.
    """
    wv = model.wv

    evid_vecs = []
    evid_used = []
    for w in evidence_words:
        if w in wv:
            evid_vecs.append(wv[w])
            evid_used.append(w)

    intu_vecs = []
    intu_used = []
    for w in intuition_words:
        if w in wv:
            intu_vecs.append(wv[w])
            intu_used.append(w)

    if not evid_vecs:
        raise ValueError("No evidence words found in Word2Vec vocabulary")
    if not intu_vecs:
        raise ValueError("No intuition words found in Word2Vec vocabulary")

    evidence_vec = np.mean(np.stack(evid_vecs, axis=0), axis=0)
    intuition_vec = np.mean(np.stack(intu_vecs, axis=0), axis=0)

    logging.info("Evidence words used in embeddings: %d", len(evid_used))
    logging.info("Intuition words used in embeddings: %d", len(intu_used))

    return evidence_vec, intuition_vec, evid_used, intu_used

def compute_raw_similarities(
    model: Word2Vec,
    evidence_vec: np.ndarray,
    intuition_vec: np.ndarray
) -> pd.DataFrame:
    """
    For each valid speech, compute:
    - length (tokens)
    - cosine similarity to evidence and intuition vectors
    Store also id, year, party.
    """
    wv = model.wv
    files = sorted(DATA_DIR.glob("speeches_*.jsonl"))

    rows: List[Dict[str, Any]] = []

    logging.info("Computing cosine similarities for speeches")

    for fp in files:
        year = get_year_from_filename(fp)
        with fp.open("r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                party = obj.get("speaker_party")
                if party not in {"D", "R"}:
                    continue

                speaking = obj.get("speaking", "")
                if not speaking:
                    continue

                tokens = tokenize(speaking)
                if len(tokens) <= MIN_WORDS:
                    continue

                valid_tokens = [t for t in tokens if t in wv]
                if not valid_tokens:
                    continue

                vecs = np.stack([wv[t] for t in valid_tokens], axis=0)
                text_vec = np.mean(vecs, axis=0)

                cos_evid = cosine_similarity(text_vec, evidence_vec)
                cos_intu = cosine_similarity(text_vec, intuition_vec)

                rows.append(
                    {
                        "id": obj.get("id"),
                        "year": year,
                        "speaker_party": party,
                        "length": len(tokens),
                        "cos_evidence": cos_evid,
                        "cos_intuition": cos_intu,
                    }
                )

    df = pd.DataFrame(rows)
    logging.info("Computed similarities for %d speeches", len(df))
    return df

def length_correct_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Bin speeches by length.
    2. For each bin, subtract the mean cosine similarity of that bin
       from each speech similarity in the bin.
    3. Apply z transform separately to adjusted evidence and intuition scores.
    4. Compute EMI = z_evidence minus z_intuition.
    """
    logging.info("Applying length correction and z scoring")

    df = df.copy()

    df["length_bin"] = pd.qcut(
        df["length"],
        q=min(NUM_LENGTH_BINS, df["length"].nunique()),
        duplicates="drop"
    )

    for col in ["cos_evidence", "cos_intuition"]:
        mean_by_bin = df.groupby("length_bin")[col].transform("mean")
        df[f"{col}_adj"] = df[col] - mean_by_bin

    for col in ["cos_evidence_adj", "cos_intuition_adj"]:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        if sigma == 0:
            df[f"z_{col}"] = 0.0
        else:
            df[f"z_{col}"] = (df[col] - mu) / sigma

    df["z_evidence"] = df["z_cos_evidence_adj"]
    df["z_intuition"] = df["z_cos_intuition_adj"]

    df["emi"] = df["z_evidence"] - df["z_intuition"]

    return df

def main():

    logging.info("Loading dictionary files")
    evidence_words = load_dictionary_words(EVIDENCE_FILE)
    intuition_words = load_dictionary_words(INTUITION_FILE)
    logging.info("Loaded %d evidence words, %d intuition words",
                 len(evidence_words), len(intuition_words))

    model = train_word2vec_model()

    evidence_vec, intuition_vec, evid_used, intu_used = build_dictionary_vectors(
        model,
        evidence_words,
        intuition_words
    )
    logging.info("Dictionary vectors built")

    df_raw = compute_raw_similarities(model, evidence_vec, intuition_vec)
    if df_raw.empty:
        logging.error("No speeches passed the filters, nothing to compute")
        return

    df_emi = length_correct_and_standardise(df_raw)

    cols_to_save = [
        "id",
        "year",
        "speaker_party",
        "length",
        "cos_evidence",
        "cos_intuition",
        "cos_evidence_adj",
        "cos_intuition_adj",
        "z_evidence",
        "z_intuition",
        "emi",
    ]
    df_emi[cols_to_save].to_csv(OUTPUT_EMI_CSV, index=False)
    logging.info("Saved EMI scores to %s", OUTPUT_EMI_CSV)

    summary = df_emi.groupby(["year", "speaker_party"])["emi"].mean().reset_index()
    logging.info("Example EMI summary by year and party:")
    logging.info("\n%s", summary.head(20).to_string(index=False))

if __name__ == "__main__":
    main()

