import glob, json
import pandas as pd
import nltk
from nltk.corpus import opinion_lexicon
import spacy
from scipy.stats import zscore

files = glob.glob("data/speeches_*.jsonl")

rows = []
for fp in files:

    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:

                continue

df = pd.DataFrame(rows)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

df = df[
    (df['chamber'] == 'Senate') &
    (df['year'].between(1989, 2006)) &
    (df['speaker_gender'].isin(['M', 'F']))
].copy()

df = df[df['speaking'].str.split().str.len() >= 50]

print(len(df))
print(df['speaker_gender'].value_counts(normalize=True))
print(df.groupby(['year','speaker_gender']).size().unstack(fill_value=0))

PPRON = {
    "i","me","my","mine","myself",
    "we","us","our","ours","ourselves",
    "you","your","yours","yourself","yourselves",
    "he","him","his","himself",
    "she","her","hers","herself",
    "they","them","their","theirs","themselves"
}

NEGATE = {"no","not","never","none","nothing","nowhere","neither","hardly","scarcely","barely"}
CERTAIN = {"always","never","definitely","certainly","clearly","obviously","undoubtedly","surely","absolutely","of course"}

COGPROC = {"think","know","believe","because","why","reason","understand","realize","consider","decide","idea","mind"}

ARTICLES = {"a","an","the"}

nltk.download("opinion_lexicon")

POS_EMO = set(opinion_lexicon.positive())
NEG_EMO = set(opinion_lexicon.negative())

ANX = {"afraid","anxious","worried","nervous","scared","fear","fearful","terrified","panic","panicked"}

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def speech_features(text):
    doc = nlp(text)
    n_tokens = sum(1 for t in doc if t.is_alpha)
    if n_tokens == 0:
        return {k: 0.0 for k in [
            "ppron","verb","auxverb","negemo","anx",
            "negate","certain","cogproc","article","preps",
            "posemo","sixltr"
        ]}
    counts = dict.fromkeys([
        "ppron","verb","auxverb","negemo","anx",
        "negate","certain","cogproc","article","preps",
        "posemo","sixltr"
    ], 0)

    for tok in doc:
        if not tok.is_alpha:
            continue
        w = tok.text.lower()
        pos = tok.pos_

        if w in PPRON:
            counts["ppron"] += 1
        if pos == "VERB":
            counts["verb"] += 1
        if pos == "AUX":
            counts["auxverb"] += 1
        if w in NEG_EMO:
            counts["negemo"] += 1
        if w in ANX:
            counts["anx"] += 1
        if w in NEGATE:
            counts["negate"] += 1
        if w in CERTAIN:
            counts["certain"] += 1
        if w in COGPROC:
            counts["cogproc"] += 1
        if w in ARTICLES:
            counts["article"] += 1
        if pos == "ADP":
            counts["preps"] += 1
        if w in POS_EMO:
            counts["posemo"] += 1
        if len(w) >= 6:
            counts["sixltr"] += 1

    for k in counts:
        counts[k] = counts[k] / n_tokens
    return counts

feat_df = df['speaking'].apply(speech_features).apply(pd.Series)
df = pd.concat([df, feat_df], axis=1)

for col in ["ppron","verb","auxverb","negemo","anx","negate","certain","cogproc",
            "article","preps","posemo","sixltr"]:
    df[col + "_z"] = zscore(df[col].fillna(0.0))

df["feminine_index"] = (
    df["ppron_z"] + df["verb_z"] + df["auxverb_z"] +
    df["negemo_z"] + df["anx_z"] + df["negate_z"] +
    df["certain_z"] + df["cogproc_z"]
)

df["masculine_index"] = (
    df["article_z"] + df["preps_z"] +
    df["posemo_z"] + df["sixltr_z"]
)

df.drop(columns=["speaking"]).to_csv("Scores.csv", index=False, encoding="utf-8-sig")

