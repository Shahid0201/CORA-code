import pandas as pd
import glob, json
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

df = pd.read_csv("Scores.csv", encoding="utf-8-sig")

df["feminine_index_corrected"] = (
    df["ppron_z"] +
    df["verb_z"] +
    df["auxverb_z"] +
    df["cogproc_z"]
    - df["negemo_z"]
    - df["anx_z"]
    - df["negate_z"]
    - df["certain_z"]
)

df["masculine_index_corrected"] = (
    df["article_z"] +
    df["preps_z"] +
    df["posemo_z"] +
    df["sixltr_z"]
)

df.to_csv("Scores_CORRECTED.csv", index=False, encoding="utf-8-sig")

print("Saved Scores_CORRECTED.csv with corrected composite indices.")

files = glob.glob(r"data\speeches_*.jsonl")

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
    (df['year'].between(1873, 2025)) &
    (df['speaker_gender'].isin(['M', 'F']))
].copy()

df = df[df['speaking'].str.split().str.len() >= 50]

df['speaker_gender'].value_counts(normalize=True)

male_speeches = df[df['speaker_gender'] == 'M']
female_speeches = df[df['speaker_gender'] == 'F']

len(male_speeches), len(female_speeches)

scores = pd.read_csv("Scores_CORRECTED.csv", encoding="utf-8-sig")

df['id'] = df['id'].astype(str)
scores['id'] = scores['id'].astype(str)

df = df.merge(scores[['id', 'feminine_index', 'masculine_index']], on='id', how='left')

df["feminine_index"]  = -df["feminine_index"]
df["masculine_index"] = -df["masculine_index"]

summary = df.groupby('speaker_gender')[['feminine_index', 'masculine_index']].mean()
print(summary)

male_speeches = df[df['speaker_gender'] == 'M']
female_speeches = df[df['speaker_gender'] == 'F']

def cohens_d(group1, group2):
    x1 = group1.values
    x2 = group2.values
    n1 = x1.size
    n2 = x2.size
    mean1 = x1.mean()
    mean2 = x2.mean()
    var1 = x1.var(ddof=1)
    var2 = x2.var(ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    d = (mean1 - mean2) / np.sqrt(pooled_var)
    return d

male_speeches = df[df['speaker_gender'] == 'M']
female_speeches = df[df['speaker_gender'] == 'F']

d_fem = cohens_d(male_speeches['feminine_index'],
                 female_speeches['feminine_index'])

d_masc = cohens_d(male_speeches['masculine_index'],
                  female_speeches['masculine_index'])

print("Cohen's d (male − female) feminine:", d_fem)
print("Cohen's d (male − female) masculine:", d_masc)

t_fem, p_fem = ttest_ind(
    male_speeches['feminine_index'],
    female_speeches['feminine_index'],
    equal_var=True
)

t_masc, p_masc = ttest_ind(
    male_speeches['masculine_index'],
    female_speeches['masculine_index'],
    equal_var=True
)

print("Feminine t, p:", t_fem, p_fem)
print("Masculine t, p:", t_masc, p_masc)

means = summary
genders = ['M', 'F']
x = np.arange(2)
width = 0.35

color_masc = 'tab:blue'
color_fem  = 'tab:orange'

fig, ax = plt.subplots(figsize=(6, 4))

masc_vals = means.loc[genders, 'masculine_index'].values
fem_vals  = means.loc[genders, 'feminine_index'].values

ax.bar(x - width/2, masc_vals, width, label='Masculine language', color=color_masc)
ax.bar(x + width/2, fem_vals,  width, label='Feminine language',  color=color_fem)

ax.set_xticks(x)
ax.set_xticklabels(['Male senators', 'Female senators'])
ax.set_ylabel('Composite index (z sum)')
ax.set_title('Feminine and masculine language by senator gender (1873-2025, CORA)')
ax.legend(loc='upper left')

ax.set_ylim(-0.15, 0.45)

ax.grid(axis='y', linestyle='--', alpha=0.6)

threshold = 0.02
for i, v in enumerate(masc_vals):
    if abs(v) < threshold:
        ax.text(x[i] - width/2, 0.01, f"{v:.4f}", ha='center', fontsize=9)

for i, v in enumerate(fem_vals):
    if abs(v) < threshold:
        ax.text(x[i] + width/2, 0.01, f"{v:.3f}", ha='center', fontsize=9)

plt.tight_layout()
fig.savefig("topic_valid_lan_gend.png", dpi=300, bbox_inches='tight')
plt.show()

export_df = (
    means
    .loc[genders, ['masculine_index', 'feminine_index']]
    .reset_index()
    .rename(columns={'index': 'speaker_gender'})
)

export_df.to_csv(
    "cora_1873_2025.csv",
    index=False,
    encoding="utf-8"
)

