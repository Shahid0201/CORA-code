import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CORA_EMI_FILE = Path("emi_scores_by_speech.csv")
OUTPUT_FIG = Path("CORA_emi_party_over_time.png")

def map_year_to_session_start(year: int) -> int:
    """
    Map a calendar year to congressional session starting year.
    Sessions are two years long, starting in odd years, for example 1879, 1881, 1883.
    1879 -> 1879
    1880 -> 1879
    1881 -> 1881
    """
    if pd.isna(year):
        return None
    year = int(year)
    if year % 2 == 1:
        return year
    return year - 1

def map_party_code_to_label(code: str) -> str:
    """
    Map party codes to full labels.
    """
    if code == "D":
        return "Democrat"
    if code == "R":
        return "Republican"
    return None

print("Loading CORA EMI data from:", CORA_EMI_FILE)
cora_df = pd.read_csv(CORA_EMI_FILE)

required_cols = {"id", "year", "speaker_party", "emi"}
missing = required_cols - set(cora_df.columns)
if missing:
    raise ValueError(f"CORA EMI file is missing required columns: {missing}")

cora_df["starting_year"] = cora_df["year"].apply(map_year_to_session_start)

cora_df["party"] = cora_df["speaker_party"].apply(map_party_code_to_label)

cora_df = cora_df.dropna(subset=["starting_year", "party"])

cora_party_session = (
    cora_df
    .groupby(["starting_year", "party"], as_index=False)["emi"]
    .mean()
    .rename(columns={"emi": "cora_emi"})
)

print("CORA aggregated EMI by party and session, head:")
print(cora_party_session.head())

plt.figure(figsize=(10, 5))

parties = ["Democrat", "Republican"]
colors = {
    "Democrat": "tab:blue",
    "Republican": "tab:red",
}

for party in parties:
    sub = cora_party_session[cora_party_session["party"] == party]

    if sub.empty:
        continue

    sub = sub.sort_values("starting_year")

    plt.plot(
        sub["starting_year"],
        sub["cora_emi"],
        label=party,
        color=colors.get(party, None),
        linewidth=2,
    )

plt.axhline(0.0, color="grey", linewidth=1, linestyle=":")

plt.xlabel("Starting year of congressional session")
plt.ylabel("Evidence minus intuition")
plt.title("CORA - EMI by party over time")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FIG, dpi=300)
print("Saved CORA EMI figure to:", OUTPUT_FIG)

plt.show()

MY_EMI_FILE = Path("emi_scores_by_speech.csv")
SEGUN_FILE = Path("segun_data/congress_EMI_party_chamber_w2v_bootstrap_CIs.csv")

OUTPUT_FIG = Path("emi_party_comparison.png")

def map_year_to_session_start(year: int) -> int:
    """
    Map a calendar year to congressional session starting year.
    Sessions are two years long, starting in odd years (1879, 1881, ...).
    So:
        1879 -> 1879
        1880 -> 1879
        1881 -> 1881
        etc.
    """
    if pd.isna(year):
        return None
    year = int(year)
    if year % 2 == 1:
        return year
    return year - 1

def map_party_code_to_label(code: str) -> str:
    """
    Map 'D'/'R' codes to 'Democrat'/'Republican' to match Segun's file.
    """
    if code == "D":
        return "Democrat"
    if code == "R":
        return "Republican"
    return None

print("Loading CORA EMI data from:", MY_EMI_FILE)
my_df = pd.read_csv(MY_EMI_FILE)

required_cols = {"id", "year", "speaker_party", "emi"}
missing = required_cols - set(my_df.columns)
if missing:
    raise ValueError(f"CORA EMI file is missing required columns: {missing}")

my_df["starting_year"] = my_df["year"].apply(map_year_to_session_start)

my_df["party"] = my_df["speaker_party"].apply(map_party_code_to_label)

my_df = my_df.dropna(subset=["starting_year", "party"])

my_party_session = (
    my_df
    .groupby(["starting_year", "party"], as_index=False)["emi"]
    .mean()
    .rename(columns={"emi": "my_emi"})
)

print("My aggregated EMI by party and session (head):")
print(my_party_session.head())

print("Loading Segun EMI data from:", SEGUN_FILE)
segun_df = pd.read_csv(SEGUN_FILE)

segun_party_session = (
    segun_df
    .groupby(["starting_year", "party"], as_index=False)
    .agg(
        segun_emi=("evidence_minus_intuition_score", "mean"),
        segun_lower=("lower_bound", "mean"),
        segun_upper=("upper_bound", "mean"),
    )
)

print("Segun aggregated EMI by party and session (head):")
print(segun_party_session.head())

merged = pd.merge(
    my_party_session,
    segun_party_session,
    on=["starting_year", "party"],
    how="inner",
)

print("Merged data (head):")
print(merged.head())

merged = merged.sort_values(["party", "starting_year"])

plt.figure(figsize=(10, 5))

parties = ["Democrat", "Republican"]
colors = {
    "Democrat": "tab:blue",
    "Republican": "tab:red",
}

for party in parties:
    sub = merged[merged["party"] == party]

    if sub.empty:
        continue

    x = sub["starting_year"]

    plt.plot(
        x,
        sub["my_emi"],
        label=f"CORA EMI – {party}",
        color=colors.get(party, None),
        linewidth=2,
    )

    plt.plot(
        x,
        sub["segun_emi"],
        label=f"Segun EMI – {party}",
        color=colors.get(party, None),
        linewidth=1.5,
        linestyle="--",
    )

    plt.fill_between(
        x,
        sub["segun_lower"],
        sub["segun_upper"],
        color=colors.get(party, None),
        alpha=0.1,
    )

plt.axhline(0.0, color="grey", linewidth=1, linestyle=":")

plt.xlabel("Starting year of congressional session")
plt.ylabel("EMI (evidence minus intuition)")
plt.title("EMI by party over time: CORA dataset vs Segun et al.")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FIG, dpi=300)
print("Saved comparison figure to:", OUTPUT_FIG)

plt.show()

