"""
UK Price Paid — association rule mining (Apriori) on discretised 2018+ features.

Reads derived raw feature rows under data/parallel/; never touches repo-root CSVs.
Writes tables under data/processed/ and a quick preview under figures/.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAR = ROOT / "data" / "parallel"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

RAW_FEAT = PAR / "uk_features_raw_2018plus.txt"
# Columns: log_price, year, month, type_code, is_new, is_freehold
TYPE_LABEL = {0: "D", 1: "S", 2: "T", 3: "F", 4: "O"}
TYPE_FULL = {
    "D": "Detached",
    "S": "Semi-detached",
    "T": "Terraced",
    "F": "Flat / maisonette",
    "O": "Other",
}

SAMPLE_N = 500_000
RNG_SEED = 42
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.45
MIN_LIFT = 1.15
MAX_ITEMSET_K = 3


def load_sample(n: int = SAMPLE_N) -> pd.DataFrame:
    """Reservoir sample of raw feature rows (single pass)."""
    if not RAW_FEAT.exists():
        raise FileNotFoundError(f"Missing {RAW_FEAT}; run scripts/build_uk_panel.py first.")
    cols = ["log_price", "year", "month", "type_code", "is_new", "is_freehold"]
    rng = np.random.default_rng(RNG_SEED)
    buf = np.empty((n, 6), dtype=np.float64)
    filled = 0
    seen = 0
    with RAW_FEAT.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                row = (float(parts[0]), float(parts[1]), float(parts[2]),
                       float(parts[3]), float(parts[4]), float(parts[5]))
            except ValueError:
                continue
            seen += 1
            if filled < n:
                buf[filled] = row
                filled += 1
            else:
                j = int(rng.integers(0, seen))
                if j < n:
                    buf[j] = row
    df = pd.DataFrame(buf[:filled], columns=cols)
    print(f"Sampled {len(df):,} / {seen:,} rows from {RAW_FEAT.name}")
    return df


# BASIC RUN — Task 3. One sale = one basket: property type, tenure (freehold/leasehold), price band.
def to_transactions(df: pd.DataFrame) -> list[frozenset[str]]:
    q33, q66 = df["log_price"].quantile([1 / 3, 2 / 3]).to_numpy()
    tcode = np.rint(df["type_code"].to_numpy()).astype(np.int32)
    is_new = np.rint(df["is_new"].to_numpy()).astype(np.int32)
    is_fh = np.rint(df["is_freehold"].to_numpy()).astype(np.int32)
    year = np.rint(df["year"].to_numpy()).astype(np.int32)
    logp = df["log_price"].to_numpy()

    txs: list[frozenset[str]] = []
    for i in range(len(df)):
        tlab = TYPE_LABEL.get(int(tcode[i]), "O")
        if tlab == "O":
            continue
        items = {
            f"type={tlab}",
            "new" if is_new[i] == 1 else "resale",
            "freehold" if is_fh[i] == 1 else "leasehold",
        }
        if logp[i] <= q33:
            items.add("price=Low")
        elif logp[i] <= q66:
            items.add("price=Mid")
        else:
            items.add("price=High")
        # Omit period bins: year+type→price rules are weak / expected and clutter the report.
        txs.append(frozenset(items))
    return txs


# BASIC RUN — Task 3. Count how often those items occur together (single-machine Python).
def apriori(
    transactions: list[frozenset[str]],
    min_support: float,
    max_k: int = MAX_ITEMSET_K,
) -> dict[frozenset[str], float]:
    n = len(transactions)
    min_count = max(1, int(np.ceil(min_support * n)))

    c1: Counter[str] = Counter()
    for tx in transactions:
        c1.update(tx)
    freq: dict[frozenset[str], float] = {
        frozenset([a]): c / n for a, c in c1.items() if c >= min_count
    }
    prev_sets = list(freq.keys())

    for k in range(2, max_k + 1):
        prev_list = [set(s) for s in prev_sets]
        candidates: set[frozenset[str]] = set()
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                u = prev_list[i] | prev_list[j]
                if len(u) == k:
                    candidates.add(frozenset(u))
        prev_set_set = set(prev_sets)
        pruned: list[frozenset[str]] = []
        for c in candidates:
            if all(frozenset(sub) in prev_set_set for sub in combinations(c, k - 1)):
                pruned.append(c)
        counts: Counter[frozenset[str]] = Counter()
        for tx in transactions:
            for c in pruned:
                if c <= tx:
                    counts[c] += 1
        new_freq = {c: counts[c] / n for c in pruned if counts[c] >= min_count}
        freq.update(new_freq)
        prev_sets = list(new_freq.keys())
        if not prev_sets:
            break
    return freq


# BASIC RUN — Task 3. Turn frequent itemsets into rules with support, confidence, and lift.
def generate_rules(
    freq: dict[frozenset[str], float],
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    rows = []
    for itemset, supp in freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for ant_t in combinations(items, r):
                ant = frozenset(ant_t)
                cons = itemset - ant
                if not cons:
                    continue
                ant_supp = freq.get(ant)
                cons_supp = freq.get(cons)
                if ant_supp is None or cons_supp is None or ant_supp <= 0:
                    continue
                conf = supp / ant_supp
                lift = conf / cons_supp if cons_supp > 0 else np.nan
                if conf < min_confidence or lift < min_lift:
                    continue
                rows.append(
                    {
                        "antecedent": " & ".join(sorted(ant)),
                        "consequent": " & ".join(sorted(cons)),
                        "support": round(supp, 4),
                        "confidence": round(conf, 4),
                        "lift": round(float(lift), 4),
                        "antecedent_support": round(ant_supp, 4),
                        "consequent_support": round(cons_supp, 4),
                        "n_items": len(itemset),
                    }
                )
    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules
    rules = rules.drop_duplicates(subset=["antecedent", "consequent"])
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return rules.reset_index(drop=True)


def pretty_item(item: str) -> str:
    if item.startswith("type="):
        code = item.split("=", 1)[1]
        return TYPE_FULL.get(code, code)
    mapping = {
        "new": "New-build",
        "resale": "Resale",
        "freehold": "Freehold",
        "leasehold": "Leasehold",
        "price=Low": "Low price",
        "price=Mid": "Mid price",
        "price=High": "High price",
    }
    return mapping.get(item, item)


def pretty_side(side: str) -> str:
    return " + ".join(pretty_item(x.strip()) for x in side.split("&"))


def select_display_rules(rules: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Build display rows: Flat+High↔Leasehold once; other reverse pairs collapsed to ↔."""
    # Reuse the paper figure helper if available; otherwise local fall-back.
    try:
        from make_figures4papers import _association_display_rows

        rows = _association_display_rows(rules, top_n=top_n)
    except Exception:
        rows = []
    if not rows:
        return pd.DataFrame(columns=["label", "lift"])
    return pd.DataFrame(rows, columns=["label", "lift"])


def plot_top_rules(rules: pd.DataFrame, top_n: int = 10) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    if rules.empty:
        raise RuntimeError("No rules to plot.")
    top = select_display_rules(rules, top_n=top_n).iloc[::-1].copy()
    if top.empty:
        raise RuntimeError("No display rules.")
    labels = top["label"].tolist()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.barh(range(len(top)), top["lift"], color="#1f4e79", height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Lift", labelpad=8)
    ax.set_title("Association", pad=10)
    xmax = float(top["lift"].max())
    pad = 0.08
    for i, lift in enumerate(top["lift"]):
        ax.text(
            float(lift) + pad,
            i,
            f"{lift:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
            clip_on=False,
        )
    ax.set_xlim(0, xmax + pad + 0.55)
    fig.subplots_adjust(left=0.42, right=0.96, top=0.90, bottom=0.12)
    out = FIG / "rule_mining_top_lift.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    print("Loading sample for rule mining ...")
    df = load_sample(SAMPLE_N)
    print("Building transactions ...")
    txs = to_transactions(df)
    print(f"Transactions (excl. Other): {len(txs):,}")

    print(
        f"Apriori min_support={MIN_SUPPORT}, "
        f"min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT} ..."
    )
    freq = apriori(txs, MIN_SUPPORT, MAX_ITEMSET_K)
    itemsets = (
        pd.DataFrame(
            [
                {
                    "itemset": " & ".join(sorted(s)),
                    "support": round(sup, 4),
                    "n_items": len(s),
                }
                for s, sup in freq.items()
            ]
        )
        .sort_values(["n_items", "support"], ascending=[True, False])
        .reset_index(drop=True)
    )
    itemsets_path = PROC / "association_itemsets.csv"
    itemsets.to_csv(itemsets_path, index=False)
    print(f"Wrote {itemsets_path} ({len(itemsets)} itemsets)")

    rules = generate_rules(freq, MIN_CONFIDENCE, MIN_LIFT)
    rules_path = PROC / "association_rules.csv"
    rules.to_csv(rules_path, index=False)
    print(f"Wrote {rules_path} ({len(rules)} rules)")

    if not rules.empty:
        nice = rules.head(25).copy()
        nice["antecedent_label"] = nice["antecedent"].map(pretty_side)
        nice["consequent_label"] = nice["consequent"].map(pretty_side)
        nice_path = PROC / "association_rules_top25.csv"
        nice.to_csv(nice_path, index=False)
        print(f"Wrote {nice_path}")
        fig_path = plot_top_rules(rules, top_n=12)
        print(f"Wrote {fig_path}")
        print("\nTop 8 rules by lift:")
        for _, r in rules.head(8).iterrows():
            print(
                f"  {pretty_side(r['antecedent'])} → {pretty_side(r['consequent'])}  "
                f"(lift={r['lift']:.2f}, conf={r['confidence']:.2f}, supp={r['support']:.2f})"
            )
    else:
        print("WARNING: no rules passed thresholds; try lowering MIN_* constants.")


if __name__ == "__main__":
    main()
