# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # StreamGuard — EDA & feature freezing (milestone M1)
#
# Goals (PLAN.md §5, §6):
# 1. Load `data/train_transaction.csv` (+ identity join) with polars.
# 2. Confirm temporal span, fraud rate (~3.5%), null profile of V/dist/device.
# 3. Pick the **top-30 V-features by mutual information, computed on the train
#    split only** (never val/test) → write them into `configs/features.yaml`.
# 4. Sanity-check card_key cardinality (velocity grouping) and the temporal
#    split boundaries.
#
# Run as a notebook (`uv run jupyter lab`, percent format via jupytext) or as a
# script: `uv run python notebooks/01_eda.py [--freeze]` (`--freeze` rewrites v_subset).

# %%
import re
import sys

import numpy as np
import polars as pl

from streamguard.config import REPO_ROOT
from streamguard.offline.features import select_v_subset, temporal_split, velocity_features
from streamguard.schemas import derive_card_key

DATA = REPO_ROOT / "data"
FEATURES_YAML = REPO_ROOT / "configs" / "features.yaml"
V_COLS = [f"V{i}" for i in range(1, 340)]

# %% [markdown]
# ## 1. Load & basic profile

# %%
tx = pl.read_csv(
    DATA / "train_transaction.csv",
    schema_overrides={
        c: pl.Float64 for c in ["card2", "card5", "addr1", "dist1", "dist2", *V_COLS]
    },
).sort("TransactionDT", "TransactionID")
identity = pl.read_csv(DATA / "train_identity.csv", columns=["TransactionID", "DeviceType"])
df = tx.join(identity, on="TransactionID", how="left")
span_days = (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400
print(
    f"rows={df.height:,}  cols={df.width}  fraud rate={df['isFraud'].mean():.4f}  span={span_days:.1f} days"
)

# %% [markdown]
# ## 2. Nulls & identity join coverage

# %%
groups = {
    "V": V_COLS,
    "C": [f"C{i}" for i in range(1, 15)],
    "M": [f"M{i}" for i in range(1, 10)],
    "dist": ["dist1", "dist2"],
    "device": ["DeviceType"],
}
for name, cols in groups.items():
    share = df.select(pl.col(cols).null_count()).sum_horizontal().item() / (df.height * len(cols))
    print(f"null share {name:>6}: {share:.1%}")
print(
    f"identity join rate: {df['DeviceType'].is_not_null().mean():.1%} (144,233 / 590,540 expected ≈ 24%)"
)

# %% [markdown]
# ## 3. V-feature selection (train split only!)
#
# Split by event time FIRST, then rank V columns by discrete mutual information
# (20 quantile bins + a NaN bin) on the train split. Val/test never influence the choice.

# %%
splits = temporal_split(df, "TransactionDT")
train = splits["train"]
for name, part in splits.items():
    print(f"{name:>5}: {part.height:>7,} rows  fraud={part['isFraud'].mean():.4f}  "
          f"DT=[{part['TransactionDT'].min()}, {part['TransactionDT'].max()}]")  # fmt: skip
v_subset = select_v_subset(train.select(V_COLS), train["isFraud"].to_numpy(), k=30)
print("top-30 V by MI (train):", v_subset)

# %%
if "--freeze" in sys.argv:
    text = FEATURES_YAML.read_text(encoding="utf-8")
    frozen = f"v_subset: [{', '.join(v_subset)}]   # frozen at M1: top-30 by MI, train split only"
    text, n = re.subn(r"^v_subset:.*$", frozen, text, count=1, flags=re.M)
    assert n == 1, "v_subset line not found in features.yaml"
    FEATURES_YAML.write_text(text, encoding="utf-8")
    print(f"froze v_subset into {FEATURES_YAML.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## 4. card_key cardinality & velocity preview

# %%
keys = df.select(
    pl.struct("card1", "card2", "card5", "addr1")
    .map_elements(
        lambda r: derive_card_key(r["card1"], r["card2"], r["card5"], r["addr1"]),
        return_dtype=pl.Utf8,
    )  # fmt: skip
    .alias("card_key"),
    "TransactionDT",
    "TransactionAmt",
)
per_card = keys.group_by("card_key").len()["len"].to_numpy()
print(
    f"cards={per_card.size:,}  tx/card p50={np.median(per_card):.0f} p99={np.quantile(per_card, 0.99):.0f} max={per_card.max():,}"
)
busiest = keys.group_by("card_key").len().sort("len", descending=True)["card_key"][0]
busy = keys.filter(pl.col("card_key") == busiest)
velocity = velocity_features(
    busy["card_key"].to_numpy(), busy["TransactionDT"].to_numpy().astype(float),
    busy["TransactionAmt"].to_numpy(), {"24h": 86400}, ["txn_count", "amt_sum"], -1.0,
)  # fmt: skip
print(busy.with_columns(**velocity).tail(5))
