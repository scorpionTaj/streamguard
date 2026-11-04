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
# Run as a notebook: `uv run jupyter lab` (percent format via jupytext).

# %% [markdown]
# ## 1. Load & basic profile

# %%
# TODO(M1): polars read_csv, shape, isFraud mean, TransactionDT span in days

# %% [markdown]
# ## 2. Nulls & identity join coverage

# %%
# TODO(M1): null share per column group (V, C, M, dist, device); identity join rate (144k/590k)

# %% [markdown]
# ## 3. V-feature selection (train split only!)

# %%
# TODO(M1): temporal split first → mutual information on train → top-30 → freeze into configs/features.yaml

# %% [markdown]
# ## 4. card_key cardinality & velocity preview

# %%
# TODO(M1): group by card_key (schemas.derive_card_key) → tx/card distribution; preview 24h velocity for a busy card
