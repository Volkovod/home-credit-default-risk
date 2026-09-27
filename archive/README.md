# Executed source snapshot

These files are unchanged copies of successfully executed Kaggle sources.
They preserve historical input dependencies. Use `src/` or notebooks 01-03
as the public entry points for new runs.

| File | Kaggle kernel / version |
|---|---|
| `executed/build_features.py` | `mrfirstik/home-credit-feature-cache-20260925` / 1 |
| `executed/augment_features.py` | `mrfirstik/home-credit-research-features-20260926` / 1 |
| `executed/train_research.py` | `mrfirstik/home-credit-research-gpu-20260926` / 1 |

The archived training script reuses previous benchmark predictions after
checking row IDs, labels and AUC. These private outputs are not bundled.
The public script can regenerate missing references.
Source SHA-256 digests: [source_lineage.json](../reports/metrics/source_lineage.json).
