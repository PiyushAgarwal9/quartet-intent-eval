# Stable and Sensitive? Quartet benchmark for Hindi and Tamil banking intent classification

Artifacts for the paper:

> Piyush Agarwal. *Stable and Sensitive? A Controlled Quartet Evaluation of Hindi and Tamil Banking Intent Classification.* 22nd International Conference on Natural Language Processing (ICON 2026).

All utterances are synthetic. There is no customer data, and no real institution is named in any utterance (two intent definitions mention UPI apps by name as channel examples).

## Quick check (standard library only, no API key, no network)

    python scripts/verify.py

It checks the dataset hash against `FREEZE.txt`, the quartet structure and the 6,240 result records, recomputes the main statistics from the per-item predictions, and compares them with the stored analysis.

## Files

| Path | What it is |
|---|---|
| `data/benchmark_480.csv` | The frozen benchmark: 240 Hindi and 240 Tamil items in 120 quartets. Tamil-script items are the same writer's renditions of the romanized items (meaning-matched pairs). |
| `data/taxonomy.yaml` | The 163 intent labels with domain, one-line definition and the three English examples used by the embedding baseline. |
| `data/pilot_dev_80.json` | An AI-drafted development pilot (development only; it shares 2 items with the benchmark, as the paper states). |
| `results/results_final_v2.jsonl` | 6,240 per-item predictions: 5 systems, LLM arms run three times; final parsed predictions. |
| `results/analysis_v4.json` | The stored analysis the paper's tables were generated from. |
| `results/per_boundary_pooled.json` | Per-boundary accuracies (Appendix D of the paper). |
| `validation/` | The two Hindi validators' independent sheets, the Tamil independent-review sheet, the revision records and the Tamil reconfirmation sheet (all anonymized). |
| `scripts/verify.py` | One-command integrity and recomputation check. |
| `scripts/analysis.py` | The full analysis, including the cluster bootstrap (15 boundary clusters; sensitivity with 30 and 120 clusters). |
| `scripts/generate_tables.py` | Regenerates the submitted table values from `results/analysis_v4.json`. The seven camera-ready rounding corrections are listed below. |
| `scripts/eval_rerun.py` | Optional: re-runs the models (needs `GROQ_API_KEY`; the two LLMs are accessed through a hosted API and may change or be withdrawn). |
| `FREEZE.txt` | Dataset hash chronology, recorded before the final model runs. It names the benchmark by its working file name, `final_items_hi_ta_v2.csv`; that file is `data/benchmark_480.csv` here (same hash). |
| `MANIFEST.sha256` | SHA-256 of every file in this release. |

## What differs from the bundle described at submission time

- `data/taxonomy.yaml` is reduced to the label set the evaluation uses (names, definitions, three examples per intent). The working taxonomy also holds slot patterns, thresholds and workflow settings that the paper does not use; those are not released. The classification prompt needs only the 163 names, and the embedding baseline needs only the definitions and these three examples, so `scripts/eval_rerun.py` runs unchanged.
- `README.md` and the licence files were rewritten for publication, and `MANIFEST.sha256` was regenerated.
- The benchmark, the pilot, the per-item predictions, the stored analysis, the scripts and the validation files are byte-identical to the submission-time bundle.

## Rounding note

`results/analysis_v4.json` stores values rounded to four decimals, and `scripts/generate_tables.py` rounds those again to three. Seven cells of the submitted paper therefore differed by 0.001 from the correctly rounded exact value. The camera-ready paper prints the exact values; `generate_tables.py` still reproduces the submitted rounding.

| Cell | generate_tables.py | Paper (exact) |
|---|---|---|
| Main table, paraphrase O-E, qwen3.6-27b | +0.004 | +0.005 (exact +0.004514) |
| Main table, contrast O-E, gpt-oss translate arm | +0.021 | +0.020 (exact +0.020486) |
| Interval table, gpt-oss-20b (30 clusters), contrast lower bound | +0.004 | +0.005 |
| Interval table, gpt-oss translate arm (30), contrast upper bound | +0.037 | +0.038 |
| Interval table, gpt-oss-20b (120), contrast upper bound | +0.029 | +0.028 |
| Interval table, gpt-oss translate arm (120), paraphrase upper bound | +0.004 | +0.003 |
| Interval table, qwen translate arm (120), contrast upper bound | +0.004 | +0.003 |

## Note on the Hindi revision record

`validation/revision_record.csv` is kept as it was at submission; it repeats the same `flagged_by` text on every row. The initial sheets give the attribution per quartet: both validators flagged B03-native-q1 and B04-roman-q1; only Validator 1 flagged B15-roman-q1; Validator 2 alone flagged the other 13.

## Note on file dates

The validation spreadsheets were re-saved to remove personal document properties, so their internal timestamps are later than the dates on which the validators filled them.

## Licences

Data (`data/`, `validation/`, `results/`): CC BY 4.0, see `LICENSE-DATA`. Code (`scripts/`): Apache License 2.0, see `LICENSE-CODE`.

## Citation

```bibtex
@inproceedings{agarwal2026stable,
  title     = {Stable and Sensitive? {A} Controlled Quartet Evaluation of {Hindi} and {Tamil} Banking Intent Classification},
  author    = {Agarwal, Piyush},
  booktitle = {Proceedings of the 22nd International Conference on Natural Language Processing (ICON)},
  year      = {2026}
}
```

## Contact

Please open an issue at https://github.com/PiyushAgarwal9/quartet-intent-eval.
