# Turkish LLM Multi-Axis Evaluation Suite — v2.0.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18861749.svg)](https://doi.org/10.5281/zenodo.18861749)
[![Paper DOI](https://img.shields.io/badge/IEEE-10.1109%2FASYU71790.2026.11772376-blue)](https://doi.org/10.1109/ASYU71790.2026.11772376)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This release contains the code and per-item results for:

> M. T. Yılmaz, M. A. Pala, Y. Çakır, "A Multi-Axis Comparative Evaluation of Turkish Large Language Models across Knowledge, Reasoning, and Reliability," *2026 Innovations in Intelligent Systems and Applications Conference (ASYU)*, IEEE, 2026. doi:[10.1109/ASYU71790.2026.11772376](https://doi.org/10.1109/ASYU71790.2026.11772376)

Seven open-weight Turkish LLMs are evaluated on six benchmarks. All models use a shared scoring protocol and fixed inference settings. Pairwise differences between models are tested with McNemar's test, and scores are reported with Wilson confidence intervals.

**Changes from v1.0.0.** v1.0.0 was the pre-review pipeline. v2.0.0 is the version used for the published paper:

- Multiple-choice tasks are scored by log-likelihood. There is no parsing of generated text.
- TruthfulQA uses the MC1/MC2 protocol.
- Generation uses greedy decoding with seed 42.
- Every model is evaluated on the same fixed subsets.
- 4-bit NF4 is the only quantization setting used.
- Statistical testing has been added.

The v1.0.0 code is still available under the git tag `v1.0.0`.

## Contents

| Path | What it is |
|---|---|
| `notebook/turkish_llm_eval_asyu2026.ipynb` | The evaluation notebook as run on Google Colab. Code is unchanged; outputs are removed. The first cell gives the run order. |
| `results/result_<model>.json` | Per model and benchmark: the summary metrics, plus `item_correct`, a 0/1 correctness value for every scored item. Also `mc2_per_item` for TruthfulQA, `pred`/`gold` class indices where they were recorded, and the confusion counts for HalluVerse-M3. |
| `data/subset_manifest.csv` | The exact items evaluated, given as hashes. It contains no benchmark text. |
| `scripts/recompute_stats.py` | Recomputes the Wilson CIs, the MC2 bootstrap CIs and the pairwise McNemar tests from `results/`. Needs numpy and scipy only. |
| `scripts/check_subset.py` | Checks that a `benchmarks.json` you rebuilt matches the subset used in the paper. |
| `requirements.txt` | The package versions from the original run. |
| `SHA256SUMS.txt` | Checksums for every file in the release. |

## Models

| Key | Hugging Face ID | Prompting |
|---|---|---|
| `cosmos_gemma_9b` | ytu-ce-cosmos/Turkish-Gemma-9b-v0.1 | chat template, zero-shot |
| `trendyol_7b_chat` | Trendyol/Trendyol-LLM-7b-chat-v0.1 | chat template, zero-shot |
| `trendyol_7b_chat_v4` | Trendyol/Trendyol-LLM-7B-chat-v4.1.0 | chat template, zero-shot |
| `wiroai_9b` | WiroAI/wiroai-turkish-llm-9b | chat template, zero-shot |
| `wiroai_8b` | WiroAI/wiroai-turkish-llm-8b | chat template, zero-shot |
| `kocdigital_8b` | KOCDIGITAL/Kocdigital-LLM-8b-v0.1 | chat template, zero-shot |
| `kumru_2b_base` | vngrs-ai/Kumru-2B-Base | raw prompt; 3-shot on GSM8K-TR only |

All models were loaded in 4-bit NF4 with bitsandbytes (double quantization, fp16 compute).

## Benchmarks and protocol

| Axis | Benchmark (source) | Items scored | Scoring |
|---|---|---|---|
| Knowledge | MMLU-TR, 12 subjects (`malhajar/mmlu-tr`) | 2905 | log-likelihood of the option letter; accuracy and macro-F1 |
| Scientific reasoning | ARC-TR, Challenge (`malhajar/arc-tr`) | 1167 of 1169 | as for MMLU-TR |
| Commonsense | HellaSwag-TR (`malhajar/hellaswag-tr`) | 1000 | log-likelihood of the full ending, normalized by token length |
| Math reasoning | GSM8K-TR (`malhajar/gsm8k_tr-v0.2`) | 500 | greedy generation, max 300 new tokens; exact match on the final number |
| Reliability | TruthfulQA-TR (`malhajar/truthfull_qa-tr`) | 500 | length-normalized log-likelihood; MC1 and MC2 |
| Hallucination | HalluVerse-M3, Turkish QA (`sabdalja/HalluVerse-M3`) | 800 (402 hallucinated / 398 faithful) | greedy generation, max 8 new tokens; DOĞRU/YANLIŞ label; accuracy and F1 |

**How the subsets were drawn.** Each benchmark was shuffled with `random.seed(42)` and the first N items were kept. All models see the same items in the same order. This pairing is what makes McNemar's test valid.

**ARC item count.** Two ARC items have the answer key `"4"`. The scorer skips them, so 1167 of the 1169 items are scored. See *Known issue* below.

**Kumru-2B-Base few-shot examples.** The 3-shot context for Kumru on GSM8K-TR uses three hand-written examples. These are `_GSM8K_FEWSHOT_DEFAULT` in the notebook. The dataset has no train split, so the code falls back to them. They are not part of the test set. The Kumru GSM8K score is a base-model reference only and is not comparable to the zero-shot scores.

## Known issue: ARC-TR answer keys given as digits

Most ARC-TR answer keys are letters (`A`–`D`). In the 1169-item subset, 22 keys are digits (`"1"`–`"4"`). These digits follow the original ARC convention and are **1-based**. The notebook's `_normalize_gold` reads them as **0-based**. As a result:

- 20 scored items were compared against the option *after* the correct one.
- The 2 items with key `"4"` were treated as out of range and skipped.

The manifest marks all 22 items with `flag = arc_digit_key`. The files in `results/` keep the scores exactly as computed for the paper. For models whose predictions were stored, the corrected ARC-TR scores can be recomputed:

```bash
python scripts/recompute_stats.py --arc-corrected
```

This re-scores the 20 items as `pred == gold - 1`. The 2 skipped items cannot be recovered without re-running the models. Every other benchmark is unaffected: MMLU and HellaSwag keys are 0-based, and TruthfulQA, GSM8K and HalluVerse do not use option indices.

**Impact on ARC-TR accuracy (n = 1167).** Four models have stored predictions, so their corrected scores are exact. Three do not. For those three, the range is bounded as follows:

- Lower bound: every flagged item is wrong. An item counted correct before is now certainly wrong, and an item counted wrong before is assumed to stay wrong.
- Upper bound: every flagged item counted wrong before is now correct.

| Model | Published | Corrected |
|---|---|---|
| CosmosGemma-9B | 78.1 | 78.0 – 79.6 (no stored predictions) |
| WiroAI-9B | 73.9 | 75.3 |
| Trendyol-7B v4.1 | 68.9 | 70.1 |
| WiroAI-8B | 63.5 | 64.7 |
| KoçDigital-8B | 54.2 | 55.4 |
| Trendyol-7B v0.1 | 23.5 | 23.4 – 25.0 (no stored predictions) |
| Kumru-2B-Base | 23.4 | 22.9 – 24.1 (no stored predictions) |

The corrections change scores by at most about 1.5 percentage points. The number of significantly different ARC-TR pairs stays at 20/21 under both bounds. The only pair that is not significant is still Trendyol-7B v0.1 vs Kumru-2B-Base. The ranking of the top five models does not change. The paper's qualitative conclusions are therefore unaffected.

## Reproducing the statistics (no GPU)

```bash
pip install numpy scipy
python scripts/recompute_stats.py --csv stats_out/
```

The output gives, for each benchmark, every model's score with its 95% Wilson CI and the number of model pairs that differ significantly by McNemar (p < 0.05). The paper reports these pair counts as 20/21 for knowledge, 20/21 for scientific reasoning, 18/21 for commonsense, 17/21 for math reasoning, 9/21 for reliability and 15/21 for hallucination detection.

## Re-running the evaluation (GPU)

Open the notebook in Google Colab with a GPU runtime and follow the run order in its first cell. Two things differ from the original run:

1. **Rebuilding the data.** The notebook downloads the benchmarks from Hugging Face at run time. Datasets can change upstream. After building `benchmarks.json`, run `python scripts/check_subset.py benchmarks.json` to confirm you have the same items.
2. **Run-to-run differences.** The `torch` version and the GPU type were not recorded. Even with greedy decoding, 4-bit GPU inference is not bit-exact across runs. An earlier WiroAI-9B run with the same code and data differed from the released run in two ways:
   - 4 of 500 GSM8K-TR items (40.2% vs 40.6%);
   - small differences in TruthfulQA MC2.

   The files in `results/` are the runs used in the paper.

## Data and licensing

- **Code** (the notebook and scripts) is released under the MIT License.
- **Benchmark text is not redistributed.** The data belongs to the respective dataset authors and is subject to their licenses. Only hashes and per-item 0/1 outcomes are included here.
- **Results** in `results/` are released under the same MIT terms as the code.

## Citation

Please cite the paper above. Machine-readable metadata is in `CITATION.cff`.
