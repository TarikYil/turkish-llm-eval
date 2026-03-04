# Turkish LLM Benchmark Suite

**Comparative Analysis of Reliability and Reasoning in Turkish Large Language Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18861750.svg)](https://doi.org/10.5281/zenodo.18861750)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Google Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/)
 
---

## Overview

This repository contains the benchmarking code and evaluation pipeline accompanying the paper:

> **Comparative Analysis of Reliability and Reasoning in Turkish Large Language Models**
> *Submitted to SIU 2026 — IEEE 34th Signal Processing and Communications Applications Conference*

The framework evaluates **7 open-source Turkish LLMs** across **6 diverse benchmarks**, covering knowledge, reasoning, mathematical problem-solving, commonsense inference, truthfulness, and hallucination detection capabilities.

---

## Evaluated Models

| Model | HuggingFace ID | Parameters | Family |
|-------|---------------|------------|--------|
| CosmosGemma-9B (Instruct) | `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` | 9B | Cosmos |
| Trendyol-7B-Chat v0.1 | `Trendyol/Trendyol-LLM-7b-chat-v0.1` | 7B | Trendyol |
| Trendyol-7B-Chat v4.1 | `Trendyol/Trendyol-LLM-7B-chat-v4.1.0` | 7B | Trendyol |
| WiroAI-9B (Instruct) | `WiroAI/wiroai-turkish-llm-9b` | 9B | WiroAI |
| WiroAI-8B (Instruct) | `WiroAI/wiroai-turkish-llm-8b` | 8B | WiroAI |
| KoçDigital-8B v0.1 | `KOCDIGITAL/Kocdigital-LLM-8b-v0.1` | 8B | KoçDigital |
| Kumru-2B | `vngrs-ai/Kumru-2B` | 2B | Kumru |

---

## Benchmarks

| Benchmark | Task Type | Source Dataset | Metric(s) |
|-----------|-----------|---------------|------------|
| **MMLU-TR** | Multiple-choice knowledge | `malhajar/mmlu-tr` (12 subject configs) | Accuracy, Macro F1 |
| **ARC-TR** | Science reasoning (multiple-choice) | `malhajar/arc-tr` | Accuracy, Macro F1 |
| **HellaSwag-TR** | Commonsense / sentence completion | `malhajar/hellaswag-tr` | Accuracy, Macro F1 |
| **GSM8K-TR** | Mathematical reasoning (open-ended) | `malhajar/gsm8k_tr-v0.2` | Exact Match, Loose Match |
| **TruthfulQA-TR** | Truthfulness evaluation | `malhajar/truthfulqa-tr` | ROUGE-L, Keyword Match |
| **HalluVerse-M3** | Hallucination detection | `sabdalja/HalluVerse-M3` (Turkish QA subset) | Miss Rate, Binary Acc, F1, Factual Accuracy |

For MMLU-TR, the following 12 subject categories are used: formal logic, logical fallacies, philosophy, moral scenarios, conceptual physics, machine learning, global facts, world religions, international law, security studies, professional medicine, and virology.

---

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── turkish_llm_benchmark.ipynb        # Main evaluation notebook
├── requirements.txt                   # Python dependencies
└── siu2026_results/                   # Output directory (generated at runtime)
    ├── full_results_<timestamp>.json  # Detailed per-question results
    └── summary_<timestamp>.csv        # Aggregated performance table
```

---

## Requirements

- Python 3.8+
- CUDA-compatible GPU with at least 16 GB VRAM (24 GB recommended)
- ~50 GB disk space for model weights (downloaded on first run)

### Dependencies

```
torch>=2.0
transformers>=4.38
accelerate>=0.25
bitsandbytes>=0.46.1
datasets
rouge-score
sentencepiece
openpyxl
```

---

## Quick Start

### Option 1: Google Colab (Recommended)

1. Upload `turkish_llm_benchmark.ipynb` to Google Colab.
2. Select a GPU runtime: **Runtime → Change runtime type → T4 or A100**.
3. Run all cells sequentially. The first two cells install required packages.
4. Results will be saved to the `siu2026_results/` directory.

### Option 2: Local Execution

```bash
# Clone the repository
git clone https://github.com/TarikYil/turkish-llm-eval.git
cd turkish-llm-eval

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook turkish_llm_benchmark.ipynb
```

---

## Configuration

Key parameters can be adjusted in the `CONFIG` dictionary at the top of the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantization` | `"4bit"` | Quantization mode: `"4bit"`, `"8bit"`, or `"none"` |
| `max_samples_per_benchmark` | `100` | Maximum number of questions per benchmark |
| `max_new_tokens` | `256` | Maximum generation length per response |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `seed` | `42` | Random seed for reproducibility |
| `output_dir` | `"siu2026_results"` | Directory for saving results |

---

## Pipeline Architecture

The evaluation pipeline follows a modular design with the following stages:

1. **Configuration & Model Registry** — Defines hyperparameters and model metadata.
2. **Dataset Loaders** — Loads Turkish benchmark datasets from HuggingFace Hub with built-in fallback questions if remote loading fails.
3. **Model Loader** — Loads models with configurable quantization (4-bit NF4, 8-bit, or FP16) using BitsAndBytes.
4. **Inference Engine** — Generates responses with automatic chat template detection and application (critical for Qwen2.5-based and Gemma-2 architectures).
5. **Benchmark Runners** — Task-specific evaluation functions with per-class F1 tracking, subject-level breakdowns, and detailed logging.
6. **Answer Extraction** — Robust post-processing with multi-pattern matching for choice extraction and numerical answer parsing.
7. **Results Export** — Saves full per-question results (JSON) and aggregated summary tables (CSV).

---

## Output Format

### JSON (Full Results)

Each model's entry contains per-benchmark detailed results including every question, the model's raw response, extracted prediction, expected answer, and correctness flag. This enables fine-grained error analysis.

### CSV (Summary)

A single row per model with columns for each metric across all benchmarks, including an overall average accuracy score.

---

## Methodology Notes

- **Quantization**: All models are evaluated under 4-bit NF4 quantization by default to ensure fair comparison under constrained GPU memory. This allows running 9B-parameter models on a single T4 GPU.
- **Chat Templates**: The inference engine automatically detects and applies model-specific chat templates (e.g., for Qwen2.5-based models), falling back to raw prompt injection when no template is available.
- **Sampling**: 100 questions per benchmark are randomly sampled (with a fixed seed) from each dataset to balance evaluation thoroughness with computational cost.
- **Hallucination Evaluation**: Uses the HalluVerse-M3 Turkish QA subset, generating paired faithful/hallucinated examples from each data point. Detection relies on linguistic pattern matching for Turkish hedging and confidence markers.

---

## Reproducibility

To reproduce the results reported in the paper:

1. Use the default `CONFIG` values (seed=42, quantization=4bit, max_samples=100).
2. Run on a CUDA-enabled GPU (results were generated on NVIDIA T4 / A100).
3. Ensure `transformers>=4.38` for consistent chat template behavior.
4. Dataset versions are pinned to the HuggingFace Hub identifiers listed in the Benchmarks section.

---

## Citation

If you use this code or the benchmark results in your research, please cite:

```bibtex
@inproceedings{turkish_llm_eval_siu2026,
  title     = {Comparative Evaluation of Open-Source Turkish Large Language Models Across Multiple Benchmarks},
  author    = {Yilmaz, Muhammet Tarik and Cakir, Yusuf},
  booktitle = {IEEE 34th Signal Processing and Communications Applications Conference (SIU)},
  year      = {2026},
  note      = {Code available at https://github.com/TarikYil/turkish-llm-eval}
}

@software{yilmaz_turkish_llm_eval_2026,
  author    = {Yilmaz, Muhammet Tarik},
  title     = {Turkish LLM Benchmark Suite},
  year      = {2026},
  version   = {v1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18861750},
  url       = {https://doi.org/10.5281/zenodo.18861750}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The benchmark datasets are subject to their respective licenses on HuggingFace Hub. The evaluated models are subject to their own model licenses as specified by their creators.

---

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for the Transformers library and dataset hosting.
- Dataset creators: [malhajar](https://huggingface.co/malhajar) for MMLU-TR, ARC-TR, HellaSwag-TR, GSM8K-TR, and TruthfulQA-TR translations; [sabdalja](https://huggingface.co/sabdalja) for HalluVerse-M3.
- Model creators: YTU CE Cosmos, Trendyol, WiroAI, KoçDigital, and VNGRS AI teams.
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for efficient quantization support.
