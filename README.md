# MICL

This repository contains scripts for [Multimodal In-Context Learning for ASR of Low-Resource Languages](https://arxiv.org/abs/2601.05707).


---

## `speechllm_icl_strategy.py`

Main entry point for running ICL experiments.

**Key functions:**
- `icl_prompt_audios_generation()` — Builds ICL prompts (types 1–4) with audio inputs for a given sample and selection strategy.
- `asr_prompt_audios_generation()` — Builds a plain ASR prompt (no ICL context).
- `select_indices_for_icl()` — Selects ICL pool indices based on the chosen strategy.
- `load_sonar_results()` — Loads precomputed SONAR similarity scores for embedding-based sample selection.
- `calculate_ppl()` — Computes perplexity of a target text given a prompt and optional audio context.
- `do_icl()` — Main loop: iterates over the test set, runs the chosen task, and saves results.
- `collect_attention_dfs()` — Collects per-layer self-attention matrices for visualization.


---

## `utils.py`

Shared utility functions used across scripts.


---

## `evaluate.py`

Evaluates transcription outputs against reference texts.

**Key responsibilities:**
- Loads hypothesis `.txt` files produced by `speechllm_icl_strategy.py`.
- Computes WER / CER against reference transcripts.
- Supports aggregation across multiple languages (e.g. ML-SUPERB2) and multiple shot settings.

---

## `finetune_phi4.py`

Fine-tunes the Phi-4-multimodal model on low-resource ASR data.

**Key responsibilities:**
- Loads a target language dataset and prepares audio–transcript pairs.
- Configures LoRA / PEFT adapters on top of Phi-4-multimodal.
- Runs a supervised fine-tuning loop with a language modelling objective.
- Saves adapter checkpoints for downstream evaluation.

