# Deterministic LoRA Fine-Tuning for Nemotron Reasoning

![Kaggle](https://img.shields.io/badge/Kaggle-Bronze%20Medal-CA9266)
![Rank](https://img.shields.io/badge/Rank-333%2F4185-blue)
![Metric](https://img.shields.io/badge/Metric-Accuracy-2ea44f)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20r%3D32-6f42c1)

> This repository documents a medal-winning solution to the NVIDIA Nemotron Model Reasoning Challenge. The task asks a model to infer transformation rules from input-output examples and solve a held-out instance across bit manipulation, ciphers, arithmetic, unit conversion, and related structured reasoning categories. The solution applies supervised fine-tuning to Nemotron-3-Nano-30B-A3B with a rank-32 LoRA adapter under the competition's deterministic vLLM evaluation protocol. The training notebook implements architecture-specific LoRA injection, fp32/bf16 mixed-precision control, Cut Cross-Entropy for memory-efficient 8192-token training, MoE expert weight tying, Mamba CUDA fast-path activation, token-level loss masking, and submission-compatible adapter packaging. The solution achieved a Bronze Medal, ranking 333rd out of 4,185 teams (top 8%). It demonstrates practical parameter-efficient fine-tuning for long-context reasoning on a 30B-class hybrid MoE model, with a clear separation between notebook-implemented components and solution-design components reported in public community write-ups.

## 1. Task & Dataset

### Task definition

Each prompt contains several input-output examples and one target instance. The model must identify the underlying transformation rule and produce the final answer inside a `\boxed{}` expression. The benchmark covers multiple structured reasoning categories, including:

- bit manipulation;
- numeric equations and cryptarithms;
- ciphers and text transforms;
- numeral systems;
- unit conversion;
- gravity and related physical calculations.

### Data form

| File | Fields | Description |
|---|---|---|
| `train.csv` | `id`, `prompt`, `answer` | Training puzzles with ground-truth solutions |
| `test.csv` | `id`, `prompt` | Sample test format; the private evaluation uses several hundred problems |
| `synthetic.json` | `tokens`, `mask` | Tokenized CoT corpus consumed by the training notebook |

The public test sample is intentionally small. The submitted run therefore depends on the tokenized synthetic corpus mounted through the referenced Kaggle dataset.

### Evaluation metric

Submissions are scored by **Accuracy**. The metric extracts the final answer from the generated text, prioritising a valid `\boxed{}` expression and otherwise applying fallback heuristics. A prediction is correct when it matches the ground truth exactly as a string or within a relative numerical tolerance of `1e-2`.

Evaluation is performed with vLLM using:

| Parameter | Value |
|---|---:|
| `max_lora_rank` | 32 |
| `max_tokens` | 7680 |
| `top_p` | 1.0 |
| `temperature` | 0.0 |
| `max_num_seqs` | 64 |
| `gpu_memory_utilization` | 0.85 |
| `max_model_len` | 8192 |

### Research and engineering pain points

Structured reasoning requires long, low-variance chains of thought, while the evaluation imposes a strict token budget and deterministic decoding. The base model is a 30B-class MoE model with Mamba and attention components, creating substantial memory pressure during 8192-token training. Sparse expert activation also makes parameter updates uneven. These constraints motivate parameter-efficient adaptation, memory-aware loss computation, and careful control of numerical precision.

## 2. Solution Pipeline

### 2.1 Preprocessing

- Read the pre-tokenised corpus from `synthetic.json`.
- Use `tokens` as the full sequence and `mask` as the token-level loss indicator.
- Skip empty samples and samples whose mask contains no active token.
- Truncate sequences longer than 8192 tokens.
- Construct the next-token objective with `input = tokens[:-1]`, `target = tokens[1:]`, and `weights = mask[1:]`.
- Ignore prompt tokens during optimisation and keep loss on answer or CoT tokens only.

### 2.2 Data synthesis and augmentation

The provided training notebook does not generate the CoT corpus. It consumes externally generated tokenised data. The attached solution report and the public discussion describe code-generated deterministic CoT traces with high candidate coverage, minimal ambiguity, and no reliance on path diversity.

This distinction is important: the notebook implementation and the wider algorithmic design are documented separately in this repository.

### 2.3 Model and parameter-efficient adaptation

- Base model: `Nemotron-3-Nano-30B-A3B` in bf16.
- Adapter: LoRA rank 32, alpha 64, dropout 0.
- Injected modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `in_proj`, `out_proj`, and `lm_head`.
- The notebook manually injects LoRA into `lm_head` because the default MoE path drops that adapter.
- LoRA parameters and MoE router parameters are kept in fp32; frozen base weights remain in bf16.
- The Mamba CUDA fast path is explicitly enabled after validating `causal_conv1d` and `mamba_ssm`.

### 2.4 Training strategy

- Optimizer: AdamW with `betas=(0.9, 0.95)`, `eps=1e-8`, and zero weight decay.
- Learning rate: `2e-4`, linearly decayed to zero.
- Logical batch size: 32. Micro-batch size: 4 with gradient accumulation.
- Maximum optimisation steps: 1000, capped by the number of complete corpus batches.
- Mixed precision: bf16 autocast with fp32 LoRA and router parameters.
- Memory strategy: Unsloth gradient checkpointing, eager attention, and right-side padding.
- Stability strategy: expert-layer LoRA weight tying at initialisation and after backward.

### 2.5 Loss

The notebook replaces the standard CausalLM forward with a Cut Cross-Entropy path. Instead of materialising the full `[batch, sequence, vocabulary]` logits tensor, `linear_cross_entropy` computes per-token cross-entropy from hidden states and the effective `lm_head` weight. The weighted loss is:

```text
loss = sum(per_token_ce * token_weight) / sum(token_weight)
```

This preserves token-level masking while reducing peak VRAM during long-sequence training.

### 2.6 Cross-validation and OOF

No K-fold, grouped validation, holdout evaluation, or OOF generation is present in the provided notebook. The public discussion reports category-level validation scores for a related community notebook, but those values should not be presented as the validation result of this submitted adapter.

### 2.7 Ensemble

No model ensemble, adapter averaging, weight merging, or multi-checkpoint fusion is implemented in the supplied training notebook.

### 2.8 Post-processing and packaging

- Remove stale `adapter*` files from the working directory.
- Save the PEFT adapter and `adapter_config.json`.
- Rename `lm_head` LoRA keys from `base_model.model.lm_head.*` to `base_model.model.backbone.lm_head.*`.
- Remove the `unsloth_compiled_cache` directory.
- Package all adapter files into `submission.zip`.

The key-path repair prevents the manually injected `lm_head` adapter from being ignored by the submission loader.

## 3. Key Innovations

1. **Deterministic CoT supervision for a temperature-zero benchmark.** The solution treats SFT as a way to teach a fixed reasoning procedure, rather than relying on RL or distillation. The target is not answer diversity but stable token-level reproduction of a known algorithm.

2. **Task-level dimensionality reduction for bit manipulation.** The published solution design reduces the search from full expressions to 354 input-bit candidates, comprising 18 unary and 336 binary combinations per output bit. Bitsum hashing and bidirectional stride matching further reduce the token cost. This component is documented in the public community write-up and solution report; it is not implemented in the supplied training notebook.

3. **Architecture-specific LoRA and precision control.** LoRA is injected into attention, MLP, Mamba projections, and `lm_head`. The base remains bf16 while sensitive LoRA and router parameters are promoted to fp32. This supports both memory efficiency and stable adaptation of a hybrid MoE model.

4. **Memory-efficient training for long-context reasoning.** Cut Cross-Entropy bypasses logits materialisation and computes per-token loss directly, enabling 8192-token training on a 30B-class MoE model within the available accelerator memory.

5. **MoE-aware optimisation and submission repair.** Expert-layer LoRA gradients are tied across experts to reduce sparse-update divergence, and the final adapter keys are rewritten so the manually injected `lm_head` module loads correctly during evaluation.

## 4. Project Structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── configs/
│   └── reference_config.yaml
├── data/
│   └── README.md
├── docs/
│   ├── PIPELINE.md
│   └── REPRODUCIBILITY.md
├── notebooks/
│   ├── README.md
│   └── nvidia_nemotron_solution_code.ipynb
├── outputs/
│   └── .gitkeep
├── scripts/
│   └── run_kaggle_training.sh
└── src/
    └── README.md
```

The notebook is the canonical implementation in this repository. A production refactor would move configuration, corpus loading, model construction, training, and packaging into `src/` while preserving the current behaviour.

## 5. Environment & Dependencies

The original run uses Kaggle's Linux, Python 3.12, CUDA environment and the offline wheel sources listed below. `causal-conv1d` and `mamba-ssm` must match the installed PyTorch, CUDA, C++ ABI, Python, and Linux platform versions.

```text
unsloth
trl
peft
transformers
datasets
accelerate
bitsandbytes
kagglehub
protobuf==6.33.5
sentencepiece
safetensors
huggingface_hub
cut-cross-entropy
```

The notebook installs these CUDA kernels from prebuilt wheels:

```text
causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

See [requirements.txt](requirements.txt) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the full setup.

## 6. Reproduction Guide

1. Create a Kaggle Notebook with GPU enabled and attach the competition datasets and model.
2. Attach the following resources:

```text
llkh0a/rtx-wheels
huikang/huikang-nemotron-repository-snapshot
mayukh18/nemotron-packages
metric/nemotron-3-nano-30b-a3b-bf16
```

3. Open [notebooks/nvidia_nemotron_solution_code.ipynb](notebooks/nvidia_nemotron_solution_code.ipynb).
4. Confirm that the mounted paths match the constants in `configs/reference_config.yaml`.
5. Execute the notebook from top to bottom.
6. Confirm that training completes and that `submission.zip` is created.
7. Validate that the archive contains `adapter_config.json` and `adapter_model.safetensors`.
8. Confirm that the adapter rank does not exceed 32 and that `lm_head` keys use the `backbone.lm_head` path.

The repository does not distribute the private tokenized corpus or the base model weights.

## 7. Result & Discussion

### Official result

| Item | Result |
|---|---|
| Competition | NVIDIA Nemotron Model Reasoning Challenge |
| Medal | Bronze Medal |
| Final rank | 333 / 4,185 |
| Percentile | Approximately top 8% |
| Submission | Rank-32 LoRA adapter for Nemotron-3-Nano-30B |

### Main difficulties

- **Long-sequence memory pressure.** The benchmark permits 8192-token model inputs and 7680 generated tokens. Standard logits materialisation is expensive for a 30B-class vocabulary.
- **MoE update imbalance.** Sparse expert routing produces uneven gradients. Expert-level LoRA tying was introduced to stabilise the optimisation.
- **Online and offline environment mismatch.** The notebook depends on prebuilt CUDA wheels and offline Kaggle paths. A version mismatch can disable the Mamba fast path or cause import failures.
- **Reproducibility boundary.** The corpus generator and task-specific bit-manipulation solver are not included in the provided training notebook. The repository therefore documents those methods as external solution-design components rather than claiming they are part of the trainer.
- **Validation gap.** No fold-level validation, OOF predictions, or ensemble are present in the supplied notebook. The official leaderboard result is the strongest verified performance signal.

### Limitations

- The solution is not reproducible from raw prompts using this repository alone.
- The synthetic corpus, solver, and full CoT generator are not included.
- The notebook is tied to one Kaggle environment and does not expose a unified CLI or YAML-driven runtime configuration.
- No systematic ablation, calibration study, or per-category error analysis for the submitted adapter is included.
- No independent validation protocol establishes the relationship between local development scores and the private leaderboard.

### Future improvements

- Release the deterministic CoT generator and solver code.
- Add stratified per-category validation and OOF scoring.
- Add controlled ablations for CCE, MoE weight tying, `lm_head` LoRA, and precision settings.
- Refactor the notebook into a configuration-driven training package with unit tests.
- Evaluate adapter merging and ensemble strategies only after a trustworthy validation protocol is established.
- Add inference-time answer validation without altering the deterministic decoding protocol.

## 8. Citation & Acknowledgement

### Competition citation

```bibtex
@misc{nvidia_nemotron_reasoning_2026,
  title        = {NVIDIA Nemotron Model Reasoning Challenge},
  author       = {Semaan, Jamil C and Puget, Jean-Francois and Henkel, Christof and Dong, Yi and Howard, Addison and Oldacre, Ashley and Holbrook, Ryan and Alexiuk, Chris and Kao, Rebecca},
  year         = {2026},
  howpublished = {Kaggle},
  url          = {https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge}
}
```

### Public solution-design references

- Tong Hui Kang, [Strategy to solve 85% of bit manipulation](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307).
- William Lu, [R00021 public notebook](https://www.kaggle.com/code/williamlu931/r00021).

### Open-source acknowledgements

This work builds on NVIDIA Nemotron, Unsloth, Hugging Face Transformers, PEFT, TRL, Mamba, `causal-conv1d`, Cut Cross-Entropy, vLLM, and the Kaggle competition environment. The repository preserves the original notebook and documents the source boundary of public solution-design material.
