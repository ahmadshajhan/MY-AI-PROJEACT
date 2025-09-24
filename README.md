# LLM from Scratch — WinCode × Ahmad Edition ✨🎨

<div align="center">

<img src="https://raw.githubusercontent.com/your-org/your-repo/main/assets/banner-llm-sparkles.gif" alt="LLM From Scratch Colorful Banner" width="100%" />

<p>
  <strong>Build, train, and align modern LLMs end-to-end in PyTorch — with a bold, colorful, community-first vibe.</strong>
</p>

<p>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch&logoColor=white" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776ab?logo=python&logoColor=white" /></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-11.8%2B-76b900?logo=nvidia&logoColor=white" /></a>
  <a href="https://github.com/ahmad-wincode/llm-from-scratch/actions"><img src="https://img.shields.io/github/actions/workflow/status/ahmad-wincode/llm-from-scratch/ci.yml?label=CI&logo=github" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-8a2be2?logo=open-source-initiative&logoColor=white" /></a>
  <a href="https://discord.gg/wincode"><img src="https://img.shields.io/badge/Discord-WinCode%20Community-5865F2?logo=discord&logoColor=white" /></a>
  <img src="https://img.shields.io/badge/PRs-welcome-0aa80a?logo=github" />
  <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-ff3366" />
</p>

<p>
  <a href="#quickstart">Quickstart</a> •
  <a href="#curriculum--learning-path">Curriculum</a> •
  <a href="#features">Features</a> •
  <a href="#demo--notebooks">Demos</a> •
  <a href="#contributing">Contribute</a>
</p>

</div>

<p align="center">🌈 🟪🟦🟩🟨🟧🟥🟪🟦🟩🟨🟧🟥🟪🟦🟩🟨🟧🟥 🌈</p>

> Crafted with love for the Ahmad & WinCode Community. Add your flair, share your experiments, and level up together.

---

## Why this repo?

- Learn the full LLM pipeline by building it — no magic, no black boxes.
- Modern architecture patterns: RoPE, RMSNorm, SwiGLU, KV cache, sliding attention.
- Complete alignment stack: SFT → Reward Modeling → RLHF (PPO & GRPO).
- Friendly DX: configs, CLI, logging, CI, Docker, Colab, and reproducibility.
- Colorful docs, animated references, and community-first contributions.

---

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Demo & Notebooks](#demo--notebooks)
- [Project Structure](#project-structure)
- [Curriculum & Learning Path](#curriculum--learning-path)
- [Core Examples](#core-examples)
- [Training & Inference CLI](#training--inference-cli)
- [Evaluation & Logging](#evaluation--logging)
- [Scaling & Performance](#scaling--performance)
- [RLHF: PPO & GRPO](#rlhf-ppo--grpo)
- [Data & Tokenization](#data--tokenization)
- [FAQ](#faq)
- [Troubleshooting Cheatsheet](#troubleshooting-cheatsheet)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Community](#community)
- [License & Citation](#license--citation)
- [Acknowledgements](#acknowledgements)

---

## Features

- End-to-end, build-from-scratch Transformer in PyTorch 2.x
- Positional encodings: Learned, Sinusoidal, RoPE (Rotary)
- Normalization: LayerNorm vs RMSNorm
- MLP: GELU, SwiGLU
- Samplers: temperature, top-k, nucleus (top-p), repetition penalty
- KV Cache, sliding window attention, attention sink
- Tokenization: byte-level, BPE (tiktoken/HF)
- Configurable training: mixed precision, grad accumulation, schedulers
- Logging & profiling: TensorBoard, wandb, PyTorch Profiler
- Checkpointing, resume, evaluation loops
- RLHF: Reward Modeling + PPO + GRPO
- MoE: gating, load balancing, hybrid dense-MoE
- Scaling: DDP/FSDP/DeepSpeed, FlashAttention 2, xFormers, torch.compile
- Quantization options: bitsandbytes (8/4-bit), AWQ/GPTQ (tips)
- Docker + VS Code Devcontainers + Colab support
- Friendly API + CLI, modular codebase, comprehensive comments

---

## Quickstart

- Requirements: Python 3.10/3.11, PyTorch 2.2+ (CUDA Optional), Git

Option A — Conda
```bash
conda create -n llm_from_scratch python=3.11 -y
conda activate llm_from_scratch
pip install -U pip
pip install -r requirements.txt
```

Option B — uv (faster installs)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Option C — Docker
```bash
docker build -t wincode/llm-scratch .
docker run -it --gpus all -v $PWD:/workspace wincode/llm-scratch bash
```

Sanity test
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Train a tiny model
```bash
python train.py --config configs/tiny.yaml
```

Generate text
```bash
python generate.py --ckpt runs/tiny/latest.ckpt --prompt "WinCode LLM says:"
```

---

## Demo & Notebooks

- Open in Colab: [Launch](https://colab.research.google.com/) • Kaggle: [Notebook](https://kaggle.com/)
- Hugging Face Spaces Demo: [Try it](https://huggingface.co/spaces/)
- Example notebooks
  - 01_positional_embeddings.ipynb
  - 02_self_attention_first_principles.ipynb
  - 03_build_attention_head.ipynb
  - 10_train_tiny_llm.ipynb
  - 40_rlhf_ppo_walkthrough.ipynb

Tip: See assets/ for GIFs and diagrams; replace placeholders with your own visuals.

---

## Project Structure

```
llm_scratch/
  core/
    model.py            # Transformer, blocks, attention
    rope.py             # Rotary embeddings
    norms.py            # LayerNorm, RMSNorm
    mlp.py              # GELU, SwiGLU
    sampler.py          # top-k, top-p, temperature
  data/
    tokenize.py         # byte-level & BPE
    datasets.py         # streaming & shard-aware datasets
  train/
    loop.py             # train/val loop
    sched.py            # LRs, warmup, cosine
    utils.py            # mixed precision, grad clip
  rlhf/
    reward_model.py     # pairwise reward model
    ppo.py              # PPO trainer
    grpo.py             # GRPO trainer
  moe/
    layers.py           # experts, gating, load-balancing loss
  cli/
    train.py            # CLI wrapper for training
    generate.py         # sampling CLI
    eval.py             # perplexity, ppl@val
configs/
  tiny.yaml
  base.yaml
  moe_sft.yaml
  rlhf_ppo.yaml
assets/
  banner-llm-sparkles.gif
  divider-rainbow.gif
  diagrams/
requirements.txt
README.md
```

---

## Curriculum & Learning Path

Below is a colorful, extended version of your original curriculum — with extra context, links, and bite-size goals.

### Part 0 — Foundations & Mindset
- 0.1 Pipeline overview: pretraining → finetuning → alignment → serving
- 0.2 Setup: PyTorch, CUDA/Metal, mixed precision, profiling
- 0.3 Reproducibility: seeds, determinism, data splits
- 0.4 Debugging tactics: unit tests, gradient checks, NaN hunters

Install
```bash
conda create -n llm_from_scratch python=3.11 -y
conda activate llm_from_scratch
pip install -r requirements.txt
```

Mermaid: Training pipeline
```mermaid
flowchart LR
  A[Raw Text Corpora] --> B[Pretraining (Causal LM)]
  B --> C[Supervised Fine-tuning (SFT)]
  C --> D[Reward Modeling]
  D --> E[RLHF: PPO / GRPO]
  E --> F[Quantization + Serving]
```

### Part 1 — Core Transformer Architecture
- 1.1 Positional embeddings: learned vs sinusoidal vs RoPE
- 1.2 Self-attention from first principles (manual calc)
- 1.3 Single attention head in PyTorch
- 1.4 Multi-head attention (split/concat/proj)
- 1.5 MLP: GELU vs SwiGLU, expansion ratios
- 1.6 Residuals, LayerNorm/RMSNorm, full block

Key snippet (single head)
```python
class AttentionHead(torch.nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q = torch.nn.Linear(d_model, d_head, bias=False)
        self.k = torch.nn.Linear(d_model, d_head, bias=False)
        self.v = torch.nn.Linear(d_model, d_head, bias=False)
        self.scale = d_head ** -0.5

    def forward(self, x, mask=None):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        att = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None: att.masked_fill_(mask == 0, float('-inf'))
        p = att.softmax(dim=-1)
        return p @ V
```

### Part 2 — Training a Tiny LLM
- 2.1 Byte-level tokenization
- 2.2 Dataset batching & shift
- 2.3 Cross-entropy loss
- 2.4 From-scratch training loop
- 2.5 Sampling: temp/top-k/top-p
- 2.6 Val loss, early stopping

### Part 3 — Modernizing the Architecture
- 3.1 RMSNorm vs LayerNorm: gradient/variance behavior
- 3.2 RoPE: theory & implementation
- 3.3 SwiGLU activations
- 3.4 KV cache for faster inference
- 3.5 Sliding window attention, attention sink
- 3.6 Rolling buffer KV cache for streaming

### Part 4 — Scaling Up
- 4.1 BPE tokenization (tiktoken/HF)
- 4.2 Grad accumulation, mixed precision
- 4.3 LR schedules & warmup
- 4.4 Checkpointing & resume
- 4.5 Logging: TensorBoard / wandb
- 4.6 torch.compile, FlashAttention 2, xFormers

### Part 5 — Mixture-of-Experts (MoE)
- 5.1 Routing & load balancing (aux loss)
- 5.2 MoE layers in PyTorch
- 5.3 Hybrid dense+MoE, capacity, top-1/2 gating

Mermaid: MoE routing
```mermaid
flowchart TB
  X[Token Hidden] --> G[Gate (softmax)]
  G -->|top-1/2| E1[Expert 1]
  G -->|top-1/2| E2[Expert 2]
  E1 --> M[Merge (weighted sum)]
  E2 --> M
```

### Part 6 — Supervised Fine-Tuning (SFT)
- 6.1 Instruction formatting (prompt + response)
- 6.2 Causal LM with masked labels
- 6.3 Curriculum learning for instructions
- 6.4 Eval vs gold answers (BLEU/ROUGE optional)

### Part 7 — Reward Modeling
- 7.1 Preference datasets (pairwise)
- 7.2 Reward model (encoder or LM head)
- 7.3 Losses: Bradley–Terry, margin ranking
- 7.4 Sanity checks, reward scaling

Bradley–Terry (pairwise)
```latex
\mathcal{L}_{BT} = -\log \sigma(r(x^{+}) - r(x^{-}))
```

### Part 8 — RLHF with PPO
- 8.1 Policy: SFT LM + value head
- 8.2 Reward: learned reward model
- 8.3 PPO: KL penalty to stay close to SFT
- 8.4 Loop: sample → score → update
- 8.5 Tricks: reward norm, KL control, clip grads

PPO loss core
```latex
\mathcal{L}_{PPO} = -\mathbb{E}\left[\min\left(r_t(\theta) A_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)\right]
+ \beta \, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
```

### Part 9 — RLHF with GRPO
- 9.1 Group-relative baseline: sample k completions/prompt
- 9.2 A = reward − group mean
- 9.3 Policy-only clipped loss
- 9.4 Explicit KL(π‖π_ref) regularization
- 9.5 Practical loop differences

GRPO advantage
```latex
A_{i,j} = r_{i,j} - \frac{1}{k}\sum_{m=1}^{k} r_{i,m}
```

---

## Core Examples

Minimal training step
```python
def train_step(model, batch, optimizer, scaler, scheduler):
    model.train()
    x, y = batch["input_ids"], batch["labels"]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            y[:, 1:].contiguous().view(-1),
            ignore_index=-100
        )
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return loss.item()
```

Sampling with temperature + top-p
```python
tokens = tokenizer.encode("Ahmad & WinCode say: ")
generated = generate(model, tokens, max_new_tokens=128, temperature=0.8, top_p=0.9)
print(tokenizer.decode(generated))
```

Config (YAML)
```yaml
# configs/tiny.yaml
model:
  d_model: 256
  n_heads: 4
  n_layers: 8
  d_mlp: 1024
  vocab_size: 50257
  rope: true
  norm: rmsnorm
train:
  batch_size: 32
  seq_len: 256
  steps: 20000
  lr: 3e-4
  warmup_steps: 500
  weight_decay: 0.1
  grad_clip: 1.0
  amp: true
data:
  tokenizer: byte
  dataset: "data/tiny_text/*.txt"
log:
  wandb: true
  project: "wincode-llm-scratch"
```

---

## Training & Inference CLI

Train
```bash
python cli/train.py --config configs/base.yaml
# or override on the fly
python cli/train.py --config configs/base.yaml train.lr=2e-4 train.batch_size=64 model.rope=true
```

Evaluate PPL
```bash
python cli/eval.py --ckpt runs/base/latest.ckpt --split val
```

Generate
```bash
python cli/generate.py --ckpt runs/base/latest.ckpt \
  --prompt "Design me a colorful WinCode LLM README:" \
  --temperature 0.7 --top_p 0.9 --max_new_tokens 200
```

---

## Evaluation & Logging

- Logging: wandb or TensorBoard (scalars, histograms, text samples).
- Checkpoints: every N steps, keep top-K by val loss.
- Eval: loss, perplexity, length-normalized metrics.
- Profiling: PyTorch Profiler traces (TensorBoard plugin).

Tip: enable wandb with WANDB_API_KEY env variable.

---

## Scaling & Performance

- Mixed precision (autocast + GradScaler)
- Gradient accumulation for large batch sizes
- torch.compile for graph capture
- FlashAttention 2 via optional backend
- DDP/FSDP/DeepSpeed ZeRO-2/3 setups
- Activation checkpointing
- Offloading (CPU/NVMe) tips
- Quantization (int8/int4) for inference

Memory math (rough)
```latex
\text{Mem} \approx 2 \cdot \text{Params} \cdot \text{bytes} + \text{Optimizer} + \text{Activations}
```

---

## RLHF: PPO & GRPO

Mermaid: PPO loop
```mermaid
sequenceDiagram
  participant U as Prompts
  participant P as Policy (SFT LM + V)
  participant R as Reward Model
  participant Opt as PPO Optimizer
  U->>P: generate k completions
  P->>R: score completions
  R-->>P: rewards
  P->>Opt: compute advantages, KL, update
```

GRPO highlights
- No value head
- k samples per prompt
- Group mean rewards as baseline
- Explicit KL penalty

---

## Data & Tokenization

- Byte-level tokenizer for tiny experiments
- BPE via tiktoken / Hugging Face tokenizers
- Streaming datasets (WebDataset shards)
- Dedup & filtering suggestions (minhash/simhash)
- Safety filters & PII scrubbing (optional)
- Balanced mixes for instruction tuning

Example: BPE build
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=50257, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
tok.train(files=["data/corpus.txt"], trainer=trainer)
tok.save("bpe.json")
```

---

## FAQ

- Do I need a GPU?
  - Not strictly for tiny runs, but yes for anything practical. Start with a single RTX 3060/3070+, or use Colab/Kaggle.
- I get CUDA OOM
  - Reduce batch_size/seq_len, enable grad accumulation, switch to bf16/fp16, use activation checkpointing.
- Windows?
  - Use WSL2 + CUDA. Or Docker with GPU support.
- How big should my dataset be?
  - Tiny runs: 10–100MB just to verify. Serious pretraining: many GBs to TBs.
- Why RMSNorm?
  - Often more stable with modern RoPE-based LLMs and works well with SwiGLU.

---

## Troubleshooting Cheatsheet

- PyTorch/CUDA mismatch
  - Install correct torch build: pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
- NaNs in loss
  - Lower LR, enable grad clipping, check for fp16 underflow, verify data/labels alignment.
- Bad generations
  - Try higher temperature and top-p, train longer, inspect val loss, remove repetition penalty if over-penalizing.

---

## Roadmap

- [x] Tiny Transformer + sampling
- [x] RoPE, RMSNorm, SwiGLU
- [x] KV cache, sliding attention
- [x] BPE tokenizer
- [x] SFT pipeline
- [x] Reward model + PPO
- [x] GRPO trainer
- [x] MoE layers (top-1/2)
- [ ] FlashAttention 2 integration switch
- [ ] Quantization recipes (bitsandbytes, AWQ/GPTQ)
- [ ] FSDP + ZeRO configs
- [ ] Extensive eval suite (MMLU/tiny tasks)
- [ ] Docs site with colorful theme

---

## Contributing

We welcome contributions from Ahmad, WinCode community, and you!

- Fork, create a feature branch
- Add tests where applicable
- Follow style: ruff + black
- Commit using Conventional Commits (feat:, fix:, docs:, refactor:, chore:)
- Open a PR with screenshots/logs where relevant

Dev setup
```bash
pip install -r requirements-dev.txt
pre-commit install
pytest -q
ruff check . && black --check .
```

Code of Conduct: Be kind, constructive, and inclusive. We grow together.

---

## Community

- Discord: Join the WinCode server — share runs, troubleshoot, and showcase demos.
- Showcase: Submit your models, configs, and GIFs via PR (assets/showcase/).
- Weekly Meetup: Hands-on deep dive on a module each week (announce in Discord).

Shout-out to Ahmad and the WinCode community for the colorful energy and collaboration!

---

## License & Citation

- License: MIT — do anything, just keep the notice.
- If this helps your research or project, please cite:

```bibtex
@software{wincode_llm_scratch_2025,
  author = {Ahmad, WinCode Community},
  title  = {LLM from Scratch — WinCode × Ahmad Edition},
  year   = {2025},
  url    = {https://github.com/ahmad-wincode/llm-from-scratch}
}
```

---

## Acknowledgements

- Transformer: Vaswani et al. 2017
- RoPE: Su et al. 2021
- RMSNorm: Zhang & Sennrich 2019
- SwiGLU: Shazeer 2020
- PPO: Schulman et al. 2017
- GRPO: Group-Relative Policy Optimization variants in recent RLHF papers
- Open-source inspirations: nanoGPT, minGPT, HF Transformers, trl, DeepSpeed

---

Back to top ↑

---

