# Accelerating GRPO Post-Training with Completion Pruning (CPPO)

GRPO and CPPO post-training pipelines for **Qwen3-0.6B** on **DAPO-Math-17k**,
built on [TRL](https://github.com/huggingface/trl), evaluated with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on
`gsm8k`, `minerva_math` and `aime24`.

> **The question this repository answers:** how accurate is
> [CPPO](https://arxiv.org/pdf/2503.22342), and how much does it actually
> accelerate RL training?
>
> **Short answer, measured on 1× A100 over 8,192 DAPO-Math problems:**
> CPPO **halves wall-clock training time** (2.03× at P = 0.875) and costs
> **nothing measurable in accuracy** — at a matched optimiser-step count,
> pruning 75% of completions scored *higher* than the baseline on
> minerva_math (+0.56 pp).
>
> The interesting part is *why* it beat the prediction. A cost model based on
> Amdahl's law, `S(P) = 1/((1-f) + f(1-P))` with a measured `f = 0.46`, caps
> the speedup at 1.86×. The measurement reached **2.06×** — because the model
> assumes the rollout is untouched, and it is not: dynamic allocation enlarges
> the generation batch, which raises vLLM throughput by a further 12-20%.
> The bound turns out to be a **lower** bound.
> See [Results](#8-results) for the falsification and
> [Analysis](#7-analysis) for the model.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Repository layout](#2-repository-layout)
3. [Background: GRPO](#3-background-grpo)
4. [CPPO: completion pruning](#4-cppo-completion-pruning)
5. [Implementation](#5-implementation)
6. [Experimental setup](#6-experimental-setup)
7. [Analysis](#7-analysis)
8. [Results](#8-results)
9. [Reproducing the study](#9-reproducing-the-study)
10. [Code quality](#10-code-quality)
11. [Limitations](#11-limitations)
12. [References](#12-references)

---

## 1. Quick start

```bash
git clone https://github.com/saifutdinov2008/cppo-grpo-acceleration.git
cd cppo-grpo-acceleration

# Creates .venv and installs the project plus its eval/dev extras.
# Add --with-vllm on a CUDA machine to install the fast rollout backend.
bash scripts/setup.sh
source .venv/bin/activate

# Validate the whole pipeline without a GPU (~ a few minutes):
# lint + types + unit tests, then two real training steps of GRPO and of
# CPPO on Qwen3-0.6B and DAPO-Math-17k.
bash scripts/smoke_test.sh
```

Then, on a GPU:

```bash
bash scripts/train.sh configs/grpo.yaml         # GRPO baseline
bash scripts/train.sh configs/cppo_p75.yaml     # CPPO, P = 0.75
bash scripts/evaluate.sh outputs/cppo-p75 results/cppo-p75/eval.json

# ...or the entire study (baseline + 3 pruning rates + ablation + evals):
bash scripts/run_all.sh
```

---

## 2. Repository layout

```
src/cppo/
  pruning.py     Top-k selection by |advantage| — the mathematical core
  geometry.py    Pruning rate -> TRL batch parameters (dynamic allocation)
  trainer.py     ProfiledGRPOTrainer (baseline) and CPPOTrainer
  profiling.py   Stage timing + peak memory, shared by both trainers
  rewards.py     Format and accuracy rewards (math_verify)
  data.py        DAPO-Math-17k loading and prompt templating
  train.py       Training entry point            (`python -m cppo.train`)
  evaluate.py    lm-evaluation-harness wrapper   (`python -m cppo.evaluate`)
  report.py      Renders result tables           (`python -m cppo.report`)
configs/         base.yaml + one file per experiment (YAML `extends:` inheritance)
scripts/         setup / preflight / train / evaluate / benchmark / smoke_test / run_all / lint
docs/RUNBOOK.md  rented-GPU walkthrough for the full study
benchmarks/      update_stage_benchmark.py — isolates CPPO's effect from the rollout
                 plot_results.py            — renders the report's figures
tests/           93 unit tests + 4 end-to-end trainer tests
report/          report.tex, references.bib, figures/, Makefile
results/         JSON artefacts produced by the runs (measurements live here)
```

---

## 3. Background: GRPO

For a question `q`, the old policy samples a **group** of `G` completions.
GRPO replaces PPO's learned critic with a baseline computed from that group,
maximising

```
J_GRPO(θ) = E_q, {o_i} [ (1/G) Σ_i (1/|o_i|) Σ_t {
                min( ρ_i,t(θ)·A_i ,  clip(ρ_i,t(θ), 1-ε, 1+ε)·A_i )
                − β·D_KL[π_θ ‖ π_ref] } ]
```

with the token-level importance ratio `ρ_i,t(θ) = π_θ(o_i,t|·) / π_old(o_i,t|·)`
and the group-standardised advantage

```
A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)
```

`A_i` is **one scalar per completion** — constant across its tokens. That is
the hook CPPO hangs everything on.

The reward is rule-based, `r_i = R_format(o_i) + R_accuracy(o_i)`, with
`R_format ∈ {0,1}` and `R_accuracy ∈ {0,2}`. No reward model is trained.

**Where the time goes.** Per optimiser step with `B` questions:

```
T_step  =  B·G·c_gen                                   (rollout)
        +  B·G·(n_fwd·c_fwd + c_bwd)                   (update)
```

`n_fwd` counts forward passes over completions: the policy, plus the reference
model when `β ≠ 0`, plus the old policy when samples are off-policy. Both terms
grow linearly in `G` — which is exactly why bigger, better-conditioned groups
are so expensive.

---

## 4. CPPO: completion pruning

### 4.1 The observation

Differentiating the objective and dropping the KL term gives

```
∇_θ J(θ) ≈ E [ (1/G) Σ_i (1/|o_i|) Σ_t  ρ_i,t(θ) · A_i · ∇_θ log π_θ(o_i,t|·) ]
                                        └─post-fwd─┘  └─PRE-fwd─┘  └──post-fwd──┘
```

All three factors must be non-negligible for a completion to move the policy.
Two are knowable only *after* the expensive forward pass — but `A_i` is known
*before* it, as soon as the group has been scored. A completion with `|A_i| ≈ 0`
can therefore be identified and discarded before any compute is spent on it.

### 4.2 Which completions get discarded

With a two-part reward there are four outcome types:

| Format | Answer | Typical \|A\| | Training signal |
|---|---|---|---|
| ✓ | ✓ | large | "do this" — clean positive signal |
| ✗ | ✗ | large | "avoid this" — clean negative signal |
| ✓ | ✗ | small | ambiguous: rewards form over substance |
| ✗ | ✓ | small | ambiguous: rewards luck over method |

Pruning by `|A_i|` removes precisely the partially-correct completions whose
gradient is both weak *and* semantically muddled. That is why CPPO can
**improve** accuracy rather than merely preserve it.

### 4.3 The pruned objective

Rather than thresholding `|A_i| ≥ γ` — which would leave each device with a
different number of surviving completions, so the slowest device sets the pace
(the "bucket effect") — CPPO keeps a fixed top-`k` per group:

```
k = floor(G · (1 − P)),                     P ∈ [0, 1)
I = { i : |A_i| is among the top k values }

J_CPPO(θ) = E [ (1/k) Σ_{i∈I} (1/|o_i|) Σ_t { min(...) − β·D_KL[...] } ]
                └── 1/k, not 1/G ──┘
```

The normaliser matters: leaving `1/G` in place would shrink the gradient by
`k/G` and act as a silent learning-rate decay.

### 4.4 Dynamic completion allocation

Pruning leaves `G−k` slots empty in the update stage, under-occupying the
accelerator. CPPO refills them with completions from **additional questions**,
so each generation round covers

```
m = floor(G / k)
```

times more questions. Total rollout work per epoch is unchanged — the same
questions are still answered `G` times — but the epoch needs `m`× fewer
optimiser steps, each as wide as the baseline's.

### 4.5 Algorithm

```
k ← floor(G(1−P));   m ← floor(G/k)
for each generation round:
    sample m·B questions                          # allocation: m× more questions
    for each question q_j:
        sample G completions from π_old(·|q_j)    # rollout — unchanged
        score r_j,i = R_format + R_accuracy
        A_j,i ← (r_j,i − mean_i r_j,·) / std_i r_j,·       # PRE-forward info
        I_j ← top-k over { |A_j,i| }                        # ← PRUNING
    B ← ⋃_j { o_j,i : i ∈ I_j }                   # ≈ B·G completions again
    forward+backward π_θ on B only; optimiser step
```

Only two lines differ from GRPO: the top-`k` selection, and the `m`-fold
widening of the question batch.

---

## 5. Implementation

The package **layers on** TRL rather than forking it, so the GRPO baseline is
provably stock TRL (`ProfiledGRPOTrainer` adds instrumentation and nothing
else). Both trainers share the same profiling mixin, so the baseline and the
accelerated run are measured by identical code.

### 5.1 Selection

`cppo/pruning.py` builds the retention mask per group with a **stable**
descending sort of `|A|`, so ties resolve to the lowest index and every
data-parallel rank selects the same completions from the same advantages
without communicating.

### 5.2 Dynamic allocation needs no TRL surgery

TRL derives its rollout batch as

```
generation_batch_size = per_device_train_batch_size × num_processes × steps_per_generation
```

samples `generation_batch_size / G` distinct questions per round, and splits
the result into `steps_per_generation` optimiser micro-batches.

The useful observation: **multiplying `per_device_train_batch_size` by `m`
makes TRL sample `m`× more questions per round, and pruning then shrinks each
micro-batch back to exactly the baseline width.** No batching code is touched.
For `G = 8`, baseline micro-batch 16, `steps_per_generation = 8`:

| Run | P | k | m | `per_device_train_batch_size` | Questions/round | Update micro-batch |
|---|---:|---:|---:|---:|---:|---:|
| GRPO baseline | 0.000 | 8 | 1 | 16 | 16 | **16** |
| CPPO P=0.50 | 0.500 | 4 | 2 | 32 | 32 | **16** |
| CPPO P=0.75 | 0.750 | 2 | 4 | 64 | 64 | **16** |
| CPPO P=0.875 | 0.875 | 1 | 8 | 128 | 128 | **16** |
| CPPO P=0.75, no allocation | 0.750 | 2 | 1 | 16 | 16 | 4 |

Identical update-stage tensor shapes — identical memory, identical kernel
occupancy — while covering `m`× more questions per step. The last row is the
paper's "+ Completion Pruning" ablation: pruning without refilling.

This arithmetic lives in `cppo/geometry.py` and is covered by unit tests.

### 5.3 Where pruning is applied, and why

The CPPO paper prunes before the policy, reference **and** old-policy forward
passes. **TRL computes rewards — and hence advantages — *after* the reference
and old-policy log-probabilities**, so a literal "prune first" ordering would
mean reimplementing a 600-line private method against internal TRL state:
fragile across versions and hard to audit.

This implementation instead prunes at the end of the rollout, immediately
before the policy forward/backward, and adopts a configuration in which the
auxiliary passes **do not exist**:

- **`β = 0`** — no reference model, hence no reference forward pass. This is
  TRL's own default, and it is exactly the KL-free gradient approximation from
  which CPPO derives its pruning criterion. It also removes a whole model from
  GPU memory.
- **`μ = 1`** with generation aligned to the optimiser step — samples are
  on-policy, so TRL skips the old-policy forward pass.
- **vLLM importance-sampling correction off** — otherwise TRL recomputes
  old-policy log-probabilities.

Under these settings the only pass over completions is the policy's, and
pruning before it reproduces the CPPO objective exactly. If a user overrides
them, `CPPOTrainer` **warns**, naming each unpruned forward pass: training
stays correct, but the measured speedup becomes a lower bound. Surfacing that
trade-off explicitly is deliberate.

### 5.4 Correctness details that are easy to get wrong

- **Loss normalisation.** TRL's default `dapo` loss divides by
  `num_items_in_batch`, the loss-carrying token count of the *whole* generation
  batch. Pruning removes tokens; leaving the pre-pruning count would scale the
  loss — and the effective learning rate — by `k/G`. The trainer recomputes the
  normaliser over the retained completions. This is the implementation-level
  counterpart of the `1/k` in the objective.
- **Group locality.** Pruning is per question, so a group must live entirely on
  one device. The trainer validates that the local generation batch is a whole
  multiple of `G` and fails loudly otherwise.
- **Degenerate groups.** When all `G` completions get the same reward, `std = 0`
  and every advantage is zero: the group contributes *no* gradient yet still
  costs a full forward and backward. The trainer logs the fraction of such
  groups (`cppo/frac_degenerate_groups`) and offers an optional
  `drop_zero_advantage` mode. That mode is exact only when `β = 0` and is
  restricted to single-process runs, since a data-dependent batch size would
  deadlock a collective operation.

### 5.5 Logged diagnostics

Every CPPO step logs:

| Metric | Meaning |
|---|---|
| `cppo/retention` | fraction of completions surviving pruning (`≈ k/G`) |
| `cppo/retained_signal_fraction` | share of the batch's total `\|A\|` mass kept |
| `cppo/abs_advantage_kept` / `_dropped` | mean `\|A\|` on each side of the cut |
| `cppo/frac_degenerate_groups` | groups with zero reward variance |

`retained_signal_fraction` is the key number: it quantifies *how little* is
actually thrown away, and is the mechanism behind CPPO's accuracy preservation.

### 5.6 Accelerating the rollout stage

The task asks for acceleration of **both** stages. CPPO addresses the update
stage only — advantages do not exist until sampling has finished, so no
completion-pruning method can shrink the rollout. The three costs named in the
problem statement map onto the pipeline like this:

| Cost named in the task | What this pipeline does | Where |
|---|---|---|
| Simultaneous storage of multiple LLMs in GPU memory | `beta = 0` removes the reference model entirely — no second set of weights, no reference forward pass. vLLM **sleep mode** releases the sampler's KV cache and weights between rollouts, so the sampler and the optimiser never hold memory at the same time. Optional LoRA (`use_peft`) drops optimiser state from 0.6B parameters to the adapter's. | `configs/base.yaml`, `cppo/train.py` |
| Slow autoregressive rollout requiring group sampling | vLLM with PagedAttention and continuous batching, in **colocate** mode so the sampler shares the training process instead of needing a second GPU. CPPO's dynamic allocation enlarges the generation batch by `m = G // k`, which raises sampler occupancy at no extra cost per question. | `use_vllm`, `vllm_mode`, `geometry.py` |
| Backpropagation updates | **CPPO**: back-propagate through `k` of `G` completions, chosen by `\|A_i\|`. Plus gradient checkpointing and bf16. | `cppo/pruning.py`, `cppo/trainer.py` |

Two of these are worth stating plainly because they are easy to miss:

- **`beta = 0` is a memory optimisation, not just a modelling choice.** It is
  the single largest memory saving in the pipeline — a whole model removed —
  and it is what makes pruning-before-every-forward-pass exact (§5.3).
- **Dynamic allocation helps the rollout too.** It is usually described as an
  update-stage trick, but enlarging the generation batch by `m` gives vLLM more
  sequences to batch per step, which is exactly what its continuous batching
  needs to reach high throughput.

What this pipeline does **not** do is reduce the number of tokens generated.
That is the remaining lever, and §7.1 explains why it is the one that now
matters most.

---

## 6. Experimental setup

| | |
|---|---|
| Policy model | `Qwen/Qwen3-0.6B` |
| Training data | `open-r1/DAPO-Math-17k-Processed` (config `en`) |
| Training subset | 8,192 problems, one epoch |
| Reward | format `{0,1}` + accuracy `{0,2}` via `math_verify` |
| Prompt | chat template, "reason step by step, answer in `\boxed{}`" |
| Thinking mode | **off** (`enable_thinking: false`) — see below |
| Group size `G` | 8 |
| Max completion length | 1024 tokens |
| Sampling | temperature 1.0, top-p 1.0 |
| Objective | `dapo` loss, ε = 0.2, ε_high = 0.28, **β = 0** |
| Optimiser | AdamW, lr 1e-6, constant + 10 warmup steps, grad-clip 0.2 |
| Rollout backend | vLLM (colocate mode) |
| Evaluation | `gsm8k` (5-shot), `minerva_math` (4-shot), `aime24` (0-shot) |
| Reference hardware | 1× A100/H100 80GB |

**Protocol.** Every run consumes the **same 8,192 problems in the same order**.
Because dynamic allocation covers `m`× more questions per optimiser step, CPPO
finishes that epoch in `m`× fewer steps. Fixing the *data* rather than the step
count is what makes a wall-clock comparison meaningful — fixing steps instead
would let CPPO see `m`× more data and confound speed with sample efficiency.

The results table therefore reports **questions per second** as the headline
metric rather than raw wall clock. Throughput is comparable under either
protocol: it credits an algorithm for covering the same training data in less
time, and never for covering more data in the same time.

**Why Qwen3's thinking mode is off.** Qwen3 is a hybrid thinking model: left
on, its chat template opens a `<think>` block and the policy reasons until it
exhausts the completion budget. Measured on this pipeline at 1024 tokens,
*every* completion was truncated — `min_length = max_length = 1024`,
`clipped_ratio = 1.0`, `mean_terminated_length = 0` — so none ever reached a
`\boxed{}` answer, `format_reward` was identically zero, and with
`mask_truncated_completions` the entire batch was masked out of the loss:
`grad_norm = 0`. The run would have trained for hours and moved no weights.

A 0.6B policy's chain of thought does not fit an affordable rollout budget, so
thinking is disabled for **both training and evaluation** (`lm_eval`'s vLLM
backend takes the same `enable_thinking` flag). Holding it identical on both
sides is what keeps the prompt formats aligned. `scripts/preflight.sh` now
fails outright if more than 50% of completions hit the cap.

**Prompt/evaluation alignment.** The policy is trained with a chat template and
a `\boxed{}` answer convention, and evaluated through `lm_eval`'s
`--apply_chat_template` with the *same* system prompt. `minerva_math` and
`aime24` extract from `\boxed{}`, and `gsm8k`'s `flexible-extract` filter takes
the last number, so the training format transfers to all three benchmarks.

---

## 7. Analysis

### 7.1 A cost model for CPPO's speedup

CPPO changes the update stage and leaves the rollout untouched. Let `f` be the
fraction of baseline training time spent in the update stage; update work
scales by `k/G = 1−P`. Amdahl's law then bounds the end-to-end speedup:

```
S(P) = 1 / ( (1−f) + f·(1−P) )          S_max = lim_{P→1} S(P) = 1 / (1−f)
```

This one equation explains the whole spread of published numbers:

- **The rollout is the ceiling.** If generation is 50% of training time, no
  pruning rate can exceed 2× overall, however aggressive.
- **CPPO pays off most where the update stage is fattest**: large `G`, long
  completions, a KL penalty adding a reference forward pass, or `μ > 1` inner
  iterations. The paper's 8.32× on GSM8K implies `f ≈ 0.88`, consistent with
  its `β = 0.04` configuration where every completion also pays a
  reference-model forward.
- **Our configuration is deliberately less favourable to CPPO.** With `β = 0`,
  `μ = 1` and a vLLM rollout, the update stage is one forward plus one
  backward, so `f` is smaller. The honest reading: the configuration that
  flatters CPPO's *relative* numbers is not the one that is best in absolute
  terms.
- **The rollout's *work* is irreducible by this method — but its *time* is
  not.** Nothing in the rollout depends on advantages, which do not exist until
  sampling has finished, so pruning cannot generate fewer tokens. But dynamic
  allocation enlarges the generation batch by `m`, and vLLM's continuous
  batching converts that into throughput: §8.2 measures the rollout running
  **12-20% faster** at the same token count. This is why the measured speedup
  *exceeds* the ceiling above — the "invariant" term is not invariant, and the
  formula is therefore a lower bound rather than an upper one. Cutting rollout
  work itself still needs an orthogonal technique: speculative or truncated
  sampling, early termination of degenerate groups, or cross-step reuse.

![End-to-end speedup ceiling as a function of the update stage share](report/figures/amdahl_ceiling.png)

Because `f` is measured directly by the profiling mixin — both stages timed by
the same code in both runs, and additive by construction — this equation is a
falsifiable prediction. **§8.2 checks it, and it fails in CPPO's favour.** That
is the most useful thing the model did: not confirm a number, but be precise
enough that reality could contradict it and say why.

### 7.2 Why accuracy survives

`cppo/retained_signal_fraction` reports the share of the batch's total `|A|`
mass that survives pruning. Because group-standardised rewards from a two-part
reward function are strongly bimodal, that share stays close to 1 even at
`P = 0.75`: the discarded completions are *numerous but individually almost
weightless*.

The paper's metric ablation confirms it is the *criterion*, not merely the act
of subsetting, that matters (GSM8K, Qwen2.5-1.5B-Instruct, `G = 16`, `P = 0.5`):

| Pruning metric | Accuracy |
|---|---:|
| keep largest \|A\| (**CPPO**) | **77.67%** |
| random subset | 76.98% |
| keep largest raw (signed) A | 76.83% |
| keep smallest \|A\| | 74.23% |

Absolute advantage beats signed advantage, confirming that large-*negative*
completions ("do not do this") are as informative as large-positive ones.

**There is a floor.** At `P = 0.9375` (`k = 1`) the paper's accuracy drops on
both GSM8K and MATH: a single retained completion makes the surviving gradient
a one-sample estimate, and genuinely useful completions start being discarded
alongside the weightless ones. **`P ∈ [0.5, 0.875]` is the usable band.**

### 7.3 Memory

Completion pruning does **not** reduce peak memory on its own — and with
dynamic allocation it is *designed* not to, since the whole point of refilling
the freed slots is to keep update-stage tensors at the width the device can
hold. CPPO buys **throughput at constant memory**, not a smaller footprint.

The memory saving in this pipeline comes from a different decision: `β = 0`
removes the reference model entirely. The `no_allocation` ablation is the one
configuration where pruning *does* lower peak memory, at the cost of leaving
the device under-occupied.

### 7.4 Reference numbers from the paper

For calibration (Qwen2.5-1.5B-Instruct on GSM8K, `G = 16`; not reproduced here):

| Method | P | k | Accuracy | Training time | Speedup |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct (untrained) | – | – | 55.72% | – | – |
| GRPO | 0.00% | 16 | 77.05% | 23,393 s | 1.00× |
| CPPO | 50.00% | 8 | 77.67% | 12,930 s | 1.81× |
| CPPO | 75.00% | 4 | 78.81% | 7,159 s | 3.27× |
| CPPO | 87.50% | 2 | **80.41%** | 4,781 s | 4.89× |
| CPPO | 93.75% | 1 | 78.20% | 2,813 s | **8.32×** |

And the component ablation (Qwen2.5-7B-Instruct on MATH, `G = 16`, `P = 0.5`),
which isolates the two mechanisms:

| Method | Accuracy | Time | Speedup |
|---|---:|---:|---:|
| GRPO | 75.20% | 33,902 s | 1.00× |
| + completion pruning | 75.80% | 27,547 s | 1.23× |
| + dynamic allocation | 75.20% | 20,550 s | **1.65×** |

Note how much of the benefit comes from *allocation*, not pruning alone — which
is precisely the Amdahl argument: pruning alone shrinks the update stage but
leaves the device idle, whereas allocation converts the freed capacity into
fewer, fatter optimiser steps.

---

## 8. Results

All numbers below were measured on **1× NVIDIA A100-SXM4-80GB**. Every run
consumed the same 8,192 DAPO-Math problems in the same order, with identical
data, rewards, seeds and instrumentation. Raw artefacts are in [`results/`](results/) and the tables are regenerated from
them by `python -m cppo.report`. The `eval.json` files carry aggregate metrics,
task configs and versions; lm_eval's per-document dumps (~46 MB each) are
stripped, and `cppo.evaluate` now passes `log_samples=False` so they are not
produced again.

### 8.1 Training performance

| Method | P | k | m | Steps | Wall clock | Rollout | Update | Peak mem | Questions/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grpo-baseline | 0% | 8 | 1× | 510 | 11401 s | 5331 s | 5260 s | 48.05 GiB | 0.716 | 1.00× |
| cppo-p50 | 50% | 4 | 2× | 255 | 7559 s | 4373 s | 2752 s | 47.78 GiB | 1.080 | **1.51×** |
| cppo-p75 | 75% | 2 | 4× | 127 | 5917 s | 4253 s | 1428 s | 47.90 GiB | 1.374 | **1.92×** |
| cppo-p875 | 87.5% | 1 | 8× | 63 | 5545 s | 4672 s | 740 s | 47.76 GiB | 1.454 | **2.03×** |
| cppo-p75-no-allocation | 75% | 2 | 1× | 510 | 6465 s | 4787 s | 1439 s | **32.96 GiB** | 1.262 | 1.76× |

**CPPO cut wall-clock training time by half at P = 0.875** — 3.17 h down to
1.54 h for the same 8,192 problems.

### 8.2 The cost model was right about the update stage and wrong about the rollout

§7.1 predicted `S(P) = 1/((1−f) + f(1−P))`. From the baseline,
`f = 5260/11401 = 0.461`, which puts the ceiling at `1/(1−f) = 1.86×`. Against
that prediction:

| P | Predicted | **Measured** | Update stage | Rollout stage |
|---:|---:|---:|---:|---:|
| 0.50 | 1.30× | **1.51×** | 0.523× (theory 0.500) | 0.82× (theory 1.00) |
| 0.75 | 1.53× | **1.93×** | 0.271× (theory 0.250) | 0.80× (theory 1.00) |
| 0.875 | 1.68× | **2.06×** | 0.141× (theory 0.125) | 0.88× (theory 1.00) |

Two things happened, and only one of them was predicted.

**The update stage scaled almost exactly as modelled.** Measured 0.523 / 0.271
/ 0.141 against a theoretical `k/G` of 0.500 / 0.250 / 0.125. The small excess
is the batch-independent cost the micro-benchmark isolated — on A100 the fit is

```
T(c) = 0.0447 s + 0.0209 s × c        R² = 0.99978
```

with the intercept at **6.3%** of the baseline step, giving an update-stage
ceiling of 15.8×.

**The rollout was supposed to be invariant. It was not.** Generation dropped to
**0.80–0.88×** of the baseline. Dynamic allocation enlarges the generation
batch by `m`, which gives vLLM more sequences to batch per decoding step — and
continuous batching converts that directly into throughput. §5.6 predicted this
qualitatively; here it is quantitatively.

The consequence is that **the measured 2.06× at P = 0.875 exceeds the Amdahl
ceiling of 1.86×**. That is not a paradox — it falsifies the model's premise.
Amdahl's law bounds what you get by accelerating *one* part while the rest is
held fixed. CPPO's allocation strategy does not hold the rollout fixed, so the
bound does not apply as stated. The correct reading: **the cost model is a
lower bound on CPPO's benefit, not an upper one**, precisely because allocation
touches both stages.

### 8.3 Memory: the ablation proves the claim exactly

```
with allocation:     48.05 → 47.78 / 47.90 / 47.76 GiB   (unchanged)
without allocation:  48.05 → 32.96 GiB                   (−31%)
```

§7.3 argued that CPPO buys throughput *at constant memory*, and that only the
no-allocation ablation lowers the peak. Both halves confirmed: the three
allocated runs sit within 0.3 GiB of the baseline, and removing allocation cuts
31% of peak memory — at the cost of leaving three quarters of each micro-batch
empty (still 1.76×, but 1.92× was available).

### 8.4 Accuracy: the result that contradicts expectations

| Model | Steps | gsm8k | Δ | minerva_math | Δ | aime24 |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-0.6B (untrained)** | — | 36.85 | — | **46.58** | — | 3.33 |
| grpo-baseline | 510 | **39.12** | +2.27 | 44.56 | **−2.02** | 0.00 |
| cppo-p50 | 255 | 38.51 | +1.67 | 45.48 | −1.10 | 3.33 |
| cppo-p75 | 127 | 38.13 | +1.29 | 45.80 | −0.78 | 0.00 |
| cppo-p875 | 63 | 38.13 | +1.29 | 46.12 | −0.46 | 3.33 |
| cppo-p75-no-allocation | 510 | **39.35** | +2.50 | 45.12 | −1.46 | 0.00 |

**One epoch of GRPO improved gsm8k by ~2 points and degraded minerva_math by
~2 points.** The untrained model is the best minerva_math scorer in the table.
This is not a CPPO result — the GRPO baseline shows it too, and more strongly
than any CPPO run.

The training logs explain it. Reward rose from 0.61 to 0.74 over the epoch, but
the rise was **entirely in the format component** (0.456 → 0.576) while
accuracy stayed flat (0.158 → 0.181). Mean completion length fell from 799 to
755 tokens. The policy learned to emit exactly one `\boxed{}` and to stop
sooner. On gsm8k, where the flexible-extract filter takes the last number, that
helps. On minerva_math, where 4-shot prompting already produced parseable
answers, shorter reasoning on competition-level problems costs more than
cleaner formatting gains.

**Why CPPO degrades it less, and why that is mostly not about pruning.** The
minerva_math loss tracks the optimiser-step count almost perfectly:

```
510 steps → −2.02      255 → −1.10      127 → −0.78      63 → −0.46
```

CPPO at P = 0.875 takes eight times fewer steps for the same data, so it simply
moves the model less. Attributing this to pruning would be wrong, and the
monotone trend in `P` is confounded with the step count.

**The ablation disentangles it.** `cppo-p75-no-allocation` runs the **same 510
optimiser steps as the baseline** on the same data, but prunes to `k = 2`. At
matched step count:

| | minerva_math | gsm8k |
|---|---:|---:|
| grpo-baseline | 44.56 | 39.12 |
| cppo-p75-no-allocation | **45.12** | **39.35** |
| difference | **+0.56** | **+0.23** |

So pruning itself is **neutral to slightly positive** — directionally
consistent with the paper's claim, but the effect is under one point on a
single seed and should not be called significant. What is solid is the negative
result: **pruning 75% of completions did not hurt accuracy.** That is the
hypothesis CPPO actually needs, and it holds.

`aime24` is 0.00 or 3.33 throughout — zero or one problem out of thirty. Noise.

### 8.5 Update-stage micro-benchmark on A100

| P | k | Completions/step | Step time | Speedup (pruning only) | Speedup (with allocation) |
|---:|---:|---:|---:|---:|---:|
| 0% | 8 | 32 | 0.7083 s | 1.00× | 1.00× |
| 50% | 4 | 16 | 0.3816 s | 1.86× | 1.99× |
| 75% | 2 | 8 | 0.2120 s | 3.34× | 3.98× |
| 87.5% | 1 | 4 | 0.1287 s | 5.50× | 7.96× |

Reproduces the Apple-MPS run closely (3.23×/4.01× and 5.39×/8.02× there), on
different hardware and a 5.5× faster device — the ratio between pruning alone
and pruning with allocation is a property of the method, not of the machine.

### 8.6 Summary

| Question the task asks | Answer |
|---|---|
| **Training time** | 1.51× / 1.92× / **2.03×** at P = 0.50 / 0.75 / 0.875 |
| **Maximum memory** | unchanged with allocation; **−31%** without it |
| **Final model quality** | no degradation from pruning; at matched steps, +0.56 pp on minerva_math |
| **How accurate is CPPO?** | As accurate as GRPO. Discarding 75% of completions costs nothing measurable. |
| **How does it influence acceleration?** | Halves wall-clock time. The gain exceeds the naive Amdahl bound because allocation also accelerates the rollout. |

---

## 9. Reproducing the study

> **Running the full study on a rented GPU?** Follow
> **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — GPU choice, RunPod setup, expected
> per-stage timings, result collection and troubleshooting. Budget ~6-8 h and
> ~$10-15 on an A100 80GB.

| Script | What it does |
|---|---|
| `scripts/setup.sh` | create `.venv`, install everything (`--with-vllm` on CUDA; `CPPO_NO_VENV=1` to use a cloud image's existing torch) |
| `scripts/preflight.sh` | **run first on a GPU box**: validates the whole CUDA/vLLM path in ~5 min |
| `scripts/smoke_test.sh` | lint + types + tests + 2 real GRPO/CPPO steps on CPU |
| `scripts/train.sh <config>` | train one configuration (`NUM_GPUS>1` → `accelerate` + ZeRO-2) |
| `scripts/evaluate.sh <model> <out.json>` | `lm_eval` on gsm8k, minerva_math, aime24 |
| `scripts/benchmark_update_stage.sh [device]` | isolate the update stage from the rollout |
| `python benchmarks/plot_results.py` | render the report's figures from a benchmark JSON |
| `scripts/run_all.sh` | the full study: train + evaluate + benchmark + tables (resumable, logs to `results/logs/`) |
| `scripts/lint.sh` | pylint, mypy and pytest |

Configs use YAML inheritance (`extends: base.yaml`), so an experiment file only
states its differences:

```yaml
extends: base.yaml
run_name: cppo-p75
output_dir: outputs/cppo-p75
pruning_rate: 0.75
dynamic_allocation: true
```

Any field can be overridden on the command line:

```bash
python -m cppo.train --config configs/cppo_p75.yaml --max-samples 512 --seed 7
```

The LaTeX report lives in [`report/report.tex`](report/report.tex); the built
PDF is checked in at [`report/report.pdf`](report/report.pdf). Rebuild it with:

```bash
cd report && make        # latexmk; `make pdflatex` for a manual cycle
```

Figures are regenerated from the measured JSON so they cannot drift from the
text:

```bash
python benchmarks/plot_results.py \
  --benchmark results/update_stage_mps_qwen3_0.6b.json --outdir report/figures
```

---

## 10. Code quality

```bash
bash scripts/lint.sh
```

| Check | Status |
|---|---|
| `pylint src/cppo tests benchmarks` | **10.00/10**, zero messages |
| `mypy` (strict, 21 source files) | **no issues** |
| `pytest tests` | **97 passed** (93 unit + 4 end-to-end) |
| `shellcheck scripts/*.sh` | **clean** |

`mypy` runs in `strict` mode. Third-party packages that ship `py.typed` but
re-export lazily (`trl`, `transformers`, `accelerate`, `peft`) get
`implicit_reexport = true`; nothing in this package is exempted.
CI (`.github/workflows/ci.yml`) runs every check above on each push.

The test suite covers:

- **pruning primitives** — top-`k` correctness, sign agnosticism, deterministic
  tie-breaking, equal counts per group, degenerate groups, shape validation;
- **batch geometry** — that allocation restores the baseline micro-batch, that
  disabling it shrinks the micro-batch, and that indivisible layouts are
  rejected;
- **rewards** — nested `\boxed{}`, symbolic equivalence (`1/2` ≡ `0.5`),
  conversational format, malformed output;
- **configuration** — YAML `extends` chains and cycle detection, CLI-over-YAML
  precedence, rejection of unknown keys, and that every shipped config in
  `configs/` loads and matches its filename;
- **profiling** — that rollout and update times partition the training step
  additively, and that the memory figure records which backend produced it;
- **report rendering** — including a regression test for a bare `%` in LaTeX
  output, which would otherwise comment out the rest of the line;
- **evaluation** — the `lm_eval` argument strings for both backends, and the
  reduction of a harness payload to one headline metric per task;
- **four end-to-end tests** that build real TRL trainers on a tiny two-layer
  Qwen3 and assert that CPPO generates the full group but back-propagates
  through only the pruned subset.

---

## 11. Limitations

- **Pruning point.** Pruning happens after the rollout rather than before the
  auxiliary forward passes (§5.3). With the recommended `β = 0`, `μ = 1`
  configuration this is exact; otherwise the reported speedup understates CPPO.
- **Model scale.** Qwen3-0.6B is small enough that an optimiser step carries
  fixed overheads a 7B model would amortise, so `f` — and the achievable
  speedup — is smaller here than at scale.
- **`aime24` has 30 problems.** One problem is 3.3 points. Differences there
  are not significant without multiple seeds.
- **Single seed per configuration.** GRPO is noisy; read the accuracy column as
  "no regression" evidence rather than a precise ranking. The +0.56 pp from the
  step-matched ablation is well inside what a second seed could move.
- **Optimiser-step count is confounded with the pruning rate** in the main
  sweep: CPPO at P = 0.875 takes 8× fewer steps for the same data, so "CPPO
  preserved accuracy better" and "CPPO changed the model less" cannot be
  separated there. Only the `no_allocation` ablation, which matches the
  baseline's 510 steps, isolates pruning — and it is one configuration on one
  seed.
- **One epoch degraded minerva_math** for every configuration including the
  GRPO baseline (§8.4). The reward rose almost entirely through its format
  component, so the policy learned presentation rather than reasoning. A
  stronger study would need more epochs, a harder-to-game reward, or a model
  with real headroom on competition mathematics.
- **34% of completions were truncated** at the 1024-token cap and masked out of
  the loss, so the effective batch was ~66% of nominal. The setting is
  identical across all runs, so the comparison holds, but a 2048-token budget
  would have used the rollout more efficiently.
- **Gradient checkpointing is on throughout**, which inflates the update stage
  by roughly a third and so raises the measured `f`. This *flatters* CPPO — the
  same sweep without checkpointing would show a lower end-to-end speedup. It is
  held identical across runs so the comparison is fair, but the absolute number
  is specific to that choice. This is exactly why §7.1 states the ceiling in
  terms of a measured `f` rather than a constant.
- **Hardware provenance.** Each results table records the hardware it was
  produced on. Numbers are never extrapolated across devices.

---

## 12. References

1. Lin, Lin, Xie, Ji. *CPPO: Accelerating the Training of Group Relative Policy
   Optimization-Based Reasoning Models.* [arXiv:2503.22342](https://arxiv.org/abs/2503.22342), 2025.
2. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
   Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300), 2024.
3. DeepSeek-AI. *DeepSeek-R1.* [arXiv:2501.12948](https://arxiv.org/abs/2501.12948), 2025.
4. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.*
   [arXiv:2503.14476](https://arxiv.org/abs/2503.14476), 2025.
5. Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective.*
   [arXiv:2503.20783](https://arxiv.org/abs/2503.20783), 2025.
6. Hu et al. *Open-Reasoner-Zero.* [arXiv:2503.24290](https://arxiv.org/abs/2503.24290), 2025.
7. von Werra et al. *TRL: Transformer Reinforcement Learning.* https://github.com/huggingface/trl
8. Gao et al. *A framework for few-shot language model evaluation.*
   https://github.com/EleutherAI/lm-evaluation-harness
9. Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP, 2023.
10. Qwen Team. *Qwen3 Technical Report.* [arXiv:2505.09388](https://arxiv.org/abs/2505.09388), 2025.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
