# Runbook — running the full study on a rented GPU

This walks through reproducing the complete experiment (GRPO baseline, three
CPPO pruning rates, the no-allocation ablation, and `lm_eval` on all three
benchmarks) on a rented GPU. It is written for **RunPod**; the commands are the
same on Lambda, Vast.ai or any box with a CUDA GPU and SSH.

**Budget: roughly 6-8 hours and $10-15 on an A100 80GB.**

---

## 1. Pick the GPU

This workload is dominated by the autoregressive rollout, and decoding is bound
by **memory bandwidth**, not by FLOPS. Pick on bandwidth, not on price per hour
— the cheap cards are not cheaper overall, they just take longer and give a
dropped connection more chances to bite.

| GPU | VRAM | Bandwidth | Typical $/h | Est. total | Est. cost |
|---|---:|---:|---:|---:|---:|
| **H100 80GB** | 80 GB | 3.35 TB/s | ~$2.5-3 | ~5 h | ~$13-15 |
| **A100 80GB** | 80 GB | 2.04 TB/s | ~$1.2-1.9 | ~8 h | ~$10-15 |
| L40S / A6000 | 48 GB | ~0.86 TB/s | ~$0.8-1.1 | ~14 h | ~$11-15 |
| RTX 4090 | 24 GB | 1.01 TB/s | ~$0.35-0.5 | ~13 h | ~$5-7 |

**Recommended: A100 80GB** (or H100 if you would rather pay a little more to
halve the wall clock). Prices move — check before booking.

Qwen3-0.6B itself is small: full fine-tuning needs about 11 GB for weights,
gradients and AdamW state. The rest of the VRAM goes to the colocated vLLM KV
cache, which is what makes 48 GB workable and 80 GB comfortable.

## 2. Create the pod

On [runpod.io](https://runpod.io) → **Pods** → **Deploy**:

| Setting | Value |
|---|---|
| GPU | 1 × A100 80GB (Secure Cloud is steadier than Community for a long run) |
| Template | **RunPod PyTorch** (any CUDA 12.x PyTorch image) |
| Container disk | 20 GB |
| **Volume disk** | **60 GB**, mounted at `/workspace` |
| Ports | leave the SSH default |

60 GB is not padding: vLLM and torch wheels are several GB, the five
checkpoints are ~1.2 GB each, and the `lm_eval` datasets add a few more.
Running out of disk at hour six is the most annoying way to lose a run.

## 3. Set up, then validate before you spend anything

SSH in (RunPod shows the command under **Connect**), then:

```bash
cd /workspace
git clone https://github.com/saifutdinov2008/cppo-grpo-acceleration.git
cd cppo-grpo-acceleration

# The pod image already has a CUDA build of torch, so install into it rather
# than building a virtualenv that would re-download several GB.
CPPO_NO_VENV=1 bash scripts/setup.sh --with-vllm

# Optional but recommended: avoids Hub rate limits on the dataset downloads.
export HF_TOKEN=hf_...
```

Now the important step:

```bash
bash scripts/preflight.sh
```

This takes about five minutes and exercises the exact code path the long run
uses — CUDA, the vLLM rollout, both trainers, a checkpoint save/load round
trip, `lm_eval` through the vLLM backend, and table rendering. **Do not skip
it.** It turns "the sweep died three hours in" into "the sweep did not start."

If it prints `PREFLIGHT PASSED`, you are clear.

## 4. Run the sweep

Run it under `tmux` so a dropped SSH session does not kill it:

```bash
tmux new -s cppo
bash scripts/run_all.sh 2>&1 | tee results/run_all.log
```

Detach with `Ctrl-b` then `d`; reattach later with `tmux attach -t cppo`.

Expected progression on an A100 80GB:

| Stage | Expected |
|---|---:|
| `eval-base` | ~20 min |
| `train-grpo` | ~1.5 h |
| `train-cppo_p50` | ~1.2 h |
| `train-cppo_p75` | ~1.1 h |
| `train-cppo_p875` | ~1.1 h |
| `train-cppo_p75_no_allocation` | ~1.2 h |
| 5 × `eval-*` | ~1.5 h total |
| `benchmark` + tables | ~10 min |

`run_all.sh` is **resumable**: any stage whose artefact already exists is
skipped, and every stage tees to `results/logs/`. If something fails, fix it
and re-run the same command — completed runs are not repeated. To force a
stage to re-run, delete its artefact (e.g. `rm results/cppo_p75/profile.json`).

### Useful levers

```bash
# Just the headline comparison, if you are short on budget (~3 h):
CONFIGS="configs/grpo.yaml configs/cppo_p75.yaml" bash scripts/run_all.sh

# Cheaper evaluation: minerva_math is 5,000 problems and dominates eval time.
EVAL_LIMIT=500 bash scripts/run_all.sh

# On 80 GB you can turn gradient checkpointing off for ~25-30% faster updates.
# Change it for ALL runs or none — it shifts the rollout/update split and so
# changes the measured f that the Amdahl analysis depends on.
bash scripts/train.sh configs/grpo.yaml --no-gradient-checkpointing
```

## 5. Collect the results

Everything the report needs is small — a few MB. From your **laptop**:

```bash
# RunPod shows the host and port under Connect → SSH.
scp -P <port> -r root@<host>:/workspace/cppo-grpo-acceleration/results ./
scp -P <port> -r root@<host>:/workspace/cppo-grpo-acceleration/report/figures ./report/
scp -P <port> root@<host>:/workspace/cppo-grpo-acceleration/report/generated_tables.tex ./report/
```

Then locally:

```bash
cd report && make          # rebuild the PDF with the real numbers
git add -A && git commit -m "Add full-scale results" && git push
```

Pulling the artefacts down and committing from your laptop is preferable to
pushing from the pod, because it keeps your GitHub credentials off a rented
machine you are about to destroy.

## 6. Stop the pod

**Terminate** it when the transfer is done — a pod left running bills by the
minute whether or not it is doing anything. Stopping (rather than terminating)
still bills for the volume.

---

## Troubleshooting

**CUDA OOM during training.** Lower the vLLM share first, then the batch:

```bash
bash scripts/train.sh configs/grpo.yaml \
  --vllm-gpu-memory-utilization 0.2 --per-device-train-batch-size 8
```

If you change the batch size, change it for every run — the comparison assumes
an identical update-stage shape across configurations.

**CUDA OOM during evaluation.** `--gpu-memory-utilization 0.6` or
`--max-model-len 2048` on `cppo.evaluate`, or switch that step to
`BACKEND=hf`.

**vLLM fails to import or crashes on start.** vLLM pins specific torch
versions and is the most fragile dependency here. Reinstall it last so it wins
the version negotiation:

```bash
pip install -U vllm
python -c "import vllm; print(vllm.__version__)"
```

If it still fails, run without it — everything works with the Transformers
backend, just slower:

```bash
bash scripts/train.sh configs/grpo.yaml --no-use-vllm
```

**`bfloat16 is not supported`.** Pre-Ampere GPU (T4, V100). Use
`--torch-dtype float16`, and note that pure fp16 RL training can be unstable.

**Hub rate limits / 429s.** Set `HF_TOKEN`.

**Disk full.** `du -sh ~/.cache/huggingface outputs` — the checkpoints and the
Hub cache are the culprits. `--no-save-final-model` on runs you do not intend
to evaluate.

**`flash_attention_2` errors.** The configs default to `sdpa`, which needs no
build step. Only pass `--attn-implementation flash_attention_2` after
`pip install flash-attn --no-build-isolation` has succeeded.
