### Training performance

| Method | P | k | m | Steps | Questions | Wall clock (s) | Rollout (s) | Update (s) | Peak mem (GiB) | Questions/s | Throughput gain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke-grpo | 0.00% | 4 | 1x | 2 | 2 | 553.0 | 21.1 | 526.1 | 7.47 | 0.0036 | 1.00x |
| smoke-cppo | 50.00% | 2 | 2x | 2 | 4 | 550.9 | 28.7 | 516.4 | 6.99 | 0.0073 | 2.01x |

### Downstream accuracy (lm-evaluation-harness, % exact match)

_No evaluation results available._

### Update-stage micro-benchmark

| P | k | Completions/step | Tokens/step | Step time (s) | Speedup (pruning only) | Speedup (with allocation) |
|---|---|---|---|---|---|---|
| 0.00% | 8 | 8 | 1024 | 4.4768 +/- 0.0016 | 1.00x | 1.00x |
| 12.50% | 7 | 7 | 896 | 3.9283 +/- 0.0027 | 1.14x | 1.13x |
| 25.00% | 6 | 6 | 768 | 3.4326 +/- 0.0017 | 1.30x | 1.30x |
| 37.50% | 5 | 5 | 640 | 2.9138 +/- 0.0011 | 1.54x | 1.53x |
| 50.00% | 4 | 4 | 512 | 2.4255 +/- 0.0019 | 1.85x | 2.00x |
| 62.50% | 3 | 3 | 384 | 1.8822 +/- 0.0031 | 2.38x | 2.61x |
| 75.00% | 2 | 2 | 256 | 1.3869 +/- 0.0060 | 3.23x | 4.01x |
| 87.50% | 1 | 1 | 128 | 0.8302 +/- 0.0019 | 5.39x | 8.02x |
