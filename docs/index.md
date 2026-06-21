# glasstrace

> Per-layer latency and memory profiler for transformer inference.

[![PyPI](https://img.shields.io/pypi/v/glasstrace.svg)](https://pypi.org/project/glasstrace/)
[![CI](https://github.com/manu-j3400/glasstrace/actions/workflows/ci.yml/badge.svg)](https://github.com/manu-j3400/glasstrace/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
pip install glasstrace
```

## Quick start

```python
import glasstrace
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")

def warmup():
    model.generate(**inputs, max_new_tokens=5, do_sample=False)

with glasstrace.profile(model, warmup=warmup) as p:
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20, do_sample=False)

print(p.report())
p.save_html("report.html")
```

## CLI

```bash
glasstrace profile --model Qwen/Qwen2.5-0.5B --prompt "Hello" --max-tokens 20
```

## What it does

glasstrace registers forward hooks on every `nn.Linear` and `nn.LayerNorm`
in your model. On CUDA it uses `torch.cuda.Event` for accurate GPU timing.
It detects prefill vs decode from the sequence dimension of each layer's input,
and tracks KV-cache memory growth during decode.

The `warmup` parameter runs one forward pass before attaching hooks, paying
the one-time GPU cold-start cost so it doesn't distort measurements.

## API reference

### `glasstrace.profile(model, warmup=None)`

Context manager. Profile any `nn.Module`.

| Parameter | Type | Description |
|---|---|---|
| `model` | `nn.Module` | The model to instrument |
| `warmup` | `Callable \| None` | Zero-arg callable run once before profiling. Strongly recommended on CUDA. |

Returns a `ProfileResult` object.

### `ProfileResult.report(top_n=20)`

Returns a formatted two-section text report (prefill + decode), sorted by
total time. `top_n` controls how many modules appear per section.

### `ProfileResult.save_html(path="glasstrace_report.html")`

Generates a standalone HTML report with an interactive bar chart and
sortable table. Opens in any browser, no server required.

## Findings

Running glasstrace on 4 models on a T4 GPU revealed:

- Decode speed scales sub-linearly with parameter count
- KV-cache growth depends on architecture, not model size
- lm_head's share of decode shrinks as models get deeper
- Cold-start GPU overhead distorts naive profiles by up to 20x

## Links

- [GitHub](https://github.com/manu-j3400/glasstrace)
- [PyPI](https://pypi.org/project/glasstrace/)
- [Issues](https://github.com/manu-j3400/glasstrace/issues)
