# glasstrace
[![CI](https://github.com/manu-j3400/glasstrace/actions/workflows/ci.yml/badge.svg)](https://github.com/manu-j3400/glasstrace/actions/workflows/ci.yml)

> Per-layer latency and memory profiler for transformer inference.

`glasstrace` shows you where time actually goes inside your LLM. Decomposes inference cost by layer, operation, and inference phase (prefill vs decode).

## Why

When you call `model.generate()`, you get a number: total latency. That's not enough to make anything faster. `glasstrace` turns the black box into a measured picture: which layers are slow, where memory pressure lives, and what changes when you tweak batch size or sequence length.

## Install

```bash
pip install git+https://github.com/manu-j3400/glasstrace.git
```

PyPI release coming with v1.0.

## Usage

```python
import glasstrace
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

with glasstrace.profile(model) as p:
    out = model.generate(**inputs, max_new_tokens=50)

print(p.report())
```

## Status

**v0.1.0 — alpha.** Works on Qwen 2.5 0.5B and Llama 3.2 1B with CUDA. Tracks `nn.Linear` and `nn.LayerNorm` modules. Memory tracking, HTML reports, and broader model coverage planned for v0.2.

## Roadmap

- [x] v0.1 — Per-module CUDA timing, text-table report
- [ ] v0.2 — Prefill vs decode split, memory tracking, HTML report
- [ ] v0.3 — Multi-model tested coverage, CLI
- [ ] v0.4 — Comparative analyses across Llama, Qwen, Phi (blog post)
- [ ] v1.0 — PyPI release, docs, demo video

## License

MIT
