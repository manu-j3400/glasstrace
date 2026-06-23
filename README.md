# glasstrace

[![CI](https://github.com/manu-j3400/glasstrace/actions/workflows/ci.yml/badge.svg)](https://github.com/manu-j3400/glasstrace/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/glasstrace.svg)](https://pypi.org/project/glasstrace/)
[![Python](https://img.shields.io/pypi/pyversions/glasstrace.svg)](https://pypi.org/project/glasstrace/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Documentation](https://manu-j3400.github.io/glasstrace/)** · [PyPI](https://pypi.org/project/glasstrace/) · [Issues](https://github.com/manu-j3400/glasstrace/issues)


## What is this?

When a language model answers you, it isn't one big step but more so hundreds of
small ones stacked on top of each other (the "layers"). Most tools hand you a
single number: *"that took 400 milliseconds."* Useful, but it's like a
restaurant bill that just says **$80** with no line items s0 you can't tell what
was expensive.

**glasstrace is the itemized bill.** It puts a stopwatch on every layer of the
model and shows you exactly where the time and memory went, so when something's
slow you know *which part* to fix instead of guessing.

It also splits the work into the two very different jobs a model does:

- **Prefill** — the model reads your whole prompt at once to "understand" it.
  Happens once.
- **Decode** — the model writes the answer one piece at a time. Happens once per
  chunk of output, and for longer answers this is usually where most of the
  time goes.

If you've ever wondered *why* a model feels slow and if the reason is a prompt,
the generation, or one greedy layer hogging everything then that's where glasstrace comes in.

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
p.save_html("report.html")  # interactive HTML report
```

> No NVIDIA GPU? Drop the `.to("cuda")` calls (or use `"mps"` on an Apple-silicon
> Mac) and you'll get the same per-layer breakdown. Timing just uses the wall
> clock instead of CUDA events. The `glasstrace` CLI auto-detects your device.

### Output:

<pre>
glasstrace report
  modules profiled: 169
  total events: 3380
  total measured time: 383.48 ms
  device: cuda
  kv-cache growth during decode: 0.2 MB

── prefill (1 pass, 69.7 ms total) ─────── top 5 of 169 modules ──────────
Module                            Type    Calls  Total ms  % of phase
model.layers.0.mlp.down_proj      Linear      1      1.78        2.6%
model.layers.0.mlp.gate_proj      Linear      1      1.74        2.5%
model.layers.0.mlp.up_proj        Linear      1      1.72        2.5%
lm_head                           Linear      1      1.43        2.1%
model.layers.0.self_attn.o_proj   Linear      1      0.98        1.4%

── decode (20 passes, 314.7 ms total, 15.7 ms/token avg) ─ top 5 modules ─
Module                            Type    Calls  Total ms  % of phase
lm_head                           Linear     20     37.48       11.9%
model.layers.0.mlp.down_proj      Linear     19      2.31        0.7%
model.layers.0.mlp.gate_proj      Linear     19      2.29        0.7%
model.layers.0.mlp.up_proj        Linear     19      2.28        0.7%
model.layers.0.self_attn.o_proj   Linear     19      1.95        0.6%
</pre>

### Interactive HTML report

Prefer to *see* it? `p.save_html("report.html")` writes a standalone page with
a flamegraph and sortable per-layer tables — open it in any browser, no server
needed. Same data as above, just clickable.

<!-- TODO: add a screenshot of report.html here, e.g.:
![glasstrace HTML report](figures/html_report.png)
Generate one by running the Quick start snippet, opening report.html, and
screenshotting it into figures/html_report.png. -->

## Reading the report

The header tells you the scope of the run:

| Line | What it means |
|------|---------------|
| **modules profiled** | How many layers got a stopwatch attached. |
| **total events** | How many times those layers fired across the whole run. |
| **total measured time** | Time spent *inside* the timed layers (not your Python overhead). |
| **device** | Where it ran — `cuda`, `mps`, or `cpu`. |
| **kv-cache growth during decode** | Extra memory the model held onto while writing the answer (see below). |

Then one table per phase (**prefill** and **decode**), each sorted by the
slowest layers. The columns:

- **Module** — the specific layer, named by its place in the model.
- **Type** — what kind of layer it is (`Linear`, `LayerNorm`, …).
- **Calls** — how many times it ran in that phase.
- **Total ms** — total time spent in that layer for the phase.
- **% of phase** — *the important one.* What fraction of that phase's time this
  one layer ate. Big numbers here are your bottlenecks.

> **KV-cache, in one sentence:** as the model writes each new word it keeps a
> running memory of everything so far so it doesn't re-read the whole prompt
> every time — that memory is the KV-cache, and "growth" is how fast it piles
> up. Fast growth = more memory pressure on long conversations.

## What the numbers are telling you

You don't need to read every row. Three habits get you most of the value:

1. **Look at the top of each table first.** The highest **% of phase** rows are
   where your time actually goes. If one layer is 15% and the rest are under 1%,
   you've found your bottleneck — optimize *that*, ignore the long tail.
2. **Compare prefill vs. decode totals.** Short prompt + long answer → decode
   dominates (you're generation-bound). Long prompt + short answer → prefill
   dominates (you're prompt-bound). This tells you which half is even worth
   tuning.
3. **Watch KV-cache growth if you run long contexts.** A model that grows its
   cache quickly will hit memory limits sooner — relevant when you're picking a
   model for long documents or long chats.

A common pattern you'll see: in small models `lm_head` (the final layer that
picks the next word) takes a big slice of decode, because the vocabulary is
large relative to everything else. That's normal, not a bug — it just shrinks as
models get deeper. The [benchmark](#benchmark) below shows this across four
models.

## Benchmark

4 models on a T4 GPU — fp16, 20 decode tokens, same prompt:

![glasstrace benchmark](figures/benchmark_graphic.png)

Three things stand out from the data:
- Decode speed scales sub-linearly with size. Qwen 3B is 6x larger than 0.5B but only 2.5x slower per token.
- KV-cache growth is about architecture, not parameter count. SmolLM2 1.7B grows its cache 6.8x faster than Qwen 1.5B at similar size.
- lm_head's share of decode shrinks as models get deeper because the body scales faster than the vocab projection.

## How it works

glasstrace registers forward hooks on every `nn.Linear` and
`nn.LayerNorm` in your model. On CUDA it uses `torch.cuda.Event`
for GPU timing — wall-clock time is meaningless for async GPU work.
Phase detection is based on the sequence dimension of each layer's
input: `seq_len > 1` is prefill, `seq_len == 1` is decode.

The warmup runs a forward pass before hooks are attached, paying
the one-time GPU initialization cost before measurement starts.

## Roadmap

- [x] v0.1 — per-module CUDA timing, text-table report
- [x] v0.1.1 — warmup phase, cold-start artifact fix
- [x] v0.2 — prefill/decode split, KV-cache tracking, PyPI release
- [x] v0.3 — CLI (`glasstrace profile --model Qwen/Qwen2.5-0.5B`)
- [x] v0.4 — HTML report with flamegraph
- [x] v1.0 — extended model coverage, docs site

## License

MIT
