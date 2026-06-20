"""Command-line interface for glasstrace."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="glasstrace",
        description="Per-layer latency and memory profiler for transformer inference.",
    )
    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile a HuggingFace model and print a per-layer timing report.",
    )
    profile_parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name or local path (e.g. Qwen/Qwen2.5-0.5B)",
    )
    profile_parser.add_argument(
        "--prompt",
        default="The first ten prime numbers are:",
        help="Input prompt to run inference on.",
    )
    profile_parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Number of tokens to generate (default: 20).",
    )
    profile_parser.add_argument(
        "--device",
        default=None,
        help="Device to run on: cuda, cpu, mps. Auto-detected if not specified.",
    )
    profile_parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of modules to show per section (default: 20).",
    )
    profile_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warmup pass (not recommended on CUDA).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "profile":
        _run_profile(args)


def _run_profile(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import glasstrace

    # Resolve device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading {args.model} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    warmup_fn = None
    if not args.no_warmup:
        def warmup_fn():
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=5, do_sample=False)

    print("Profiling...")
    with glasstrace.profile(model, warmup=warmup_fn) as p:
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
            )

    print(p.report(top_n=args.top_n))


if __name__ == "__main__":
    main()
