"""Smallest possible example: profile a model's forward pass."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import glasstrace

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)

    print("Profiling forward pass...")
    with glasstrace.profile(model) as p:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20, do_sample=False)

    print(p.report())


if __name__ == "__main__":
    main()
