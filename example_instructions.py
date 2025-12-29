# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, TypedDict, Literal

import fire
import torch
from torch.nn import functional as F
from transformers import LlamaTokenizer
from tqdm import tqdm

from llama import Llama

class ExampleConfig(TypedDict):
    model: str
    tokenizer: str
    temperature: float
    top_p: float
    max_gen_len: int
    output_path: str
    n_examples: int
    seed: int

class ExampleInstruction(TypedDict):
    instruction: str
    input: str
    output: str

def load_examples(path: str) -> List[ExampleInstruction]:
    # Load examples from a JSON lines file
    import json
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def main(
    model: str,  # Path to model checkpoint directory
    tokenizer: str,  # Path to tokenizer files
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_gen_len: int = 512,
    output_path: str = "generated_examples.txt",
    n_examples: int = 100,
    seed: int = 42,
):
    """Generate example instructions using the model.

    Args:
        model: Path to the model checkpoint directory.
        tokenizer: Path to the tokenizer files.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_gen_len: Maximum generation length.
        output_path: Output file to write generated examples.
        n_examples: Number of examples to generate.
        seed: Random seed for reproducibility.
    """
    torch.manual_seed(seed)
    # Build the model
    generator = Llama.build(
        ckpt_dir=model,
        tokenizer_path=tokenizer,
        max_seq_len=2048,
        max_batch_size=1,
    )

    prompts = []
    # Construct prompts for instruction generation
    for i in range(n_examples):
        # Simple placeholder prompt; replace with actual instruction templates
        prompts.append(f"Generate an instruction {i}.")
    # Encode prompts
    prompt_tokens = [generator.tokenizer.encode(p, bos=True, eos=False) for p in prompts]
    # Generate completions
    generations, _ = generator.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=False,
        echo=False,
        stop_token=generator.tokenizer.eos_id,
    )
    # Decode and write to file
    with open(output_path, "w", encoding="utf-8") as f:
        for i, gen in enumerate(generations):
            text = generator.tokenizer.decode(gen)
            f.write(f"{i}\t{text}\n")
    print(f"Generated {n_examples} examples to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
