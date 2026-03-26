"""Synthetic dataset generator for strict JSON/Python code generation tasks.

Generates prompt-instruction pairs at three difficulty levels (simple, medium, hard)
for both JSON and Python tasks. Used as the training/eval dataset for GRPO alignment.

Usage:
    python -m src.datasets.synthetic_dataset --output data/synthetic --num_samples 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

# ---------------------------------------------------------------------------
# JSON prompt templates
# ---------------------------------------------------------------------------

_JSON_SIMPLE: list[dict] = [
    {
        "instruction": (
            "Generate a valid JSON object with the following keys: "
            '"{k1}" (string), "{k2}" (integer), "{k3}" (boolean).'
        ),
        "params": lambda: {
            "k1": random.choice(["name", "title", "label", "city", "color"]),
            "k2": random.choice(["age", "count", "score", "year", "quantity"]),
            "k3": random.choice(["active", "verified", "enabled", "visible", "published"]),
        },
    },
    {
        "instruction": ("Generate a JSON array containing exactly {n} strings representing {topic}."),
        "params": lambda: {
            "n": random.randint(3, 7),
            "topic": random.choice(
                [
                    "fruit names",
                    "country names",
                    "programming languages",
                    "animal species",
                    "planet names",
                ]
            ),
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a key "{k1}" (string) and a key ' '"{k2}" which is an array of {n} integers.'
        ),
        "params": lambda: {
            "k1": random.choice(["id", "name", "label"]),
            "k2": random.choice(["values", "scores", "data", "items"]),
            "n": random.randint(3, 6),
        },
    },
]

_JSON_MEDIUM: list[dict] = [
    {
        "instruction": (
            "Generate a JSON array of {n} objects, each with keys "
            '"{k1}" (string), "{k2}" (integer), and a nested object '
            '"{k3}" containing "{nk1}" (string) and "{nk2}" (string).'
        ),
        "params": lambda: {
            "n": random.randint(2, 4),
            "k1": random.choice(["name", "title", "username"]),
            "k2": random.choice(["age", "id", "score"]),
            "k3": random.choice(["address", "contact", "location"]),
            "nk1": random.choice(["street", "city", "email"]),
            "nk2": random.choice(["zip", "country", "phone"]),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} with at least {n} fields, "
            "including one nested object and one array field."
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "user profile",
                    "product listing",
                    "blog post",
                    "movie record",
                    "employee record",
                ]
            ),
            "n": random.randint(5, 8),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a configuration file for a {app_type} "
            "with keys for {k1}, {k2}, and a nested {k3} section."
        ),
        "params": lambda: {
            "app_type": random.choice(
                [
                    "web server",
                    "database",
                    "logging service",
                    "cache layer",
                ]
            ),
            "k1": random.choice(["host", "port", "name"]),
            "k2": random.choice(["timeout", "retries", "max_connections"]),
            "k3": random.choice(["auth", "ssl", "logging", "metrics"]),
        },
    },
]

_JSON_HARD: list[dict] = [
    {
        "instruction": (
            "Generate a JSON Schema (draft-07) that validates objects representing "
            "a {entity}. The schema must include required fields, type constraints, "
            "a nested object property, and an array property with item validation."
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "REST API error response",
                    "e-commerce order",
                    "weather forecast",
                    "user registration form",
                    "CI/CD pipeline definition",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a paginated API response for a "
            "list of {entity}. Include metadata fields (page, per_page, total, "
            "total_pages) and a results array with {n} items, each having at "
            "least {f} fields including one nested object."
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "users",
                    "products",
                    "articles",
                    "transactions",
                    "repositories",
                ]
            ),
            "n": random.randint(2, 4),
            "f": random.randint(4, 6),
        },
    },
    {
        "instruction": (
            "Generate a deeply nested JSON object (at least 4 levels of nesting) "
            "representing a {domain} hierarchy. Each level should have a "
            '"name" (string), "id" (integer), and "children" (array of sub-objects).'
        ),
        "params": lambda: {
            "domain": random.choice(
                [
                    "company organizational chart",
                    "file system directory tree",
                    "category taxonomy",
                    "geographical region breakdown",
                ]
            ),
        },
    },
]

# ---------------------------------------------------------------------------
# Python prompt templates
# ---------------------------------------------------------------------------

_PYTHON_SIMPLE: list[dict] = [
    {
        "instruction": ("Write a Python function called `{fname}` that takes {args} " "and returns {ret}."),
        "params": lambda: {
            "fname": random.choice(
                [
                    "factorial",
                    "fibonacci",
                    "is_prime",
                    "reverse_string",
                    "sum_list",
                ]
            ),
            "args": random.choice(
                [
                    "an integer n",
                    "a string s",
                    "a list of integers",
                ]
            ),
            "ret": random.choice(
                [
                    "the factorial of n",
                    "the n-th Fibonacci number",
                    "True if n is prime, False otherwise",
                    "the reversed string",
                    "the sum of the list",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Write a Python function called `{fname}` that takes a list of " "{elem_type} and returns {ret}."
        ),
        "params": lambda: {
            "fname": random.choice(
                [
                    "find_max",
                    "find_min",
                    "count_even",
                    "filter_positive",
                    "unique_elements",
                ]
            ),
            "elem_type": random.choice(["integers", "floats", "strings"]),
            "ret": random.choice(
                [
                    "the maximum element",
                    "the minimum element",
                    "the number of even numbers",
                    "only the positive values",
                    "a list with duplicates removed",
                ]
            ),
        },
    },
    {
        "instruction": ("Write a Python function `{fname}` that converts {input_desc} " "to {output_desc}."),
        "params": lambda: {
            "fname": random.choice(
                [
                    "celsius_to_fahrenheit",
                    "to_uppercase",
                    "flatten_list",
                    "words_to_sentence",
                    "int_to_binary",
                ]
            ),
            "input_desc": random.choice(
                [
                    "a temperature in Celsius",
                    "a string",
                    "a nested list",
                    "a list of words",
                    "an integer",
                ]
            ),
            "output_desc": random.choice(
                [
                    "Fahrenheit",
                    "all uppercase",
                    "a flat list",
                    "a single sentence string",
                    "a binary string representation",
                ]
            ),
        },
    },
]

_PYTHON_MEDIUM: list[dict] = [
    {
        "instruction": ("Write a Python class called `{cname}` that implements a {ds} " "with methods: {methods}."),
        "params": lambda: {
            "cname": random.choice(["Stack", "Queue", "LinkedList", "MinHeap"]),
            "ds": random.choice(["stack", "queue", "singly linked list", "min-heap"]),
            "methods": random.choice(
                [
                    "push, pop, peek, is_empty, size",
                    "enqueue, dequeue, peek, is_empty, size",
                    "append, prepend, delete, search, to_list",
                    "insert, extract_min, peek, size, is_empty",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Write a Python function `{fname}` that {desc}. Handle edge cases "
            "and raise a ValueError for invalid input."
        ),
        "params": lambda: {
            "fname": random.choice(
                [
                    "merge_sorted_lists",
                    "binary_search",
                    "matrix_multiply",
                    "parse_csv_line",
                    "validate_email",
                ]
            ),
            "desc": random.choice(
                [
                    "merges two sorted lists into a single sorted list",
                    "performs binary search on a sorted list and returns the index",
                    "multiplies two 2D matrices represented as lists of lists",
                    "parses a CSV line respecting quoted fields into a list of strings",
                    "validates an email address format and returns True/False",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Write a Python class `{cname}` that {desc}. Include type hints " "for all methods and a __repr__ method."
        ),
        "params": lambda: {
            "cname": random.choice(
                [
                    "BankAccount",
                    "Matrix",
                    "Polynomial",
                    "Interval",
                    "Vector2D",
                ]
            ),
            "desc": random.choice(
                [
                    "models a bank account with deposit, withdraw, and balance methods",
                    "represents a 2D matrix supporting addition and multiplication",
                    "represents a polynomial supporting addition and evaluation at a point",
                    "represents a numeric interval supporting overlap checking and merge",
                    "represents a 2D vector supporting addition, dot product, and magnitude",
                ]
            ),
        },
    },
]

_PYTHON_HARD: list[dict] = [
    {
        "instruction": (
            "Write a Python decorator called `{dname}` that {desc}. "
            "The decorator should work with functions that have any signature."
        ),
        "params": lambda: {
            "dname": random.choice(
                [
                    "retry",
                    "cache_with_ttl",
                    "rate_limiter",
                    "log_calls",
                    "validate_types",
                ]
            ),
            "desc": random.choice(
                [
                    "retries the decorated function up to n times " "on exception with exponential backoff",
                    "caches function results with a time-to-live (TTL) in seconds",
                    "limits function calls to at most n per minute, raising RuntimeError if exceeded",
                    "logs function name, arguments, return value, and execution time",
                    "validates that arguments match the function's type annotations at runtime",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Write a Python context manager class called `{cname}` that {desc}. "
            "Implement both __enter__ and __exit__ methods with proper error handling."
        ),
        "params": lambda: {
            "cname": random.choice(
                [
                    "Timer",
                    "TempDirectory",
                    "DatabaseTransaction",
                    "FileLocker",
                    "ResourcePool",
                ]
            ),
            "desc": random.choice(
                [
                    "measures and stores the elapsed time of the code block",
                    "creates a temporary directory, yields its path, and cleans it up on exit",
                    "wraps a database transaction with automatic commit/rollback",
                    "acquires an exclusive file lock and releases it on exit",
                    "manages a pool of reusable resources with borrow/return semantics",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Write a Python class `{cname}` implementing the {pattern} design pattern. "
            "Include a concrete example with at least two {components}."
        ),
        "params": lambda: {
            "cname": random.choice(
                [
                    "EventBus",
                    "CommandHandler",
                    "PipelineBuilder",
                    "StateMachine",
                    "PluginRegistry",
                ]
            ),
            "pattern": random.choice(
                [
                    "Observer",
                    "Command",
                    "Builder",
                    "State",
                    "Strategy",
                ]
            ),
            "components": random.choice(
                [
                    "observers",
                    "commands",
                    "pipeline stages",
                    "states and transitions",
                    "strategy implementations",
                ]
            ),
        },
    },
]

# ---------------------------------------------------------------------------
# All pools with metadata
# ---------------------------------------------------------------------------

TASK_POOLS = {
    "json_simple": {"templates": _JSON_SIMPLE, "task_type": "json", "difficulty": "simple"},
    "json_medium": {"templates": _JSON_MEDIUM, "task_type": "json", "difficulty": "medium"},
    "json_hard": {"templates": _JSON_HARD, "task_type": "json", "difficulty": "hard"},
    "python_simple": {"templates": _PYTHON_SIMPLE, "task_type": "python", "difficulty": "simple"},
    "python_medium": {"templates": _PYTHON_MEDIUM, "task_type": "python", "difficulty": "medium"},
    "python_hard": {"templates": _PYTHON_HARD, "task_type": "python", "difficulty": "hard"},
}

# Difficulty distribution weights: more simple/medium, fewer hard
DIFFICULTY_WEIGHTS = {"simple": 0.40, "medium": 0.35, "hard": 0.25}
TYPE_WEIGHTS = {"json": 0.50, "python": 0.50}


def _build_system_prompt(task_type: str) -> str:
    """Build the system prompt instructing the model to output in a specific format."""
    if task_type == "json":
        return (
            "You are a helpful assistant that generates valid JSON. "
            "Respond ONLY with a JSON code block. Do not include any explanation "
            "before or after the JSON. Wrap your output in ```json and ``` markers."
        )
    return (
        "You are a helpful assistant that generates valid Python code. "
        "Respond ONLY with a Python code block. Do not include any explanation "
        "before or after the code. Wrap your output in ```python and ``` markers."
    )


def generate_sample(rng: random.Random | None = None) -> dict:
    """Generate a single prompt sample with metadata."""
    if rng is None:
        rng = random.Random()

    # Pick task type, then difficulty
    task_type = rng.choices(list(TYPE_WEIGHTS.keys()), weights=list(TYPE_WEIGHTS.values()), k=1)[0]
    difficulty = rng.choices(list(DIFFICULTY_WEIGHTS.keys()), weights=list(DIFFICULTY_WEIGHTS.values()), k=1)[0]

    pool_key = f"{task_type}_{difficulty}"
    pool = TASK_POOLS[pool_key]
    template = rng.choice(pool["templates"])

    params = template["params"]()
    instruction = template["instruction"].format(**params)
    system_prompt = _build_system_prompt(task_type)

    return {
        "system_prompt": system_prompt,
        "prompt": instruction,
        "task_type": task_type,
        "difficulty": difficulty,
    }


def generate_dataset(
    num_samples: int = 5000,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> DatasetDict:
    """Generate the full synthetic dataset as a HuggingFace DatasetDict."""
    rng = random.Random(seed)
    samples = [generate_sample(rng) for _ in range(num_samples)]

    # Deterministic split
    rng.shuffle(samples)
    split_idx = int(len(samples) * (1 - test_ratio))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    def _to_columnar(rows: list[dict]) -> dict:
        return {k: [r[k] for r in rows] for k in rows[0]}

    return DatasetDict(
        {
            "train": Dataset.from_dict(_to_columnar(train_samples)),
            "test": Dataset.from_dict(_to_columnar(test_samples)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic JSON/Python prompt dataset")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5000, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction for test split")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples (seed={args.seed})...")
    ds = generate_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    # Also save a human-readable preview
    preview_path = out_path / "preview.json"
    preview = [ds["train"][i] for i in range(min(10, len(ds["train"])))]
    preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset saved to {out_path}")
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  Test:  {len(ds['test'])} samples")

    # Print distribution stats
    for split_name in ["train", "test"]:
        split = ds[split_name]
        types = split["task_type"]
        diffs = split["difficulty"]
        print(f"\n  {split_name} distribution:")
        for tt in ["json", "python"]:
            for dd in ["simple", "medium", "hard"]:
                count = sum(1 for t, d in zip(types, diffs) if t == tt and d == dd)
                print(f"    {tt}/{dd}: {count}")


if __name__ == "__main__":
    main()
