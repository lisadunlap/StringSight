"""
Build StringSight-ready JSONL files from raw TAU2 evaluation JSON outputs.

This script mirrors the pattern used in `playground.ipynb`:
- Load each raw TAU2 eval JSON (one dict per file)
- Convert it to a tidy dataframe via `stringsight.tau2_json_to_stringsight_df`
- Concatenate all files for a given domain
- Write `data/tau2/{domain}.jsonl` (records, one per line)

Input expectations:
    - A directory containing raw TAU2 JSON files, one evaluation per file.
      In this repo those live under `data/tau2/individual/`.

Output format:
    JSONL where each line is an object with (at minimum):
    - prompt: str
    - model: str
    - model_response: list[dict]  (conversation messages)
        - assistant tool calls (if present) are emitted as:
          [{"id": str, "name": str, "arguments": dict}, ...]
    - reward: float
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

import stringsight


def _iter_tau2_json_files(input_dir: Path) -> Iterable[Path]:
    """Yield TAU2 evaluation JSON files from a directory."""
    yield from sorted(input_dir.glob("*.json"))


def build_domain_jsonl(
    *,
    input_dir: Path,
    output_path: Path,
    domain_substring: str,
    prompt_fields: Sequence[str],
) -> None:
    """
    Build a single JSONL for one domain by concatenating per-file TAU2 conversions.

    Args:
        input_dir: Directory containing raw TAU2 evaluation JSON files (one per file).
        output_path: Output JSONL path.
        domain_substring: Substring used to select files (e.g., "airline", "telecom", "retail").
        prompt_fields: TAU2 scenario instruction keys to include in the `prompt` field.
            Example: ("known_info", "reason_for_call", "task_instructions")
    """
    dfs: list[pd.DataFrame] = []
    for p in _iter_tau2_json_files(input_dir):
        if domain_substring not in p.name:
            continue

        data: dict[str, Any] = json.loads(p.read_text())
        dfs.append(
            stringsight.tau2_json_to_stringsight_df(
                data,
                prompt_fields=tuple(prompt_fields),
            )
        )

    if len(dfs) == 0:
        raise ValueError(f"No files matched domain_substring={domain_substring!r} in {str(input_dir)!r}")

    df = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/tau2/individual"),
        help="Directory containing raw TAU2 evaluation JSON files (one per file).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/tau2"),
        help="Output directory for JSONL files.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["telecom", "airline", "retail"],
        help="Domain substrings to match in filenames.",
    )
    parser.add_argument(
        "--prompt_fields",
        nargs="+",
        default=["known_info", "reason_for_call", "task_instructions"],
        help="Scenario instruction keys to concatenate into the `prompt` column.",
    )
    args = parser.parse_args()

    for domain in args.domains:
        build_domain_jsonl(
            input_dir=args.input_dir,
            output_path=args.output_dir / f"{domain}.jsonl",
            domain_substring=domain,
            prompt_fields=args.prompt_fields,
        )


if __name__ == "__main__":
    main()



