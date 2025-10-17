"""Dataset loading utilities for StringSight.

This module migrates the helpers that previously lived in *data_loader.py*
into the package namespace so they can be imported as

    from stringsight.datasets import load_arena_data, load_webdev_data, load_data

The logic is copied verbatim (with minor import-path tweaks) to keep
back-compatibility with existing scripts.
"""

from __future__ import annotations

from typing import Tuple, Callable, Any, Dict
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Helpers to extract conversation content
# ---------------------------------------------------------------------------

def _extract_content_arena(conversation):
    """Extract content for standard arena data."""
    return conversation[0]["content"], conversation[1]["content"]

selected_keys = [
    "code",
    "commentary",
    "description",
    "file_path",
    "has_additional_dependencies",
    "additional_dependencies",
    "install_dependencies_command",
    "port",
    "template",
    "title",
]

def _extract_content_webdev(conversation):
    """Extract content for webdev arena data."""
    try:
      formatted_object = ""
      for key in selected_keys:
          formatted_object += f"## {key}\n{conversation[1]['object'][key]}\n\n"
      formatted_response = (
          f"## Text response\n{conversation[1]['content'][0]['text']}\n\n{formatted_object}## Logs\n{conversation[1]['result']}"
      )
      return conversation[0]["content"][0]["text"], formatted_response
    except Exception as e:
      return None, None

# ---------------------------------------------------------------------------
# Arena loaders
# ---------------------------------------------------------------------------

def load_arena_data(args) -> Tuple[pd.DataFrame, Callable, str]:
    """Load and preprocess the standard arena dataset."""
    print("Loading arena dataset…")
    dataset = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    df = dataset.to_pandas()
    df['question_id'] = df['id']
    print(f"Loaded {len(df)} battles from arena dataset")

    if getattr(args, "filter_english", False):
        df = df[df["language"] == "English"]
        print(f"After English filter: {len(df)} battles")

    # models = [
    #     "claude-3-5-sonnet-20240620",
    #     "gpt-4o-2024-05-13",
    #     "gemini-1.5-pro-api-0514",
    #     "llama-3-70b-instruct",
    #     "gemini-1.5-pro-exp-0801",
    #     "claude-3-opus-20240229",
    #     "llama-3.1-405b-instruct",
    #     "chatgpt-4o-latest",
    #     "gpt-4-turbo-2024-04-09",
    #     "deepseek-v2-api-0628",
    #     "gpt-4o-2024-08-06",
    # ]
    # df = df[df["model_a"].isin(models) & df["model_b"].isin(models)]
    # print(f"After model filter: {len(df)} battles")

    def parse_winner(row):
        if row["winner"] == "model_a":
            return row["model_a"]
        elif row["winner"] == "model_b":
            return row["model_b"]
        else:
            return row["winner"]
    
    df["winner"] = df.apply(parse_winner, axis=1)
    df["score"] = df.apply(lambda row: {"winner": row["winner"]}, axis=1)
    df["prompt"] = df.conversation_a.apply(lambda x: x[0]["content"])
    print(df.columns)

    df = df.dropna(subset=["conversation_a", "conversation_b"])
    print(f"After removing missing conversations: {len(df)} battles")

    # ------------------------------------------------------------------
    # Extract user prompt and both model responses into explicit columns
    # ------------------------------------------------------------------
    def _extract_responses(row):
        user_prompt, model_a_resp = _extract_content_arena(row["conversation_a"])
        _, model_b_resp = _extract_content_arena(row["conversation_b"])
        return pd.Series({
            "question_id": row["question_id"],
            "model_a_response": row['conversation_a'],
            "model_b_response": row['conversation_b'],
        })

    response_df = df.apply(_extract_responses, axis=1)
    df = df.merge(response_df, on=["question_id"], how="left")
   
    if "__index_level_0__" in df.columns:
        df = df.drop(columns="__index_level_0__")

    # Deduplicate now that prompt column definitely exists
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["prompt",])
    print(f"After removing duplicates: {before_dedup - len(df)} rows dropped, {len(df)} remain")

    return df, _extract_content_arena, "one_sided_system_prompt_no_examples"

def load_arena_data_single(args) -> Tuple[pd.DataFrame, Callable, str]:
    """Load and preprocess the standard arena dataset."""
    df, _, _ = load_arena_data(args)

    df_a = df.copy()
    df_a["model"] = df['model_a']
    df_a["model_response"] = df['model_a_response']
    df_b = df.copy()
    df_b["model"] = df['model_b']
    df_b["model_response"] = df['model_b_response']
    df = pd.concat([df_a, df_b])
    df = df.drop(columns=["model_a", "model_a_response", "model_b", "model_b_response", "score", "conversation_a", "conversation_b"])
    df = df.dropna(subset=["model", "model_response"])
    print(f"After removing missing model and model response: {len(df)} battles")
    print(df.columns)
    
    return df, _extract_content_arena, "single_model_system_prompt"
    

# ---------------------------------------------------------------------------
# Web-dev loaders
# ---------------------------------------------------------------------------

def load_webdev_data(args):
    """Load and preprocess the web-dev arena dataset."""
    print("Loading webdev arena dataset…")
    dataset = load_dataset("lmarena-ai/webdev-arena-preference-10k", split="test")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} battles from webdev arena dataset")

    if getattr(args, "exclude_ties", False):
        df = df[~df["winner"].str.contains("tie")]
        print(f"After excluding ties: {len(df)} battles")

    df = df.dropna(subset=["conversation_a", "conversation_b"])
    df["prompt"] = df.conversation_a.apply(lambda x: x[0]["content"][0]["text"])
    print(f"After extracting prompt: {len(df)} battles")

    def parse_winner(row):
        if row["winner"] == "model_a":
            return row["model_a"]
        elif row["winner"] == "model_b":
            return row["model_b"]
        else:
            return row["winner"]
    
    df["winner"] = df.apply(parse_winner, axis=1)
    df["score"] = df.apply(lambda row: {"winner": row["winner"]}, axis=1)

    # Extract user prompt and both model responses
    def _extract_responses_webdev(row):
        user_prompt, model_a_resp = _extract_content_webdev(row["conversation_a"])
        _, model_b_resp = _extract_content_webdev(row["conversation_b"])
        return pd.Series({
            "question_id": row["question_id"],
            "model_a_response": [{
                "role": "user", "content": user_prompt
            },
            {
                "role": "assistant", "content": model_a_resp
            }],
            "model_b_response": [{
                "role": "user", "content": user_prompt
            }, {
                "role": "assistant", "content": model_b_resp
            }],
        })

    response_df = df.apply(_extract_responses_webdev, axis=1)
    print(response_df.columns)
    print(df.columns)
    df = df.merge(response_df, on=["question_id"], how="left")
    print("\nnew\n", df.columns)

    if "__index_level_0__" in df.columns:
      df = df.drop(columns="__index_level_0__")

    df = df.dropna(subset=["prompt", "model_a", "model_b"])

    # Deduplicate now that prompt column definitely exists
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["prompt", "model_a", "model_b"])
    print(f"After removing duplicates: {before_dedup - len(df)} rows dropped, {len(df)} remain")

    model_counts = df.model_a.value_counts()
    # remove any models with less than 100 responses
    remove_models = model_counts[model_counts < 100].index.tolist()
    df = df[~(df.model_a.isin(remove_models) | df.model_b.isin(remove_models))]
    print(f"After removing models with less than 100 responses: {len(df)} battles")
    print(df.model_a.value_counts())

    return df, _extract_content_webdev, "webdev_system_prompt_no_examples"

def load_webdev_data_single(args):
    df, _, _ = load_webdev_data(args)
    df_a = df.copy()
    df_a["model"] = df['model_a']
    df_a["model_response"] = df['model_a_response']
    df_b = df.copy()
    df_b["model"] = df['model_b']
    df_b["model_response"] = df['model_b_response']
    df = pd.concat([df_a, df_b])
    df = df.drop(columns=["model_a", "model_a_response", "model_b", "model_b_response", "score", "conversation_a", "conversation_b"])
    df = df.dropna(subset=["model", "model_response"])
    print(f"After removing missing model and model response: {len(df)} battles")
    print(df.columns)
    return df, _extract_content_webdev, "single_model_system_prompt"

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="webdev")
    parser.add_argument("--output_dir", type=str, default="data/arena/webdev.jsonl")
    parser.add_argument("--single_model", action="store_true")
    parser.add_argument("--filter_english", action="store_true")
    parser.add_argument("--exclude_ties", action="store_true")
    args = parser.parse_args()

    if args.dataset == "webdev":
        if args.single_model:
            df, _, _ = load_webdev_data_single(args)
        else:
            df, _, _ = load_webdev_data(args)
    elif args.dataset == "arena":
        df, _, _ = load_arena_data(args)
    elif args.dataset == "arena_single":
        df, _, _ = load_arena_data_single(args)

    # Drop conversation_a and conversation_b if they exist (content is already in model_response_a/b)
    columns_to_drop = [col for col in ["conversation_a", "conversation_b"] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        
    print(f"Saving {len(df)} rows to {args.output_dir}")
    df.to_json(args.output_dir, orient="records", lines=True)