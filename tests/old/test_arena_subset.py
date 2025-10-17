"""Load the first 10 rows of the arena dataset and make sure the DataFrame
has the expected columns.  This uses the helper in data_loader.py.

Note: The Hugging Face dataset download can be slow the first time.  For a
quick unit-test environment you might want to mock `datasets.load_dataset`.
"""

from types import SimpleNamespace

import pandas as pd
from stringsight.extractors.openai import OpenAIExtractor
from stringsight.postprocess.parser import LLMJsonParser
from stringsight.datasets import load_data
from stringsight.core.data_objects import PropertyDataset


def test_first_10_arena_rows():
    # Build a minimal args namespace expected by dataset loaders
    # args = SimpleNamespace(filter_english=True)

    # df, extract_content_fn, _ = load_data("arena", args)
    # print(f"Loaded {len(df)} rows")
    # print(df.columns)
    # model_options = ['llama-3-70b-instruct', 'gemini-1.5-pro-api-0514', 'claude-3-5-sonnet-20240620']
    # first10 = df[df.model_a.isin(model_options) & df.model_b.isin(model_options)].head(50)
    # print(f"Loaded {len(first10)} rows")

    # # Basic sanity checks
    # assert len(first10) == 50
    # required_cols = {"prompt", "conversation_a", "conversation_b", "model_a", "model_b", "model_a_response", "model_b_response"}
    # assert required_cols.issubset(first10.columns)

    # print("Loaded first 50 arena rows with columns:", list(first10.columns))

    first10 = pd.read_json("tests/outputs/arena_first50.jsonl", lines=True)
    required_cols = {"prompt", "conversation_a", "conversation_b", "model_a", "model_b", "model_a_response", "model_b_response"}
    assert required_cols.issubset(first10.columns)
    print(f"Loaded {len(first10)} rows")
    print(first10.columns)

    # get properties
    dataset = PropertyDataset.from_dataframe(first10, method="side_by_side")
    # save to json
    dataset.to_dataframe().to_json("tests/outputs/arena_first50.jsonl", orient="records", lines=True)
    print("..done loading properties")

    # ------------------------------------------------------------------
    # Extract, parse and save results WITH wandb (preferred path)
    # ------------------------------------------------------------------
    import wandb
    # import weave
    wandb.init(project="StringSight-test", name="test_run")

    extractor = OpenAIExtractor(verbose=True, use_wandb=True)
    parser = LLMJsonParser(verbose=True, use_wandb=True)

    dataset_after_extract = extractor(dataset)
    dataset_after_parse = parser(dataset_after_extract)

    # ------------------------------------------------------------
    # Save results for downstream clustering / analysis
    # ------------------------------------------------------------
    import pathlib
    output_dir = pathlib.Path("tests/outputs")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "arena_first50_properties.json"
    jsonl_path = output_dir / "arena_first50_properties_df.jsonl"

    dataset_after_parse.save(str(json_path), format="json")
    dataset_after_parse.to_dataframe().to_json(str(jsonl_path), orient="records", lines=True)

    print(f"Saved parsed dataset to {json_path} and {jsonl_path}")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    test_first_10_arena_rows()
    print("✅ Arena subset test passed!") 