"""
Benchmark creation script for evaluating behavior extraction methods.

This script generates synthetic datasets with known behavioral patterns by:
1. Generating diverse task-specific behaviors
2. Creating system prompts that induce these behaviors
3. Generating responses using modified models
4. Saving results for evaluation
"""

import os
import json
import yaml
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from stringsight.core.llm_utils import parallel_completions, single_completion


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark generation."""

    behavior_generation_model: str = "gemini-2.0-pro"
    response_generation_model: str = "gpt-4.1-mini"
    sample_size: Optional[int] = None
    num_behaviors: int = 10
    num_fewshot_examples: int = 10
    output_dir: str = "benchmark/results/"
    dataset_description_path: str = "input_dataset_descriptions/instructeval.yaml"
    random_seed: int = 42
    base_system_prompt: str = "You are a helpful assistant."
    enable_validation: bool = False
    validation_model: str = "gpt-4.1-mini"
    validation_max_workers: int = 10


@dataclass
class Behavior:
    """Represents a single behavioral pattern."""

    name: str
    description: str
    category: str
    full_system_prompt: str


def load_dataset_description(path: str) -> Dict:
    """Load dataset description from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str, prompt_column: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load dataset from JSONL file."""
    if dataset_path.endswith('.jsonl'):
        df = pd.read_json(dataset_path, lines=True)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")

    if prompt_column not in df.columns:
        raise ValueError(f"Prompt column '{prompt_column}' not found in dataset")

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    return df


def generate_behaviors(
    dataset_desc: Dict,
    df: pd.DataFrame,
    prompt_column: str,
    config: BenchmarkConfig
) -> List[Behavior]:
    """
    Generate diverse task-specific behaviors using LLM.

    Args:
        dataset_desc: Dataset description from YAML
        df: Dataset dataframe to sample few-shot examples from
        prompt_column: Name of column containing prompts
        config: Benchmark configuration

    Returns:
        List of Behavior objects
    """
    # Sample random few-shot examples from the dataset
    num_examples = min(config.num_fewshot_examples, len(df))
    fewshot_examples = df[prompt_column].sample(n=num_examples, random_state=config.random_seed).tolist()

    # Create prompt for behavior generation
    user_prompt = f"""You are helping create a benchmark for evaluating methods that automatically extract behavioral patterns from language model outputs.

Dataset: {dataset_desc['name']}
Description: {dataset_desc['description']}

Base system prompt that will be used: "{config.base_system_prompt}"

Example prompts from this dataset:
{chr(10).join(f"- {ex}" for ex in fewshot_examples)}

Generate exactly {config.num_behaviors} DISTINCT and EASILY DISTINGUISHABLE behavioral patterns that a language model might exhibit when responding to prompts from this specific dataset.

CRITICAL REQUIREMENTS:
1. MAXIMIZE DIVERSITY: The {config.num_behaviors} behaviors should be as different from each other as possible
   - Cover different aspects of the task (content, format, style, reasoning, errors, tool use, etc.)
   - Avoid generating multiple variations of the same underlying behavior
   - Think creatively about what could go wrong or what would be a notable behavior for the task. Not all of these behaviors should have a negative connotation.

2. OBSERVABLE: Clear, obvious differences that can be seen in the text of responses

3. CONSISTENT: Each behavior should manifest EVERY TIME it's applicable, not probabilistically
   - The behavior triggers ALWAYS when the relevant situation appears. For example, DO NOT use "occasionally", "sometimes", "50% of the time" in the system prompt

4. REALISTIC: Behaviors that actual language models might plausibly exhibit for the task

5. TASK-SPECIFIC: Analyze the example prompts above to understand what makes this task unique
   - What constraints, formats, or requirements appear in these prompts?
   - What kinds of mistakes or variations would be most relevant?
   - What stylistic choices matter for this specific task?

BEHAVIOR CATEGORIES (ensure diversity across these):
- Stylistic: How the model expresses itself, tone, verbosity, formality, structure
- Error: Mistakes in following instructions, misunderstanding requirements, violating constraints
- Reasoning: How the model approaches the task, what it prioritizes, interpretation patterns
- Safety: Overly cautious behaviors, refusals, disclaimers, hedging

For each behavior, provide:
- name: DESCRIPTIVE name in snake_case that clearly indicates the specific behavior
- description: at least 2-3 sentences explaining EXACTLY what this behavior does and how it appears in responses
- category: One of [stylistic, error, reasoning, safety]
- full_system_prompt: A COMPLETE system prompt that induces this behavior
  * Must START with the base system prompt: "{config.base_system_prompt}"
  * Then naturally integrate the behavioral instructions so the prompt flows well (not just appended awkwardly)
  * Be explicit and direct about the change in behavior
  * Must trigger CONSISTENTLY when applicable (not probabilistically)
  * Should sound natural and cohesive, blending the base prompt with new instructions
  * At least 2-3 sentences total
  * Focus on what to DO differently, not what a helpful assistant normally does

Return ONLY a JSON object with a "behaviors" key containing an array of {config.num_behaviors} behavior objects."""

    system_prompt = "You are an expert in language model behavior analysis and benchmark design. Always respond with valid JSON."

    response_text = single_completion(
        user_prompt,
        model=config.behavior_generation_model,
        system_prompt=system_prompt,
        temperature=1.0  # Higher temperature for more diverse behavior generation
    )

    # Clean up response text - remove markdown code blocks if present
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    try:
        result = json.loads(response_text)
        behaviors_list = result.get('behaviors', [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response_text[:500]}")
        raise

    behaviors = []
    for b in behaviors_list:
        behaviors.append(Behavior(
            name=b['name'],
            description=b['description'],
            category=b['category'],
            full_system_prompt=b['full_system_prompt']
        ))

    return behaviors


def generate_responses(
    df: pd.DataFrame,
    prompt_column: str,
    behavior: Behavior,
    config: BenchmarkConfig
) -> List[Dict]:
    """
    Generate responses for dataset prompts using behavior-specific system prompt.

    Args:
        df: Dataset dataframe
        prompt_column: Name of column containing prompts
        behavior: Behavior to induce
        config: Benchmark configuration

    Returns:
        List of response dictionaries
    """
    # Use the full system prompt directly (already includes base + behavioral instructions)
    system_prompt = behavior.full_system_prompt

    prompts = df[prompt_column].tolist()

    responses = parallel_completions(
        prompts,
        model=config.response_generation_model,
        system_prompt=system_prompt,
        temperature=0.7,
        max_workers=20,
        show_progress=True,
        progress_desc=f"Generating responses for {behavior.name}"
    )

    results = []
    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        result = {
            "prompt": prompt,
            "model_response": response,
            "model": behavior.name,
            "behavior_description": behavior.description,
            "category": behavior.category,
            "generation_model": config.response_generation_model,
            "base_system_prompt": config.base_system_prompt,
            "system_prompt": system_prompt,
            "metadata": {
                "row_index": idx
            }
        }
        results.append(result)

    return results


def format_behaviors_for_validation(behaviors: List[Behavior]) -> str:
    """Format behaviors for validation prompt."""
    formatted = []
    for i, b in enumerate(behaviors, 1):
        formatted.append(f"{i}. {b.name}")
        formatted.append(f"   Category: {b.category}")
        formatted.append(f"   Description: {b.description}")
        formatted.append("")
    return "\n".join(formatted)


def validate_single_response(
    prompt: str,
    response: str,
    all_behaviors: List[Behavior],
    model: str
) -> List[Dict]:
    """
    Validate a single response to identify which behaviors it exhibits.
    
    Args:
        prompt: User prompt
        response: Model response
        all_behaviors: All possible behaviors
        model: Model to use for validation
    
    Returns:
        List of detected behaviors with confidence and evidence
    """
    validation_prompt = f"""You are evaluating whether a model response exhibits specific behavioral patterns.

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

POSSIBLE BEHAVIORS:
{format_behaviors_for_validation(all_behaviors)}

TASK: Identify ALL behaviors that clearly apply to this response.
- A response can exhibit 0, 1, or multiple behaviors
- Only tag a behavior if there is clear, observable evidence in the response text
- If no specific behaviors apply, return "baseline_no_modification" only
- Be strict but thorough - look for clear manifestations of the behavior

For each behavior you identify, provide:
- behavior: the exact behavior name from the list above
- confidence: One of [high, medium, low]
  * high: Unambiguous, clear evidence directly observable in the response
  * medium: Evidence is present but requires some interpretation
  * low: Weak or circumstantial evidence, uncertain
- evidence: Specific quote or observation from the response (1-2 sentences max)

Return ONLY valid JSON in this exact format:
{{
  "behaviors": [
    {{"behavior": "behavior_name", "confidence": "high", "evidence": "specific evidence from response"}},
    ...
  ]
}}

If no specific behaviors apply, return:
{{
  "behaviors": [{{"behavior": "baseline_no_modification", "confidence": "high", "evidence": "No distinctive behavioral patterns observed"}}]
}}
"""

    system_prompt = "You are an expert at identifying behavioral patterns in language model outputs. Always respond with valid JSON."
    
    response_text = single_completion(
        validation_prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=0.0  # Deterministic for validation
    )
    
    # Clean up response
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    
    try:
        result = json.loads(response_text)
        behaviors = result.get('behaviors', [])
        # Rename 'behavior' key to 'model' for consistency with output schema
        for b in behaviors:
            if 'behavior' in b:
                b['model'] = b.pop('behavior')
        return behaviors
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse validation response: {e}")
        print(f"Response: {response_text[:200]}")
        return []


def validate_responses(
    results: List[Dict],
    all_behaviors: List[Behavior],
    config: BenchmarkConfig
) -> List[Dict]:
    """
    Validate responses with multi-label classification.
    
    Args:
        results: List of response dictionaries
        all_behaviors: All possible behaviors
        config: Benchmark configuration
    
    Returns:
        results with added validation fields:
        - validation_predictions: List[{model, confidence, evidence}]
        - validation_correct: bool (ground truth in predictions)
        - validation_strict_match: bool (ONLY ground truth predicted)
        - validation_extra_behaviors: List[str]

    """
    # Prepare validation inputs
    validation_inputs = []
    for result in results:
        validation_inputs.append({
            'prompt': result['prompt'],
            'model_response': result['model_response'],
            'index': result['metadata']['row_index']
        })
    
    # Run validation in parallel using ThreadPoolExecutor
    print(f"  Running validation on {len(validation_inputs)} responses using {config.validation_model}...")
    
    # Pre-allocate results to preserve order
    validated_results = [None] * len(results)
    
    def _validate_item(item: Dict) -> tuple:
        """Validate a single item and return (index, predictions)."""
        predictions = validate_single_response(
            item['prompt'],
            item['model_response'],
            all_behaviors,
            config.validation_model
        )
        return item['index'], predictions
    
    # Execute validations in parallel
    with ThreadPoolExecutor(max_workers=config.validation_max_workers) as executor:
        futures = {
            executor.submit(_validate_item, item): item['index']
            for item in validation_inputs
        }
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(validation_inputs), desc="Validating"):
            idx, predictions = future.result()
            
            # Find corresponding result
            result = results[idx]
            ground_truth = result['model']
            
            # Extract predicted behavior names
            predicted_behaviors = [p['model'] for p in predictions]
            
            # Calculate validation fields
            result['validation_predictions'] = predictions
            result['validation_correct'] = ground_truth in predicted_behaviors
            result['validation_strict_match'] = (
                len(predicted_behaviors) == 1 and predicted_behaviors[0] == ground_truth
            )
            result['validation_extra_behaviors'] = [
                m for m in predicted_behaviors if m != ground_truth
            ]
            
            validated_results[idx] = result
    
    return validated_results


def save_results(results: List[Dict], dataset_name: str, behavior_name: str, output_dir: str):
    """Save results to JSONL file."""
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{behavior_name}.jsonl"

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Saved {len(results)} results to {output_file}")


def load_all_results(output_dir: str, dataset_name: str) -> List[Dict]:
    """Load all results from a dataset."""
    results_path = Path(output_dir) / dataset_name
    all_results = []
    
    for behavior_file in results_path.glob("*.jsonl"):
        # Skip the combined file if it exists
        if behavior_file.name == "all_behaviors.jsonl":
            continue
        with open(behavior_file, 'r') as f:
            for line in f:
                all_results.append(json.loads(line))
    
    return all_results


def save_combined_results(all_results: List[Dict], output_dir: str, dataset_name: str):
    """Save all results to a single combined JSONL file.
    
    Args:
        all_results: List of all result dictionaries from all behaviors
        output_dir: Output directory path
        dataset_name: Name of the dataset
    """
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "all_behaviors.jsonl"
    
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nSaved {len(all_results)} total results to {output_file}")


def calculate_multilabel_metrics(results: List[Dict]) -> Dict:
    """
    Calculate multi-label classification metrics.
    
    Args:
        results: List of results with validation predictions
    
    Returns:
        Dict with overall and per-behavior metrics
    """
    # Get unique behaviors
    all_behaviors = sorted(set(r['model'] for r in results))
    
    # Overall metrics
    total = len(results)
    correct = sum(1 for r in results if r.get('validation_correct', False))
    strict_match = sum(1 for r in results if r.get('validation_strict_match', False))
    
    # Count behaviors per response
    avg_behaviors = sum(
        len(r.get('validation_predictions', [])) for r in results
    ) / total if total > 0 else 0
    
    # Per-behavior metrics
    per_behavior = {}
    for behavior in all_behaviors:
        # True positives: behavior is ground truth AND detected
        tp = sum(1 for r in results 
                if r['model'] == behavior and r.get('validation_correct', False))
        
        # False negatives: behavior is ground truth but NOT detected
        fn = sum(1 for r in results 
                if r['model'] == behavior and not r.get('validation_correct', False))
        
        # False positives: behavior is NOT ground truth but IS detected
        fp = sum(1 for r in results 
                if r['model'] != behavior and 
                behavior in [p.get('model', p.get('behavior')) for p in r.get('validation_predictions', [])])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_behavior[behavior] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3)
        }
    
    # Co-occurrence analysis
    co_occurrence = {}
    for r in results:
        predictions = [p.get('model', p.get('behavior')) for p in r.get('validation_predictions', [])]
        if len(predictions) > 1:
            # Sort to ensure consistent key
            pair_key = tuple(sorted(predictions))
            co_occurrence[pair_key] = co_occurrence.get(pair_key, 0) + 1
    
    # Convert to list of dicts
    co_occurrence_list = [
        {'behaviors': list(pair), 'count': count}
        for pair, count in sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Behavior purity (responses with only this behavior detected)
    behavior_purity = {}
    for behavior in all_behaviors:
        behavior_responses = [r for r in results if r['model'] == behavior]
        if behavior_responses:
            pure = sum(1 for r in behavior_responses if r.get('validation_strict_match', False))
            mixed = len(behavior_responses) - pure
            purity = pure / len(behavior_responses) if behavior_responses else 0
            
            behavior_purity[behavior] = {
                'pure_responses': pure,
                'mixed_responses': mixed,
                'purity': round(purity, 3)
            }
    
    return {
        'overall_metrics': {
            'total_responses': total,
            'recall': round(correct / total, 3) if total > 0 else 0,
            'strict_accuracy': round(strict_match / total, 3) if total > 0 else 0,
            'avg_behaviors_per_response': round(avg_behaviors, 2)
        },
        'per_behavior_metrics': per_behavior,
        'behavior_purity': behavior_purity,
        'co_occurrence': co_occurrence_list
    }


def save_validation_metrics(metrics: Dict, output_dir: str, dataset_name: str):
    """Save validation metrics to JSON file."""
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "validation_metrics.json"
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nValidation metrics saved to: {output_file}")


def create_benchmark(config: BenchmarkConfig):
    """Main function to create benchmark dataset."""

    print("=" * 80)
    print("BENCHMARK CREATION PIPELINE")
    print("=" * 80)

    # Warn if output directory exists
    output_path = Path(config.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        print(f"\n⚠️  WARNING: Output directory '{config.output_dir}' contains existing results.")
        print("    Old results will be mixed with new results in the viewer.")
        print(f"    Consider deleting old results: rm -rf {config.output_dir}/*")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Load dataset description
    print(f"\n[1/4] Loading dataset description from {config.dataset_description_path}")
    dataset_desc = load_dataset_description(config.dataset_description_path)
    dataset_name = dataset_desc['name']
    print(f"Dataset: {dataset_name}")
    print(f"Type: {dataset_desc['dataset_type']}")

    # Load dataset
    print(f"\n[2/4] Loading dataset from {dataset_desc['dataset_path']}")
    df = load_dataset(
        dataset_desc['dataset_path'],
        dataset_desc['prompt_column'],
        config.sample_size
    ).drop_duplicates(subset=[dataset_desc['prompt_column']])
    print(f"Loaded {len(df)} prompts")

    # Generate behaviors
    print(f"\n[3/4] Generating {config.num_behaviors} task-specific behaviors")
    print(f"Using model: {config.behavior_generation_model}")
    print(f"Using {min(config.num_fewshot_examples, len(df))} random few-shot examples from dataset")
    behaviors = generate_behaviors(dataset_desc, df, dataset_desc['prompt_column'], config)

    print(f"\nGenerated {len(behaviors)} behaviors:")
    for i, b in enumerate(behaviors, 1):
        print(f"  {i}. {b.name} ({b.category})")
        print(f"     {b.description}")

    # Add baseline behaviors (no modifications)
    print(f"\n[3.5/4] Adding baseline behaviors")
    baseline_behavior = Behavior(
        name="baseline_no_modification",
        description="Baseline responses using only the base system prompt with no behavioral modifications. This serves as a control to compare against modified behaviors.",
        category="baseline",
        full_system_prompt=config.base_system_prompt
    )
    behaviors.insert(0, baseline_behavior)
    print(f"  Added: baseline_no_modification")
    
    # Second baseline with paraphrased prompt (should be functionally identical)
    baseline_2 = Behavior(
        name="baseline_paraphrase",
        description="Baseline with paraphrased system prompt. Should be functionally identical to baseline_no_modification.",
        category="baseline",
        full_system_prompt="You are an assistant who is helpful."
    )
    behaviors.insert(1, baseline_2)
    print(f"  Added: baseline_paraphrase")

    # Generate responses for each behavior
    step_label = "[4/5]" if config.enable_validation else "[4/4]"
    print(f"\n{step_label} Generating responses for each behavior")
    print(f"Using model: {config.response_generation_model}")
    print(f"Total responses to generate: {len(behaviors)} behaviors x {len(df)} prompts = {len(behaviors) * len(df)}")
    
    if config.enable_validation:
        print(f"Validation enabled: will validate each behavior's responses")

    for behavior in behaviors:
        print(f"\n--- Processing behavior: {behavior.name} ---")
        results = generate_responses(
            df,
            dataset_desc['prompt_column'],
            behavior,
            config
        )
        
        # Validate if enabled
        if config.enable_validation:
            results = validate_responses(results, behaviors, config)
        
        save_results(results, dataset_name, behavior.name, config.output_dir)
    
    # Load all results and save combined file
    print(f"\nCombining all behavior results into single file...")
    all_results = load_all_results(config.output_dir, dataset_name)
    save_combined_results(all_results, config.output_dir, dataset_name)
    
    # Calculate and save validation metrics if enabled
    if config.enable_validation:
        print(f"\n[5/5] Calculating validation metrics")
        metrics = calculate_multilabel_metrics(all_results)
        save_validation_metrics(metrics, config.output_dir, dataset_name)
        
        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Overall Recall: {metrics['overall_metrics']['recall']:.1%} (ground truth detected)")
        print(f"Strict Accuracy: {metrics['overall_metrics']['strict_accuracy']:.1%} (only ground truth detected)")
        print(f"Avg Behaviors per Response: {metrics['overall_metrics']['avg_behaviors_per_response']:.2f}")
        
        # Show top co-occurrences
        if metrics['co_occurrence']:
            print(f"\nTop Behavior Co-occurrences:")
            for item in metrics['co_occurrence'][:5]:
                behaviors_str = " + ".join(item['behaviors'])
                print(f"  {behaviors_str}: {item['count']} times")

    print("\n" + "=" * 80)
    print("BENCHMARK CREATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {config.output_dir}/{dataset_name}/")
    print(f"  - Individual behavior files: {len(behaviors)} files")
    print(f"  - Combined file: all_behaviors.jsonl ({len(all_results)} total responses)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create benchmark dataset for behavior extraction evaluation")
    parser.add_argument("--dataset-desc", type=str, default="input_dataset_descriptions/instructeval.yaml",
                        help="Path to dataset description YAML file")
    parser.add_argument("--behavior-model", type=str, default="gpt-4.1",
                        help="Model for generating behavior descriptions")
    parser.add_argument("--response-model", type=str, default="gpt-4.1",
                        help="Model for generating responses")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of prompts to sample (None = use all)")
    parser.add_argument("--num-behaviors", type=int, default=10,
                        help="Number of task-specific behaviors to generate")
    parser.add_argument("--num-fewshot", type=int, default=10,
                        help="Number of random examples to use for few-shot behavior generation")
    parser.add_argument("--output-dir", type=str, default="benchmark/results/",
                        help="Output directory for results")
    parser.add_argument("--base-system-prompt", type=str, default="You are a helpful assistant.",
                        help="Base system prompt that will be modified by each behavior")
    parser.add_argument("--enable-validation", action="store_true",
                        help="Enable validation to check if behaviors are detectable (adds extra LLM calls)")
    parser.add_argument("--validation-model", type=str, default="gpt-4.1-mini",
                        help="Model for validation (default: gpt-4.1-mini)")
    parser.add_argument("--validation-max-workers", type=int, default=10,
                        help="Number of parallel workers for validation")

    args = parser.parse_args()

    config = BenchmarkConfig(
        dataset_description_path=args.dataset_desc,
        behavior_generation_model=args.behavior_model,
        response_generation_model=args.response_model,
        sample_size=args.sample_size,
        num_behaviors=args.num_behaviors,
        num_fewshot_examples=args.num_fewshot,
        output_dir=args.output_dir,
        base_system_prompt=args.base_system_prompt,
        enable_validation=args.enable_validation,
        validation_model=args.validation_model,
        validation_max_workers=args.validation_max_workers
    )

    create_benchmark(config)
