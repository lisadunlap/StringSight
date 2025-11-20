"""
JSON parsing stage for extracted properties.

This stage migrates the parsing logic from post_processing.py into the pipeline architecture.
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Property
from ..core.mixins import LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin


class LLMJsonParser(LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin, PipelineStage):
    """
    Parse JSON responses from LLM property extraction.
    
    This stage takes raw LLM responses and parses them into structured Property objects.
    It handles JSON parsing errors gracefully and filters out invalid responses.
    """
    
    def __init__(self, *, fail_fast: bool = False, output_dir: Optional[str] = None, **kwargs):
        """Initialize the JSON parser.

        By default ``fail_fast`` is set to *False* so that a handful of
        malformed JSON responses do **not** crash the entire pipeline.  You
        can opt-in to strict mode by passing ``fail_fast=True``.
        """
        super().__init__(fail_fast=fail_fast, **kwargs)
        self.parsing_failures = []
        self.output_dir = Path(output_dir) if output_dir else None
        
    def run(self, data: PropertyDataset) -> PropertyDataset:
        """
        Parse raw LLM responses into Property objects.
        
        Args:
            data: PropertyDataset with properties containing raw LLM responses
            
        Returns:
            PropertyDataset with parsed and validated properties
        """
        self.log(f"Parsing {len(data.properties)} raw property responses")
        
        
        parsed_properties: List[Property] = []
        parse_errors = 0
        unknown_model_filtered = 0
        empty_list_responses = 0  # Track when LLM returns empty lists
        consecutive_errors = 0  # Track consecutive parsing errors
        max_consecutive_errors = 10

        # Add progress bar for better visibility
        for i, prop in enumerate(tqdm(data.properties, desc="Parsing properties", disable=not getattr(self, 'verbose', False))):
            # We only process properties that still have raw_response
            if not prop.raw_response:
                # Debug: Print information about the property with empty raw_response
                self.log(f"⚠️  Property {i+1}/{len(data.properties)} has empty raw_response:", level="error")
                self.log(f"   • Property ID: {prop.id}", level="error")
                self.log(f"   • Question ID: {prop.question_id}", level="error")
                self.log(f"   • Model: {prop.model}", level="error")
                self.log(f"   • Raw response: {repr(prop.raw_response)}", level="error")
                
                # Find the corresponding conversation to get more context
                matching_conv = None
                for conv in data.conversations:
                    if conv.question_id == prop.question_id:
                        matching_conv = conv
                        break
                
                if matching_conv:
                    self.log(f"   • Matching conversation found:", level="error")
                    self.log(f"     - Question ID: {matching_conv.question_id}", level="error")
                    self.log(f"     - Model: {matching_conv.model}", level="error")
                    if hasattr(matching_conv, 'model_response'):
                        response_snippet = str(matching_conv.model_response)[:200] if matching_conv.model_response else "None"
                        self.log(f"     - Model response snippet: {response_snippet}", level="error")
                    if hasattr(matching_conv, 'prompt'):
                        prompt_snippet = str(matching_conv.prompt)[:200] if matching_conv.prompt else "None"
                        self.log(f"     - Prompt snippet: {prompt_snippet}", level="error")
                else:
                    self.log(f"   • No matching conversation found for question_id: {prop.question_id}", level="error")
                
                # Throw an error to help debug the extraction issue
                raise ValueError(
                    f"Property {i+1}/{len(data.properties)} (ID: {prop.id}) has empty raw_response. "
                    f"This indicates the extraction stage failed to get a response from the LLM for "
                    f"question_id: {prop.question_id}, model: {prop.model}. "
                    f"Check your API connectivity, rate limits, or extraction stage configuration. "
                    f"Total properties with empty responses: {sum(1 for p in data.properties if not p.raw_response)}"
                )

            parsed_json = self._parse_json_response(prop.raw_response)
            if parsed_json is None:
                parse_errors += 1
                consecutive_errors += 1
                
                # Analyze the raw response to determine the specific issue
                error_details = self._analyze_json_parsing_error(prop.raw_response)
                
                # Collect failure information
                self.parsing_failures.append({
                    'property_id': prop.id,
                    'question_id': prop.question_id,
                    'model': prop.model,
                    'raw_response': prop.raw_response,
                    'error_type': 'JSON_PARSE_ERROR',
                    'error_message': error_details,
                    'consecutive_errors': consecutive_errors,
                    'index': i
                })
                
                # Minimal context about the failed input
                self.log(f"Parse failure input qid={prop.question_id} model={prop.model} raw={prop.raw_response}", level="error")

                # Debug: show a snippet of the offending response to aid troubleshooting
                snippet = (prop.raw_response or "")[:200].replace("\n", " ")
                self.log(
                    f"Failed to parse JSON for property {prop.id} ({consecutive_errors} consecutive errors). {error_details} Snippet: {snippet}…",
                    level="error",
                )
                
                # Check if we've exceeded consecutive error limit
                if consecutive_errors > max_consecutive_errors:
                    error_msg = (
                        f"ERROR: More than {max_consecutive_errors} consecutive parsing errors detected "
                        f"(currently {consecutive_errors}). This indicates a systematic issue with "
                        f"the LLM responses. Check your API connectivity, model configuration, "
                        f"or system prompts. Failed at property {i+1}/{len(data.properties)}."
                    )
                    self.log(error_msg, level="error")
                    raise RuntimeError(error_msg)
                
                # Log the parsing error but continue processing (parsing failures should never terminate)
                self.log(f"Parsing failure for property {prop.id}: {error_details}", level="error")
                continue

            # Successfully parsed JSON - reset consecutive error counter
            consecutive_errors = 0
            
            # The LLM might return a single property dict or {"properties": [...]} or a list
            if isinstance(parsed_json, dict) and "properties" in parsed_json:
                prop_dicts = parsed_json["properties"]
            elif isinstance(parsed_json, list):
                prop_dicts = parsed_json
            elif isinstance(parsed_json, dict):
                prop_dicts = [parsed_json]
            else:
                consecutive_errors += 1  # Count structure errors as parsing errors too
                
                error_details = f"Parsed JSON has unsupported type: {type(parsed_json)}. Expected dict, list, or dict with 'properties' key."
                
                # Collect structure error information
                self.parsing_failures.append({
                    'property_id': prop.id,
                    'question_id': prop.question_id,
                    'model': prop.model,
                    'raw_response': prop.raw_response,
                    'parsed_json': str(parsed_json),
                    'error_type': 'UNSUPPORTED_JSON_SHAPE',
                    'error_message': error_details,
                    'consecutive_errors': consecutive_errors,
                    'index': i
                })
                
                # Minimal context about the failed input
                self.log(f"Parse failure input qid={prop.question_id} model={prop.model} raw={prop.raw_response}", level="error")

                if consecutive_errors > max_consecutive_errors:
                    error_msg = (
                        f"ERROR: More than {max_consecutive_errors} consecutive parsing errors detected "
                        f"(currently {consecutive_errors}). This indicates a systematic issue with "
                        f"the LLM responses. Check your API connectivity, model configuration, "
                        f"or system prompts. Failed at property {i+1}/{len(data.properties)}."
                    )
                    self.log(error_msg, level="error")
                    raise RuntimeError(error_msg)
                    
                # Log the structure error but continue processing (parsing failures should never terminate)
                self.log(f"Unsupported JSON structure for property {prop.id}: {error_details}", level="error")
                parse_errors += 1
                continue

            # Successfully processed structure - reset consecutive error counter
            consecutive_errors = 0
            
            # Log when LLM returns empty list (no properties found)
            if isinstance(prop_dicts, list) and len(prop_dicts) == 0:
                self.log(f"LLM returned empty list for conversation {prop.question_id} (model: {prop.model})", level="warning")
                self.log(f"  Raw response snippet: {(prop.raw_response or '')[:200]}", level="debug")
                # This is not a parsing failure - the LLM legitimately found no properties
                empty_list_responses += 1
                continue
            
            for j, p_dict in enumerate(prop_dicts):
                try:
                    parsed_properties.append(self._to_property(p_dict, prop))
                except ValueError as e:
                    if "unknown or invalid model" in str(e):
                        unknown_model_filtered += 1
                        self.log(f"Filtered property with unknown model: {e}", level="debug")
                    else:
                        parse_errors += 1
                        
                        # Analyze the property dict to determine what's missing
                        error_details = self._analyze_property_dict_error(p_dict, prop, str(e))
                        
                        # Collect property building error information
                        self.parsing_failures.append({
                            'property_id': prop.id,
                            'question_id': prop.question_id,
                            'model': prop.model,
                            'raw_response': prop.raw_response,
                            'property_dict': p_dict,
                            'error_type': 'PROPERTY_BUILDING_ERROR',
                            'error_message': error_details,
                            'consecutive_errors': consecutive_errors,
                            'index': i
                        })
                        
                        # Log the property building error but continue processing (parsing failures should never terminate)
                        self.log(f"Failed to build property from JSON for {prop.question_id}: {error_details}", level="error")
                except Exception as e:
                    parse_errors += 1
                    
                    # Collect general error information
                    self.parsing_failures.append({
                        'property_id': prop.id,
                        'question_id': prop.question_id,
                        'model': prop.model,
                        'raw_response': prop.raw_response,
                        'property_dict': p_dict if 'p_dict' in locals() else None,
                        'error_type': 'GENERAL_ERROR',
                        'error_message': str(e),
                        'consecutive_errors': consecutive_errors,
                        'index': i
                    })
                    
                    # Log the general error but continue processing (parsing failures should never terminate)
                    self.log(f"Unexpected error building property from JSON for {prop.question_id}: {str(e)}", level="error")

        self.log(f"Parsed {len(parsed_properties)} properties successfully")
        self.log(f"Filtered out {unknown_model_filtered} properties with unknown models")
        self.log(f"{parse_errors} properties failed parsing")
        self.log(f"{empty_list_responses} conversations returned empty lists (no properties found)")
        self.log(f"Collected {len(self.parsing_failures)} detailed failure records")
        
        
        # Log to wandb if enabled
        if hasattr(self, 'use_wandb') and self.use_wandb:
            self._log_parsing_to_wandb(data.properties, parsed_properties, parse_errors, unknown_model_filtered, empty_list_responses)
        
        # Auto-save parsing results if output_dir is provided
        if self.output_dir:
            self._save_stage_results(data, parsed_properties, parse_errors, unknown_model_filtered, empty_list_responses)
        
        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=parsed_properties,
            clusters=data.clusters,
            model_stats=data.model_stats
        )
    
    def _save_stage_results(self, data: PropertyDataset, parsed_properties: List[Property], parse_errors: int, unknown_model_filtered: int, empty_list_responses: int):
        """Save parsing results to the specified output directory."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"✅ Auto-saving parsing results to: {self.output_dir}")
        
        # 1. Save parsed properties as JSONL
        properties_df = pd.DataFrame([prop.to_dict() for prop in parsed_properties])
        properties_path = self.output_dir / "parsed_properties.jsonl"
        properties_df.to_json(properties_path, orient="records", lines=True)
        self.log(f"  • Parsed properties: {properties_path}")
        
        # 2. Save parsing statistics
        stats = {
            "total_input_properties": len(data.properties),
            "total_parsed_properties": len(parsed_properties),
            "parse_errors": parse_errors,
            "unknown_model_filtered": unknown_model_filtered,
            "empty_list_responses": empty_list_responses,
            "parsing_success_rate": len(parsed_properties) / len(data.properties) if data.properties else 0,
            "failures_count": len(self.parsing_failures),
        }
        
        stats_path = self.output_dir / "parsing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.log(f"  • Parsing stats: {stats_path}")
        
        # 3. Save parsing failures if any
        if self.parsing_failures:
            failures_path = self.output_dir / "parsing_failures.jsonl"
            pd.DataFrame(self.parsing_failures).to_json(failures_path, orient="records", lines=True)
            self.log(f"  • Parsing failures: {failures_path}")
            
            # Also save a summary of error types
            error_types = {}
            for failure in self.parsing_failures:
                error_type = failure['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            error_summary_path = self.output_dir / "parsing_error_summary.json"
            with open(error_summary_path, 'w') as f:
                json.dump(error_types, f, indent=2)
            self.log(f"  • Error summary: {error_summary_path}")
    
    def get_parsing_failures(self) -> List[Dict[str, Any]]:
        """Get the collected parsing failures."""
        return self.parsing_failures
    
    def _analyze_json_parsing_error(self, raw_response: str) -> str:
        """Analyze a raw response to determine why JSON parsing failed."""
        if not raw_response:
            return "Raw response is empty or None"
        
        raw_response = raw_response.strip()
        
        # Check if it looks like JSON at all
        if not (raw_response.startswith('{') or raw_response.startswith('[')):
            if '```json' in raw_response:
                return "Response contains ```json markdown block but JSON extraction failed (missing closing ``` or malformed block)"
            elif '```' in raw_response:
                return "Response contains code block but JSON extraction failed (missing closing ``` or malformed block)"
            else:
                return "Response is not formatted as JSON (doesn't start with { or [)"
        
        # Try to identify common JSON syntax errors
        try:
            # Check for unclosed brackets/braces
            brace_count = raw_response.count('{') - raw_response.count('}')
            bracket_count = raw_response.count('[') - raw_response.count(']')
            
            if brace_count != 0:
                return f"Unmatched braces: {brace_count} more opening braces than closing braces"
            if bracket_count != 0:
                return f"Unmatched brackets: {bracket_count} more opening brackets than closing brackets"
            
            # Check for common syntax issues
            if raw_response.count('"') % 2 != 0:
                return "Unmatched quotes in JSON"
            
            # Try to parse and see what the specific error is
            import json
            json.loads(raw_response)
            return "JSON appears valid but parsing still failed (unknown reason)"
            
        except json.JSONDecodeError as e:
            return f"JSON syntax error: {str(e)}"
        except Exception as e:
            return f"Unexpected error during JSON analysis: {str(e)}"
    
    def _analyze_property_dict_error(self, p_dict: Dict[str, Any], prop: Property, original_error: str) -> str:
        """Analyze a property dict to determine what's missing or invalid."""
        issues = []
        
        # Check for required fields
        required_fields = ['property_description']
        for field in required_fields:
            if field not in p_dict:
                issues.append(f"Missing required field: '{field}'")
            elif not p_dict[field]:
                issues.append(f"Required field '{field}' is empty or null")
        
        # Check for model field issues
        if 'model' not in p_dict:
            issues.append("Missing 'model' field in property dict")
        elif not p_dict['model']:
            issues.append("'model' field is empty or null")
        
        # Check for common field type issues
        if 'property_description' in p_dict and not isinstance(p_dict['property_description'], str):
            issues.append(f"'property_description' should be string, got {type(p_dict['property_description'])}")
        
        if 'category' in p_dict and not isinstance(p_dict['category'], str):
            issues.append(f"'category' should be string, got {type(p_dict['category'])}")
        
        # Check for boolean fields
        boolean_fields = ['contains_errors', 'unexpected_behavior']
        for field in boolean_fields:
            if field in p_dict and not isinstance(p_dict[field], bool):
                issues.append(f"'{field}' should be boolean, got {type(p_dict[field])}")
        
        # If we found specific issues, return them
        if issues:
            return f"Property validation failed: {'; '.join(issues)}"
        
        # If no specific issues found, return the original error
        return f"Property validation failed: {original_error}"
    
    def _parse_json_response(self, response_text: str) -> Optional[Any]:
        """
        Parse JSON response from model, handling potential formatting issues.
        
        This method migrates the parse_json_response function from post_processing.py.
        """
        if not response_text:
            return None
            
        response_text = response_text.strip()
        
        # Try multiple extraction strategies
        json_content = None
        
        # Strategy 1: Look for ```json blocks
        if "```json" in response_text:
            try:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end != -1:
                    # Found closing ```, extract content between markers
                    json_content = response_text[json_start:json_end].strip()
                else:
                    # No closing ``` found, try to extract JSON from content after ```json
                    # Skip past any newline after ```json
                    content_start = json_start
                    if response_text[content_start:content_start+1] == '\n':
                        content_start += 1
                    # Try to find JSON starting with { or [ after the marker
                    remaining_text = response_text[content_start:].strip()
                    # Use Strategy 3 logic to find complete JSON in remaining text
                    for start_char in ['{', '[']:
                        start_pos = remaining_text.find(start_char)
                        if start_pos != -1:
                            # Find matching closing bracket
                            bracket_stack = []
                            for i, char in enumerate(remaining_text[start_pos:], start_pos):
                                if char in '{[':
                                    bracket_stack.append(char)
                                elif char in ']}':
                                    if bracket_stack:
                                        if (char == '}' and bracket_stack[-1] == '{') or (char == ']' and bracket_stack[-1] == '['):
                                            bracket_stack.pop()
                                            if not bracket_stack:  # Found complete JSON
                                                json_content = remaining_text[start_pos:i+1].strip()
                                                break
                            if json_content:
                                break
            except Exception:
                pass
        
        # Strategy 2: Look for ``` blocks (any language)
        if not json_content and "```" in response_text:
            try:
                # Find the first code block
                start_marker = response_text.find("```")
                if start_marker != -1:
                    # Find the end of the first line (language identifier)
                    first_line_end = response_text.find("\n", start_marker)
                    if first_line_end != -1:
                        # Start after the language identifier
                        content_start = first_line_end + 1
                        # Find the closing ```
                        content_end = response_text.find("```", content_start)
                        if content_end != -1:
                            json_content = response_text[content_start:content_end].strip()
                        else:
                            # No closing ```, try to extract JSON from remaining content
                            remaining_text = response_text[content_start:].strip()
                            # Try to find JSON starting with { or [
                            for start_char in ['{', '[']:
                                start_pos = remaining_text.find(start_char)
                                if start_pos != -1:
                                    # Find matching closing bracket
                                    bracket_stack = []
                                    for i, char in enumerate(remaining_text[start_pos:], start_pos):
                                        if char in '{[':
                                            bracket_stack.append(char)
                                        elif char in ']}':
                                            if bracket_stack:
                                                if (char == '}' and bracket_stack[-1] == '{') or (char == ']' and bracket_stack[-1] == '['):
                                                    bracket_stack.pop()
                                                    if not bracket_stack:  # Found complete JSON
                                                        json_content = remaining_text[start_pos:i+1].strip()
                                                        break
                                    if json_content:
                                        break
            except Exception:
                pass
        
        # Strategy 3: Look for JSON-like content anywhere in the response
        if not json_content:
            # Try to find JSON-like content (starts with { or [)
            for start_char in ['{', '[']:
                start_pos = response_text.find(start_char)
                if start_pos != -1:
                    # Find matching closing bracket
                    bracket_stack = []
                    for i, char in enumerate(response_text[start_pos:], start_pos):
                        if char in '{[':
                            bracket_stack.append(char)
                        elif char in ']}':
                            if bracket_stack:
                                if (char == '}' and bracket_stack[-1] == '{') or (char == ']' and bracket_stack[-1] == '['):
                                    bracket_stack.pop()
                                    if not bracket_stack:  # Found complete JSON
                                        json_content = response_text[start_pos:i+1].strip()
                                        break
                    if json_content:
                        break
        
        # Strategy 4: Use the entire response if it looks like JSON
        if not json_content:
            if response_text.startswith('{') or response_text.startswith('['):
                json_content = response_text
        
        # If we found content, try to parse it
        if json_content:
            try:
                # Clean up common issues
                json_content = self._clean_json_content(json_content)
                return json.loads(json_content)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed_content = self._fix_common_json_issues(json_content)
                try:
                    return json.loads(fixed_content)
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def _clean_json_content(self, content: str) -> str:
        """Clean up common issues in JSON content."""
        import re
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Remove any trailing commas
        content = re.sub(r',\s*$', '', content)
        
        # Fix common quote issues
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        
        return content.strip()
    
    def _fix_common_json_issues(self, content: str) -> str:
        """Try to fix common JSON formatting issues."""
        import re
        
        # Look for the largest valid JSON object/array
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Nested arrays
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Return the longest match (most complete)
                return max(matches, key=len)
        
        return content

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _to_property(self, p: Dict[str, Any], prop: Property) -> Property:
        """Convert a dict returned by the LLM into a Property object."""

        if isinstance(prop.model, list):
            model = model_name_pass(p.get("model"), prop.model[0], prop.model[1])
        else:
            model = prop.model

        # Explicitly filter out properties with unknown models
        if (
            model == "unknown"
            or isinstance(model, (list, tuple))
            or not isinstance(model, str)
            or (isinstance(model, float) and (model != model))  # NaN check
            or model.strip() == ""
        ):
            error_details = f"Model validation failed: model='{model}' (type: {type(model)})"
            if model == "unknown":
                error_details += ". The JSON property dict is missing a 'model' field or it couldn't be resolved from the original conversation model."
            elif isinstance(model, (list, tuple)):
                error_details += ". Model should be a string, not a list/tuple."
            elif not isinstance(model, str):
                error_details += ". Model should be a string."
            elif isinstance(model, float) and (model != model):
                error_details += ". Model is NaN (Not a Number)."
            elif model.strip() == "":
                error_details += ". Model is empty or whitespace only."
            
            raise ValueError(error_details)

        return Property(
            id=str(uuid.uuid4()),
            question_id=prop.question_id,
            model=model,
            property_description=p.get("property_description"),
            category=p.get("category"),
            reason=p.get("reason"),
            evidence=p.get("evidence"),
            contains_errors=p.get("contains_errors"),
            unexpected_behavior=p.get("unexpected_behavior"),
            behavior_type=p.get("behavior_type"),
        )

    def _log_parsing_to_wandb(self, raw_properties: List[Property], parsed_properties: List[Property], parse_errors: int, unknown_model_filtered: int, empty_list_responses: int):
        """Log parsing results to wandb."""
        try:
            import wandb
            # import weave
            
            # Calculate parsing success rate
            total_raw = len(raw_properties)
            total_parsed = len(parsed_properties)
            parse_success_rate = total_parsed / total_raw if total_raw > 0 else 0
            
            # Log parsing summary statistics as summary metrics (not regular metrics)
            summary_stats = {
                "parsing_total_properties": total_raw,
                "parsing_successful_properties": total_parsed,
                "parsing_failed_properties": parse_errors,
                "parsing_success_rate": parse_success_rate,
                "parsing_filtered_unknown_models": unknown_model_filtered,
                "parsing_empty_list_responses": empty_list_responses,
                "parsing_failures_count": len(self.parsing_failures),
            }
            self.log_wandb(summary_stats, is_summary=True)
            
            # Log parsing failures if any
            if self.parsing_failures:
                # Log failure summary by error type
                error_type_counts = {}
                for failure in self.parsing_failures:
                    error_type = failure['error_type']
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                
                # Log error type distribution as summary metrics
                for error_type, count in error_type_counts.items():
                    self.log_wandb({f"parsing_failures_{error_type.lower()}": count}, is_summary=True)
                
                # Log detailed failures table (sample if too many)
                sample_size = min(100, len(self.parsing_failures))
                sample_failures = self.parsing_failures[:sample_size]
                
                failure_rows = []
                for failure in sample_failures:
                    # Truncate long text fields for wandb table display
                    raw_response_snippet = str(failure.get('raw_response', ''))[:200] + '...' if len(str(failure.get('raw_response', ''))) > 200 else str(failure.get('raw_response', ''))
                    error_message_snippet = str(failure.get('error_message', ''))[:100] + '...' if len(str(failure.get('error_message', ''))) > 100 else str(failure.get('error_message', ''))
                    
                    failure_rows.append([
                        failure.get('property_id', ''),
                        failure.get('question_id', ''),
                        failure.get('model', ''),
                        failure.get('error_type', ''),
                        error_message_snippet,
                        raw_response_snippet,
                        failure.get('consecutive_errors', 0),
                        failure.get('index', 0)
                    ])
                
                if failure_rows:
                    failure_cols = ['property_id', 'question_id', 'model', 'error_type', 'error_message', 'raw_response_snippet', 'consecutive_errors', 'index']
                    self.log_wandb({
                        "Property_Extraction/parsing_failures": wandb.Table(columns=failure_cols, data=failure_rows)
                    })
            
            # Log a sample of parsed properties (as table, not summary)
            if parsed_properties:
                sample_size = min(100, len(parsed_properties))
                sample_properties = parsed_properties[:sample_size]
                
                # Build rows with only non-null attributes
                rows: List[Dict[str, Any]] = []
                dynamic_cols: set[str] = set()

                for prop in sample_properties:
                    row: Dict[str, Any] = {
                        "question_id": prop.question_id,
                        "model": prop.model,
                        "property_description": prop.property_description,
                    }

                    # Optional fields – add only if they are not None/empty
                    optional = {
                        "reason": prop.reason,
                        "evidence": prop.evidence,
                        "category": prop.category,
                        "behavior_type": prop.behavior_type,
                        "contains_errors": prop.contains_errors,
                        "unexpected_behavior": prop.unexpected_behavior,
                    }
                    for k, v in optional.items():
                        if v not in [None, "", []]:
                            row[k] = v
                    dynamic_cols.update(row.keys())
                    rows.append(row)

                # Ensure consistent column order: mandatory first, then sorted rest
                mandatory = ["question_id", "model", "property_description"]
                other_cols = [c for c in sorted(dynamic_cols) if c not in mandatory]
                cols = mandatory + other_cols

                data_matrix = [[row.get(col, "") for col in cols] for row in rows]

                self.log_wandb({
                    "Property_Extraction/parsed_properties_sample": wandb.Table(columns=cols, data=data_matrix)
                })
            
        except Exception as e:
            self.log(f"Failed to log parsing to wandb: {e}", level="warning")

def remove_things(x):
    x = x[x.find('_')+1:]
    x = x.replace("-Instruct", "")
    return x.lower()

def model_name_pass(model, model_a, model_b):
    model_a_modified_name = remove_things(model_a)
    model_b_modified_name = remove_things(model_b)
    model_modified_name = remove_things(model)
    if model == model_a or model.lower() == "model a" or model_modified_name == model_a_modified_name:
        return model_a
    if model == model_b or model.lower() == "model b" or model_modified_name == model_b_modified_name:
        return model_b
    return "unknown"