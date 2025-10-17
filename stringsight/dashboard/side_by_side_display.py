"""
Side-by-side display component for comparing model responses.

This module provides functionality to display two model responses side by side
for comparison, specifically designed for datasets with model_a_response and 
model_b_response fields.
"""

from typing import Dict, Any, Optional
from .conversation_display import convert_to_openai_format, display_openai_conversation_html
import html

def display_side_by_side_responses(
    model_a: str,
    model_b: str, 
    model_a_response: Any,
    model_b_response: Any,
    use_accordion: bool = True,
    pretty_print_dicts: bool = True,
    score: Optional[float] = None,
    scores_a: Optional[Dict[str, Any]] = None,
    scores_b: Optional[Dict[str, Any]] = None,
    winner: Optional[str] = None,
    evidence: Any = None,
    exhibits_a: Optional[bool] = None,
    exhibits_b: Optional[bool] = None
) -> str:
    """
    Display two model responses side by side for comparison.
    
    Args:
        model_a: Name of model A
        model_b: Name of model B
        model_a_response: Response data from model A
        model_b_response: Response data from model B
        use_accordion: If True, group system and info messages in collapsible accordions
        pretty_print_dicts: If True, pretty-print embedded dictionaries
        score: Optional score for the comparison
        winner: Optional winner indication ('model_a', 'model_b', or 'tie')
        evidence: Evidence used to highlight text in both conversations (same logic as single-model view)
        exhibits_a: If True, show a badge indicating model A exhibits the property
        exhibits_b: If True, show a badge indicating model B exhibits the property
        
    Returns:
        HTML string for side-by-side display
    """
    
    # Convert responses to OpenAI format
    conversation_a = convert_to_openai_format(model_a_response) if model_a_response != 'N/A' else None
    conversation_b = convert_to_openai_format(model_b_response) if model_b_response != 'N/A' else None
    
    # Generate conversation HTML for each model
    if conversation_a:
        html_a = display_openai_conversation_html(
            conversation_a, 
            use_accordion=use_accordion, 
            pretty_print_dicts=pretty_print_dicts,
            evidence=evidence
        )
    else:
        html_a = "<p style='color: #dc3545; font-style: italic;'>No response data available</p>"
        
    if conversation_b:
        html_b = display_openai_conversation_html(
            conversation_b, 
            use_accordion=use_accordion, 
            pretty_print_dicts=pretty_print_dicts,
            evidence=evidence
        )
    else:
        html_b = "<p style='color: #dc3545; font-style: italic;'>No response data available</p>"
    
    # Create winner badges if winner is specified
    winner_badge_a = ""
    winner_badge_b = ""
    if winner:
        if winner == 'model_a':
            winner_badge_a = """
            <span style="
                background: #28a745; 
                color: white; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold;
                margin-left: 10px;
            ">
                🏆 Winner
            </span>
            """
        elif winner == 'model_b':
            winner_badge_b = """
            <span style="
                background: #28a745; 
                color: white; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold;
                margin-left: 10px;
            ">
                🏆 Winner
            </span>
            """
        elif winner == 'tie':
            tie_badge = """
            <span style="
                background: #6c757d; 
                color: white; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold;
                margin-left: 10px;
            ">
                🤝 Tie
            </span>
            """
            winner_badge_a = tie_badge
            winner_badge_b = tie_badge
    
    # Add score badges if available
    score_info = ""
    
    # Handle new scores_a/scores_b format
    if scores_a is not None or scores_b is not None:
        score_parts = []
        
        if scores_a and isinstance(scores_a, dict):
            scores_a_str = ", ".join([f"{k}: {v}" for k, v in scores_a.items()])
            score_parts.append(f"<strong>{model_a}:</strong> {scores_a_str}")
        
        if scores_b and isinstance(scores_b, dict):
            scores_b_str = ", ".join([f"{k}: {v}" for k, v in scores_b.items()])
            score_parts.append(f"<strong>{model_b}:</strong> {scores_b_str}")
        
        if score_parts:
            score_info = f"""
            <div style="text-align: center; margin-bottom: 15px;">
                <div style="
                    background: #f8f9fa; 
                    border: 1px solid #dee2e6; 
                    padding: 10px; 
                    border-radius: 8px; 
                    font-size: 14px;
                ">
                    <strong>Scores:</strong><br>
                    {"<br>".join(score_parts)}
                </div>
            </div>
            """
    
    # Handle legacy score format
    elif score is not None and score != 'N/A':
        try:
            score_val = float(score)
            score_color = '#28a745' if score_val >= 0 else '#dc3545'
            score_info = f"""
            <div style="text-align: center; margin-bottom: 15px;">
                <span style="
                    background: {score_color}; 
                    color: white; 
                    padding: 6px 12px; 
                    border-radius: 15px; 
                    font-size: 16px; 
                    font-weight: bold;
                ">
                    Comparison Score: {score_val:.3f}
                </span>
            </div>
            """
        except (ValueError, TypeError):
            pass
    
    # Create the side-by-side layout
    exhibit_badge_style = (
        "background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 12px; "
        "font-size: 12px; font-weight: 600; margin-left: 10px; border: 1px solid #c8e6c9;"
    )
    exhibit_badge_a = (
        f"<span style=\"{exhibit_badge_style}\">Exhibits property</span>" if exhibits_a else ""
    )
    exhibit_badge_b = (
        f"<span style=\"{exhibit_badge_style}\">Exhibits property</span>" if exhibits_b else ""
    )
    side_by_side_html = f"""
    <div style="margin-bottom: 20px; overflow-x: auto; -webkit-overflow-scrolling: touch;">
        {score_info}
        <div style="display: flex; gap: 20px; margin-top: 10px; flex-wrap: nowrap; min-width: 960px;">
            <!-- Model A Column -->
            <div style="flex: 0 0 50%; min-width: 0; border: 2px solid #e9ecef; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                <h4 style="margin: 0 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #dee2e6; color: #495057; display: flex; align-items: center;">
                    <span style="background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 14px; margin-right: 10px;">A</span>
                    {html.escape(model_a)}
                    {winner_badge_a}
                    {exhibit_badge_a}
                </h4>
                <div style="font-size: 15px; line-height: 1.5;">
                    {html_a}
                </div>
            </div>
            
            <!-- Model B Column -->
            <div style="flex: 0 0 50%; min-width: 0; border: 2px solid #e9ecef; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                <h4 style="margin: 0 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #dee2e6; color: #495057; display: flex; align-items: center;">
                    <span style="background: #fd7e14; color: white; padding: 4px 8px; border-radius: 4px; font-size: 14px; margin-right: 10px;">B</span>
                    {html.escape(model_b)}
                    {winner_badge_b}
                    {exhibit_badge_b}
                </h4>
                <div style="font-size: 15px; line-height: 1.5;">
                    {html_b}
                </div>
            </div>
        </div>
    </div>
    """
    
    return side_by_side_html


def is_side_by_side_dataset(example: Dict[str, Any]) -> bool:
    """
    Check if an example contains side-by-side comparison data.
    
    Args:
        example: Example dictionary from the dataset
        
    Returns:
        True if the example has both model_a_response and model_b_response
    """
    # Check if this is a side-by-side dataset by looking for both model_a_response and model_b_response
    return 'model_a_response' in example and 'model_b_response' in example and \
           example.get('model_a_response') is not None and example.get('model_b_response') is not None


def extract_side_by_side_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract side-by-side comparison data from a row.
    
    Args:
        row: Row from the dataset
        
    Returns:
        Dictionary with extracted side-by-side data
    """
    # Extract scores for side-by-side format: score_a and score_b columns
    scores_a = row.get('score_a', {})
    scores_b = row.get('score_b', {})
    
    # Ensure they are dictionaries
    if not isinstance(scores_a, dict):
        scores_a = {}
    if not isinstance(scores_b, dict):
        scores_b = {}
    
    return {
        'model_a': row.get('model_a', 'Model A'),
        'model_b': row.get('model_b', 'Model B'), 
        'model_a_response': row.get('model_a_response', 'N/A'),
        'model_b_response': row.get('model_b_response', 'N/A'),
        'winner': row.get('winner', None),
        'score_a': scores_a,
        'score_b': scores_b
    } 