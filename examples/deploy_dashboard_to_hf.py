#!/usr/bin/env python3
"""
Deploy StringSight Dashboard to HuggingFace Spaces

Usage:
    python examples/deploy_dashboard_to_hf.py
"""

from stringsight.dashboard import deploy_to_hf_spaces

# Example 1: Basic deployment
deploy_to_hf_spaces(
    results_dir="./results/my_experiment",
    space_name="my-dashboard",
    hf_username="your-hf-username",
    push=False  # Set to True to auto-push
)

# Example 2: Private Space
# deploy_to_hf_spaces(
#     results_dir="./results/my_experiment",
#     space_name="private-dashboard",
#     hf_username="your-hf-username",
#     private=True,
#     push=True
# )

# Example 3: Multiple experiments (if results/ has subdirectories)
# deploy_to_hf_spaces(
#     results_dir="./results",
#     space_name="all-experiments",
#     hf_username="your-hf-username",
#     push=True
# )

print("\n✅ Done! Review files and push to HuggingFace.")
print("See: stringsight/dashboard/HF_SPACES_DEPLOYMENT.md")
