"""Dashboard visualization for StringSight pipeline results.

This module provides a Gradio interface for exploring model performance, 
cluster analysis, and detailed examples from pipeline output.

Usage:
    from stringsight.dashboard import launch_app
    launch_app(results_dir="path/to/results")
    
Deploy to HuggingFace Spaces:
    from stringsight.dashboard import deploy_to_hf_spaces
    deploy_to_hf_spaces(
        results_dir="path/to/results",
        space_name="my-dashboard",
        hf_username="your-username"
    )
    
Or use the CLI:
    python -m stringsight.dashboard.deploy_to_hf --help
"""

from .app import launch_app, create_app

__all__ = ["launch_app", "create_app"]

# Lazy import for deployment function to avoid importing subprocess/etc unless needed
def deploy_to_hf_spaces(
    results_dir: str,
    space_name: str,
    hf_username: str,
    output_dir: str = None,
    private: bool = False,
    push: bool = False,
    create_space: bool = False
):
    """
    Deploy StringSight dashboard to HuggingFace Spaces.
    
    Creates a new Space if it doesn't exist, or updates an existing one.
    
    Args:
        results_dir: Path to results directory containing pipeline outputs
        space_name: Name for the HuggingFace Space (e.g., 'my-dashboard')
        hf_username: Your HuggingFace username
        output_dir: Output directory for Space files (default: ./hf_space_<space_name>)
        private: Make the Space private (default: False)
        push: Automatically push to HuggingFace after creating files (default: False)
              Also creates the Space if it doesn't exist.
        create_space: Automatically create the Space on HuggingFace (default: False)
        
    Returns:
        Path to the created Space directory
        
    Example:
        >>> from stringsight.dashboard import deploy_to_hf_spaces
        >>> # Deploy to new or existing Space
        >>> deploy_to_hf_spaces(
        ...     results_dir="./results/experiment1",
        ...     space_name="my-eval-dashboard",
        ...     hf_username="myusername",
        ...     push=True
        ... )
    """
    from . import deploy_to_hf as deploy_module
    from pathlib import Path
    import sys
    
    # Prepare arguments as if they came from argparse
    class Args:
        pass
    
    args = Args()
    args.results_dir = results_dir
    args.space_name = space_name
    args.hf_username = hf_username
    args.output_dir = output_dir
    args.private = private
    args.push = push
    args.create_space = create_space
    
    # Validate inputs
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    
    if not results_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {results_dir}")
    
    if not deploy_module.validate_results_directory(results_path):
        raise ValueError(
            f"Results directory does not contain required files. "
            f"Expected files: model_cluster_scores.json, cluster_scores.json, model_scores.json"
        )
    
    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path.cwd() / f"hf_space_{space_name}"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Preparing HuggingFace Space: {space_name}")
    print(f"📂 Output directory: {output_path}")
    print(f"🔒 Visibility: {'Private' if private else 'Public'}")
    print()
    
    # Create Space files
    print("📝 Creating Space files...")
    deploy_module.create_hf_app_file(output_path)
    deploy_module.create_requirements_file(output_path)
    deploy_module.create_readme_file(
        output_path,
        space_name,
        private,
        experiment_name=results_path.name
    )
    deploy_module.create_gitignore(output_path)
    
    # Copy results data
    print("\n📦 Copying results data...")
    deploy_module.copy_results_data(results_path, output_path)
    
    print(f"\n✅ Space files prepared successfully!")
    print(f"📂 Location: {output_path.absolute()}")
    
    # Create Space if requested
    if create_space or push:
        print("\n🌐 Creating Space on HuggingFace...")
        if not deploy_module.create_space_with_cli(hf_username, space_name, private):
            if push:
                print("⚠️  Could not create Space automatically. Please create it manually first:")
                print(f"   https://huggingface.co/new-space")
                raise RuntimeError("Space creation failed. Create manually and try again.")
    
    # Initialize git and push if requested
    if push:
        print("\n📤 Pushing to HuggingFace...")
        
        if not deploy_module.init_git_repo(output_path, hf_username, space_name):
            raise RuntimeError("Failed to initialize git repository")
        
        if not deploy_module.push_to_hf(output_path):
            print("\n💡 Manual push instructions:")
            print(f"   cd {output_path}")
            print(f"   git add .")
            print(f"   git commit -m 'Deploy StringSight dashboard'")
            print(f"   git push -u origin main")
            raise RuntimeError("Failed to push to HuggingFace")
        
        print(f"\n🎉 Successfully deployed!")
        print(f"🌐 View your Space at: https://huggingface.co/spaces/{hf_username}/{space_name}")
    else:
        print("\n📋 Next steps:")
        print(f"   1. Review the files in: {output_path}")
        print(f"   2. Create a new Space at: https://huggingface.co/new-space")
        print(f"   3. Push using: deploy_to_hf_spaces(..., push=True)")
    
    return str(output_path.absolute())


__all__ = ["launch_app", "create_app", "deploy_to_hf_spaces"] 