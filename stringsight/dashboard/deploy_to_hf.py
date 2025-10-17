#!/usr/bin/env python3
"""
Deploy StringSight dashboard to HuggingFace Spaces.

This script creates a HuggingFace Space with your pipeline results
embedded in the dashboard for public viewing and sharing.

Usage:
    python -m stringsight.dashboard.deploy_to_hf \
        --results_dir /path/to/results \
        --space_name my-stringsight-dashboard \
        --hf_username your-username \
        [--private] [--push]
"""

import argparse
import shutil
import sys
import json
import os
from pathlib import Path
from typing import Optional
import subprocess


def create_hf_app_file(output_dir: Path) -> None:
    """Create the app.py file for HuggingFace Spaces."""
    app_content = '''#!/usr/bin/env python3
"""
StringSight Dashboard on HuggingFace Spaces
Automatically deployed evaluation results viewer
"""

import os
from pathlib import Path

# Set the base results directory to the embedded results
# This tells the dashboard to automatically load from the results folder
os.environ["STRINGSIGHT_BASE_RESULTS_DIR"] = str(Path(__file__).parent / "results")

# Import and launch the dashboard
from stringsight.dashboard import launch_app

if __name__ == "__main__":
    # Launch with the embedded results directory
    launch_app(
        results_dir="results",
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
'''
    
    with open(output_dir / "app.py", "w") as f:
        f.write(app_content)
    
    print(f"✅ Created app.py")


def create_requirements_file(output_dir: Path) -> None:
    """Create requirements.txt with minimal dependencies for the dashboard."""
    requirements = """# StringSight Dashboard Dependencies
gradio>=5.0.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
markdown>=3.4.0

# StringSight package from PyPI
stringsight>=0.0.1
"""
    
    with open(output_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    
    print(f"✅ Created requirements.txt")


def create_readme_file(
    output_dir: Path,
    space_name: str,
    is_private: bool,
    experiment_name: Optional[str] = None
) -> None:
    """Create README.md with HuggingFace Spaces YAML frontmatter."""
    visibility = "private" if is_private else "public"
    
    readme_content = f'''---
title: {space_name}
emoji: 🧵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# StringSight Dashboard{f": {experiment_name}" if experiment_name else ""}

This Space hosts a StringSight evaluation dashboard with embedded pipeline results.

## About StringSight

StringSight extracts, clusters, and analyzes behavioral properties from Large Language Models. 
This dashboard provides an interactive interface to explore:

- **📊 Overview**: Model quality metrics and behavioral cluster summaries
- **📋 View Clusters**: Explore behavioral property clusters interactively  
- **🔍 View Examples**: Inspect individual examples with rich conversation rendering
- **📊 Plots**: Frequency and quality plots across models and clusters

## Features

### Overview Tab
Compare model quality metrics and view model cards with top behavior clusters. 
Use Benchmark Metrics to switch between Plot/Table and Filter Controls to refine results.

### View Clusters Tab
Explore clusters interactively. Use the search box to filter cluster labels. 
Sidebar Tags (when available) filter all tabs consistently.

### View Examples Tab
Inspect individual examples with rich conversation rendering. 
Filter by prompt/model/cluster; adjust max examples and formatting options; 
optionally show only unexpected behavior.

### Plots Tab
Create frequency or quality plots across models and clusters. 
Toggle confidence intervals, pick a quality metric, and select clusters to compare.

## Data

This Space contains pre-computed analysis results from the StringSight pipeline.
The dashboard is read-only and displays the embedded results.

## Learn More

- **GitHub**: [StringSight Repository](https://github.com/lisabdunlap/StringSight)
- **Documentation**: Check the repository README for full documentation

## Citation

If you use StringSight in your research, please cite our work:

```bibtex
@software{{stringsight2024,
  title = {{StringSight: Extract, cluster, and analyze behavioral properties from Large Language Models}},
  author = {{Dunlap, Lisa}},
  year = {{2024}},
  url = {{https://github.com/lisabdunlap/StringSight}}
}}
```

---

*Deployed using StringSight's automatic HuggingFace Spaces deployment*
'''
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Created README.md")


def copy_results_data(results_dir: Path, output_dir: Path) -> None:
    """
    Copy only the required results files to the output directory.
    
    Only copies files needed by the dashboard:
    - model_cluster_scores.json
    - cluster_scores.json
    - model_scores.json
    - clustered_results_lightweight.jsonl
    
    This minimizes the Space size by excluding unnecessary files like
    full embeddings, redundant formats, etc.
    
    Args:
        results_dir: Path to the results folder containing pipeline outputs
        output_dir: Path to the HF Space directory
    """
    results_output = output_dir / "results"
    
    # If results_dir is already a base directory with subdirectories (experiments),
    # copy the entire structure
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    # Check if this looks like a base directory with experiment subdirectories
    # by checking if subdirectories contain the required metrics files
    required_files = [
        "model_cluster_scores.json",
        "cluster_scores.json",
        "model_scores.json",
        "clustered_results_lightweight.jsonl"
    ]
    
    is_base_dir = False
    if subdirs:
        # Check if first subdir has the required files
        first_subdir = subdirs[0]
        if all((first_subdir / f).exists() for f in required_files):
            is_base_dir = True
    
    if is_base_dir:
        # Copy only required files from all experiment subdirectories
        print(f"📁 Detected base results directory with {len(subdirs)} experiments")
        results_output.mkdir(parents=True, exist_ok=True)
        
        for subdir in subdirs:
            dest_subdir = results_output / subdir.name
            dest_subdir.mkdir(parents=True, exist_ok=True)
            print(f"   Copying experiment: {subdir.name}")
            
            # Only copy required files (not all files)
            for filename in required_files:
                src_file = subdir / filename
                if src_file.exists():
                    shutil.copy2(src_file, dest_subdir / filename)
                    print(f"     ✓ {filename}")
        
        print(f"✅ Copied {len(subdirs)} experiments to {results_output}")
    else:
        # Single experiment - copy only required files into a single subfolder
        print(f"📁 Detected single experiment results directory")
        experiment_name = results_dir.name
        dest_subdir = results_output / experiment_name
        dest_subdir.mkdir(parents=True, exist_ok=True)
        
        # Only copy required files (not all files)
        for filename in required_files:
            src_file = results_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dest_subdir / filename)
                print(f"   ✓ {filename}")
        
        print(f"✅ Copied experiment '{experiment_name}' to {dest_subdir}")


def validate_results_directory(results_dir: Path) -> bool:
    """
    Validate that the results directory contains required files.
    
    Args:
        results_dir: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_files = [
        "model_cluster_scores.json",
        "cluster_scores.json",
        "model_scores.json",
    ]
    
    # Check if it's a single experiment directory
    if all((results_dir / f).exists() for f in required_files):
        return True
    
    # Check if it's a base directory with experiment subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if subdirs:
        # Check if any subdir has the required files
        for subdir in subdirs:
            if all((subdir / f).exists() for f in required_files):
                return True
    
    return False


def create_gitignore(output_dir: Path) -> None:
    """Create .gitignore file for the Space."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Gradio
flagged/
"""
    
    with open(output_dir / ".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print(f"✅ Created .gitignore")


def init_git_repo(output_dir: Path, hf_username: str, space_name: str) -> bool:
    """
    Initialize git repository and configure remote.
    
    Args:
        output_dir: Path to the Space directory
        hf_username: HuggingFace username
        space_name: Name of the Space
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if .git directory already exists
        git_dir = output_dir / ".git"
        if git_dir.exists():
            print(f"ℹ️  Git repository already exists")
            # Remove it to start fresh (avoids conflicts with previous attempts)
            import shutil
            shutil.rmtree(git_dir)
        
        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=output_dir,
            check=True,
            capture_output=True
        )
        
        # Add HuggingFace remote
        remote_url = f"https://huggingface.co/spaces/{hf_username}/{space_name}"
        subprocess.run(
            ["git", "remote", "add", "origin", remote_url],
            cwd=output_dir,
            check=True,
            capture_output=True
        )
        
        print(f"✅ Initialized git repository with remote: {remote_url}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize git repository: {e}")
        return False
    except FileNotFoundError:
        print("❌ Git not found. Please install git to use the --push option.")
        return False


def check_git_lfs() -> bool:
    """Check if Git LFS is installed."""
    try:
        subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def find_large_result_files(base_dir: Path, min_bytes: int = 10 * 1024 * 1024) -> list[Path]:
    """
    Find large result files that are likely to exceed regular Git size limits.

    Args:
        base_dir: The Space output directory (expects a `results/` subfolder)
        min_bytes: Threshold in bytes to consider a file "large"

    Returns:
        List of file Paths under `results/` exceeding the size threshold.
    """
    results_dir = base_dir / "results"
    if not results_dir.exists():
        return []

    large_files: list[Path] = []
    for path in results_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".parquet"}:
            try:
                if path.stat().st_size >= min_bytes:
                    large_files.append(path)
            except OSError:
                # Skip unreadable files
                continue
    return large_files


def push_to_hf(output_dir: Path) -> bool:
    """
    Push the Space to HuggingFace.
    
    Args:
        output_dir: Path to the Space directory
        
    Returns:
        True if successful, False otherwise
    """
    # Check if Git LFS is installed; if not and large files exist, abort early with guidance
    if not check_git_lfs():
        large_files = find_large_result_files(output_dir)
        if large_files:
            print("❌ Git LFS is required to push large result files (>10MB).")
            print("   Offending files (first 5):")
            for p in large_files[:5]:
                try:
                    size_mb = p.stat().st_size / (1024 * 1024)
                    print(f"   - {p} ({size_mb:.1f} MiB)")
                except OSError:
                    print(f"   - {p}")
            print("\n👉 Install Git LFS and re-run deploy:")
            print("   Ubuntu/Debian: sudo apt-get install git-lfs && git lfs install")
            print("   macOS: brew install git-lfs && git lfs install")
            print("   Docs: https://git-lfs.com/")
            return False
        else:
            print("ℹ️  Git LFS not found, but no large files detected. Proceeding without LFS...")
    
    try:
        # Set up Git LFS for large files (if available)
        if check_git_lfs():
            subprocess.run(
                ["git", "lfs", "install"],
                cwd=output_dir,
                check=True,
                capture_output=True
            )
            
            # Track large files with Git LFS
            subprocess.run(
                ["git", "lfs", "track", "results/**/*.jsonl"],
                cwd=output_dir,
                check=True,
                capture_output=True
            )
            # Also track large JSON summaries used by the dashboard
            subprocess.run(
                ["git", "lfs", "track", "results/**/*.json"],
                cwd=output_dir,
                check=True,
                capture_output=True
            )
            
            subprocess.run(
                ["git", "lfs", "track", "results/**/*.parquet"],
                cwd=output_dir,
                check=True,
                capture_output=True
            )
            
            # Add .gitattributes
            subprocess.run(
                ["git", "add", ".gitattributes"],
                cwd=output_dir,
                check=True,
                capture_output=True
            )
            
            # Commit LFS config (if it exists)
            subprocess.run(
                ["git", "commit", "-m", "Configure Git LFS"],
                cwd=output_dir,
                check=False,  # May fail if no changes
                capture_output=True
            )
        
        # Add all files
        subprocess.run(
            ["git", "add", "."],
            cwd=output_dir,
            check=True,
            capture_output=True
        )
        
        # Commit
        subprocess.run(
            ["git", "commit", "-m", "Deploy StringSight dashboard"],
            cwd=output_dir,
            check=True,
            capture_output=True
        )
        
        # Force push to overwrite HuggingFace's initial files
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main", "--force"],
            cwd=output_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            stderr_lower = (result.stderr or "").lower()
            if "contains files larger than 10 mib" in stderr_lower or "please use" in stderr_lower and "git-lfs" in stderr_lower:
                print("❌ Push rejected by remote due to large files without Git LFS.")
                print("   Install Git LFS and re-run deploy:")
                print("   Ubuntu/Debian: sudo apt-get install git-lfs && git lfs install")
                print("   macOS: brew install git-lfs && git lfs install")
                print("   Docs: https://hf.co/docs/hub/repositories-getting-started#terminal")
            else:
                print(f"❌ Push failed: {result.stderr}")
            return False
        
        print(f"✅ Successfully pushed to HuggingFace!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push to HuggingFace: {e}")
        return False


def get_hf_cli_path() -> Optional[str]:
    """Get the path to huggingface-cli in the current Python environment."""
    # First, try to import huggingface_hub to see if it's installed
    try:
        import huggingface_hub
    except ImportError:
        return None
    
    # Try to find huggingface-cli in the same bin directory as python
    python_dir = os.path.dirname(sys.executable)
    potential_paths = [
        os.path.join(python_dir, "huggingface-cli"),
        os.path.join(python_dir, "hf"),  # newer CLI command
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    # Fall back to searching PATH
    try:
        result = subprocess.run(
            ["which", "huggingface-cli"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return None


def check_hf_cli() -> bool:
    """Check if huggingface_hub CLI is installed."""
    cli_path = get_hf_cli_path()
    if not cli_path:
        return False
    
    try:
        # Try 'version' command (new CLI)
        result = subprocess.run(
            [cli_path, "version"],
            capture_output=True,
            check=False
        )
        if result.returncode == 0:
            return True
        
        # Fall back to just checking if the executable exists and is executable
        return os.path.exists(cli_path) and os.access(cli_path, os.X_OK)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_space_with_cli(hf_username: str, space_name: str, is_private: bool) -> bool:
    """
    Create a new Space using HuggingFace CLI, or verify it exists.
    
    Args:
        hf_username: HuggingFace username
        space_name: Name of the Space
        is_private: Whether the Space should be private
        
    Returns:
        True if Space exists or was created successfully, False otherwise
    """
    cli_path = get_hf_cli_path()
    if not cli_path:
        print("⚠️  HuggingFace CLI not found. Space must be created manually.")
        print("   Install with: pip install huggingface_hub[cli]")
        print(f"   Or create manually at: https://huggingface.co/new-space")
        return False
    
    try:
        visibility = ["--private"] if is_private else []
        cmd = [cli_path, "repo", "create", space_name, "--type", "space", "--space_sdk", "gradio"] + visibility
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Space might already exist (check various error messages)
            error_text = (result.stderr + result.stdout).lower()
            if any(phrase in error_text for phrase in [
                "already exists",
                "already created this",
                "409 client error: conflict"
            ]):
                print(f"✅ Space already exists: {hf_username}/{space_name}")
                print(f"   Will push to existing Space")
                return True
            else:
                print(f"❌ Failed to create Space: {result.stderr}")
                return False
        
        print(f"✅ Created new Space: {hf_username}/{space_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating Space: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy StringSight dashboard to HuggingFace Spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create deployment folder only (manual push)
  python -m stringsight.dashboard.deploy_to_hf \\
      --results_dir /path/to/results \\
      --space_name my-stringsight-dashboard \\
      --hf_username your-username \\
      --output_dir ./hf_space
  
  # Create and automatically push to HuggingFace
  python -m stringsight.dashboard.deploy_to_hf \\
      --results_dir /path/to/results \\
      --space_name my-stringsight-dashboard \\
      --hf_username your-username \\
      --push
  
  # Create private Space
  python -m stringsight.dashboard.deploy_to_hf \\
      --results_dir /path/to/results \\
      --space_name my-private-dashboard \\
      --hf_username your-username \\
      --private \\
      --push

Note: The --push option requires git and optionally huggingface-cli to be installed.
You must be logged in to HuggingFace (huggingface-cli login) for automatic Space creation.
        """
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory containing pipeline outputs"
    )
    
    parser.add_argument(
        "--space_name",
        type=str,
        required=True,
        help="Name for the HuggingFace Space (e.g., 'my-stringsight-dashboard')"
    )
    
    parser.add_argument(
        "--hf_username",
        type=str,
        required=True,
        help="Your HuggingFace username"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the Space files (default: ./hf_space_<space_name>)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Space private (default: public)"
    )
    
    parser.add_argument(
        "--push",
        action="store_true",
        help="Automatically push to HuggingFace after creating files"
    )
    
    parser.add_argument(
        "--create_space",
        action="store_true",
        help="Automatically create the Space on HuggingFace (requires huggingface-cli)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"❌ Error: Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    if not results_path.is_dir():
        print(f"❌ Error: Path is not a directory: {args.results_dir}")
        sys.exit(1)
    
    if not validate_results_directory(results_path):
        print(f"❌ Error: Results directory does not contain required files.")
        print("   Expected files: model_cluster_scores.json, cluster_scores.json, model_scores.json")
        print("   The directory should contain either:")
        print("   1. A single experiment with these files directly")
        print("   2. Multiple experiment subdirectories, each containing these files")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path.cwd() / f"hf_space_{args.space_name}"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Preparing HuggingFace Space: {args.space_name}")
    print(f"📂 Output directory: {output_path}")
    print(f"🔒 Visibility: {'Private' if args.private else 'Public'}")
    print()
    
    # Create Space files
    print("📝 Creating Space files...")
    create_hf_app_file(output_path)
    create_requirements_file(output_path)
    create_readme_file(
        output_path,
        args.space_name,
        args.private,
        experiment_name=results_path.name
    )
    create_gitignore(output_path)
    
    # Copy results data
    print("\n📦 Copying results data...")
    copy_results_data(results_path, output_path)
    
    print(f"\n✅ Space files prepared successfully!")
    print(f"📂 Location: {output_path.absolute()}")
    
    # Create Space if requested
    if args.create_space or args.push:
        print("\n🌐 Creating Space on HuggingFace...")
        if not create_space_with_cli(args.hf_username, args.space_name, args.private):
            if args.push:
                print("⚠️  Could not create Space automatically. Please create it manually first:")
                print(f"   https://huggingface.co/new-space")
                print(f"   Then run with --push to deploy.")
                sys.exit(1)
    
    # Initialize git and push if requested
    if args.push:
        print("\n📤 Pushing to HuggingFace...")
        
        if not init_git_repo(output_path, args.hf_username, args.space_name):
            print("❌ Failed to initialize git repository")
            sys.exit(1)
        
        if not push_to_hf(output_path):
            print("❌ Failed to push to HuggingFace")
            print("\n💡 Manual push instructions:")
            print(f"   cd {output_path}")
            print(f"   git add .")
            print(f"   git commit -m 'Deploy StringSight dashboard'")
            print(f"   git push -u origin main")
            sys.exit(1)
        
        print(f"\n🎉 Successfully deployed!")
        print(f"🌐 View your Space at: https://huggingface.co/spaces/{args.hf_username}/{args.space_name}")
    else:
        print("\n📋 Next steps:")
        print(f"   1. Review the files in: {output_path}")
        print(f"   2. Create a new Space at: https://huggingface.co/new-space")
        print(f"      - Name: {args.space_name}")
        print(f"      - SDK: Gradio")
        print(f"      - Visibility: {'Private' if args.private else 'Public'}")
        print(f"   3. Clone and push:")
        print(f"      cd {output_path}")
        print(f"      git init")
        print(f"      git remote add origin https://huggingface.co/spaces/{args.hf_username}/{args.space_name}")
        print(f"      git add .")
        print(f"      git commit -m 'Deploy StringSight dashboard'")
        print(f"      git push -u origin main")
        print(f"\n   Or run again with --push to deploy automatically")


if __name__ == "__main__":
    main()

