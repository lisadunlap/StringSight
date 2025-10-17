"""
Demo script showing different ways to use the StringSight Gradio visualization.

This demonstrates the Python API for launching the Gradio app.
"""

import argparse
from pathlib import Path
from stringsight.dashboard import launch_app, create_app


def demo_basic_launch():
    """Demo: Basic launch without pre-loading data."""
    print("🚀 Demo: Basic launch - data can be loaded through the UI")
    launch_app()


def demo_preload_data(results_dir: str):
    """Demo: Launch with pre-loaded data."""
    print(f"🚀 Demo: Launch with pre-loaded data from {results_dir}")
    launch_app(results_dir=results_dir)


def demo_custom_settings(results_dir: str = None):
    """Demo: Launch with custom settings."""
    print("🚀 Demo: Launch with custom settings")
    launch_app(
        results_dir=results_dir,
        share=True,  # Create public shareable link
        server_name="0.0.0.0",  # Allow access from other machines
        server_port=8080,  # Custom port
    )


def demo_programmatic_access():
    """Demo: Create app object for programmatic access."""
    print("🚀 Demo: Programmatic app creation")
    
    # Create the app object without launching
    app = create_app()
    
    # You could modify the app here if needed
    # app.title = "My Custom Title"
    
    # Launch when ready
    print("Launching app...")
    app.launch(share=False, server_port=7861, show_error=False)


def main():
    parser = argparse.ArgumentParser(description="StringSight Gradio Visualization Demo")
    parser.add_argument("--results_dir", help="Path to results directory for demos")
    parser.add_argument("--demo", choices=[
        "basic", "preload", "custom", "programmatic"
    ], default="basic", help="Which demo to run")
    
    args = parser.parse_args()
    
    if args.demo == "basic":
        demo_basic_launch()
    elif args.demo == "preload":
        if not args.results_dir:
            print("❌ Error: --results_dir required for preload demo")
            return
        demo_preload_data(args.results_dir)
    elif args.demo == "custom":
        demo_custom_settings(args.results_dir)
    elif args.demo == "programmatic":
        demo_programmatic_access()


if __name__ == "__main__":
    main() 