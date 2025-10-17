#!/usr/bin/env python3
"""
CLI launcher for StringSight Gradio visualization app.

Usage:
    python -m stringsight.dashboard.launcher --results_dir path/to/results
    
    Or directly:
    python stringsight/dashboard/launcher.py --results_dir path/to/results
"""

import argparse
import sys
from pathlib import Path
import logging

def main():
    parser = argparse.ArgumentParser(
        description="Launch StringSight Gradio visualization app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch with auto-loaded data from a base results directory
    python -m stringsight.dashboard.launcher --results_dir /path/to/results
    
    # Launch with public sharing enabled
    python -m stringsight.dashboard.launcher --results_dir /path/to/results --share
    
    # Launch on specific port
    python -m stringsight.dashboard.launcher --results_dir /path/to/results --port 8080
    
    # Launch with automatic port selection
    python -m stringsight.dashboard.launcher --results_dir /path/to/results --auto_port
    
    # Launch without auto-loading (manual selection in app)
    python -m stringsight.dashboard.launcher
        """
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to base results directory containing experiment subfolders (optional - can be loaded in the app)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    parser.add_argument(
        "--server_name",
        type=str,
        default="127.0.0.1",
        help="Server address (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860). Use --auto_port to automatically find an available port."
    )
    
    parser.add_argument(
        "--auto_port",
        action="store_true",
        help="Automatically find an available port by trying ports 8080-8089"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Handle auto_port option
    if args.auto_port:
        # Use a high port range for auto-port mode
        args.port = 8080
        print("🔍 Auto-port mode enabled - will try ports 8080-8089")
    
    # Validate results directory if provided
    if args.results_dir:
        results_path = Path(args.results_dir)
        if not results_path.exists():
            print(f"❌ Error: Results directory does not exist: {args.results_dir}")
            sys.exit(1)
        if not results_path.is_dir():
            print(f"❌ Error: Path is not a directory: {args.results_dir}")
            sys.exit(1)
    
    # Configure logging level when --debug is set
    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Import and launch the app
    try:
        from .app import launch_app
        
        print("🚀 Launching StringSight Gradio Visualization App...")
        print(f"🌐 Server: http://{args.server_name}:{args.port}")
        if args.share:
            print("🔗 Public sharing enabled")
        
        launch_app(
            results_dir=args.results_dir,
            share=args.share,
            server_name=args.server_name,
            server_port=args.port,
            debug=args.debug
        )
        
    except ImportError as e:
        print(f"❌ Error: Failed to import required modules: {e}")
        print("💡 Make sure you have gradio installed: pip install gradio")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 