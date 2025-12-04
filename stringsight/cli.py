import argparse
import os
import sys
import subprocess
import signal
from pathlib import Path
from typing import Optional

def find_frontend_dist() -> Optional[Path]:
    """Find the frontend dist directory in the package installation."""
    # Try inside the stringsight package (installed location)
    import stringsight
    package_dir = Path(stringsight.__file__).parent
    dist_path = package_dir / "frontend_dist"

    if dist_path.exists():
        return dist_path

    # Try relative to this file (for development - repo root)
    cli_dir = Path(__file__).parent.parent
    dist_path = cli_dir / "frontend" / "dist"

    if dist_path.exists():
        return dist_path

    return None

def launch(host: str = "127.0.0.1", port: int = 5180, debug: bool = False):
    """Launch the StringSight UI with backend API."""
    # Check if frontend is built
    dist_path = find_frontend_dist()

    if not dist_path:
        print("Error: Frontend not found. Please ensure the frontend is built.")
        print("\nTo build the frontend:")
        print("  1. Navigate to the frontend directory")
        print("  2. Run: npm install && npm run build")
        sys.exit(1)

    print(f"Starting StringSight UI...")
    print(f"Access at: http://{host}:{port}")
    print(f"API available at: http://{host}:{port}/api")
    if debug:
        print(f"Debug mode: enabled")
    print("\nPress Ctrl+C to stop\n")

    # Create a combined FastAPI app that serves both API and static files
    import threading
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # Import the main API app
    from stringsight.api import app as api_app

    # Create a new app that combines API and frontend
    app = FastAPI()

    # Mount the API routes
    app.mount("/api", api_app)

    # Serve static files
    app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

    # Serve index.html for all other routes (SPA routing)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Check if it's a static file
        file_path = dist_path / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html (for SPA routing)
        return FileResponse(dist_path / "index.html")

    # Start the server with appropriate log level
    log_level = "debug" if debug else "info"
    try:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    except KeyboardInterrupt:
        print("\nShutting down...")

def main():
    parser = argparse.ArgumentParser(
        description="StringSight CLI - Explain Large Language Model Behavior Patterns"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Launch command
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch the StringSight UI"
    )
    launch_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    launch_parser.add_argument(
        "--port",
        type=int,
        default=5180,
        help="Port to run on (default: 5180)"
    )
    launch_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.command == "launch":
        launch(host=args.host, port=args.port, debug=args.debug)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
