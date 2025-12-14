import argparse
import os
import sys
import subprocess
import signal
import time
import urllib.request
import shutil
from pathlib import Path
from typing import Optional

def get_config_dir() -> Path:
    """Get the StringSight configuration directory."""
    config_dir = Path.home() / ".stringsight"
    config_dir.mkdir(exist_ok=True)
    return config_dir

def get_pid_file() -> Path:
    """Get the PID file path."""
    return get_config_dir() / "server.pid"

def get_log_dir() -> Path:
    """Get the log directory."""
    log_dir = get_config_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir

def get_log_file() -> Path:
    """Get the log file path."""
    return get_log_dir() / "server.log"

def is_server_running() -> Optional[int]:
    """Check if server is running. Returns PID if running, None otherwise."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        return None

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        # Check if process is actually running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file exists but process is not running
        pid_file.unlink(missing_ok=True)
        return None

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

def launch(host: str = "127.0.0.1", port: int = 5180, debug: bool = False,
           daemon: bool = False, workers: int = 1):
    """Launch the StringSight UI with backend API."""
    # Check if server is already running
    if daemon:
        running_pid = is_server_running()
        if running_pid:
            print(f"Error: Server is already running (PID: {running_pid})")
            print(f"Stop it first with: stringsight stop")
            sys.exit(1)

    # Check if frontend is built
    dist_path = find_frontend_dist()

    if not dist_path:
        print("Error: Frontend not found. Please ensure the frontend is built.")
        
        # Check if we're in a development environment (repo root)
        cli_dir = Path(__file__).parent.parent
        frontend_dir = cli_dir / "frontend"
        frontend_package_json = frontend_dir / "package.json"
        
        if frontend_dir.exists() and not frontend_package_json.exists():
            # Frontend directory exists but is empty - submodule not initialized
            print("\nThe frontend submodule is not initialized.")
            print("To fix this:")
            print("  1. Initialize the submodule: git submodule update --init --recursive")
            print("  2. Build the frontend: ./build_frontend.sh")
            print("     (or: cd frontend && npm install && npm run build)")
        elif frontend_dir.exists():
            # Frontend exists but dist doesn't
            print("\nTo build the frontend:")
            print("  1. Navigate to the frontend directory")
            print("  2. Run: npm install && npm run build")
            print("     (or from repo root: ./build_frontend.sh)")
        else:
            # No frontend directory - probably installed via pip without frontend
            print("\nIf you installed via pip, the frontend should be included.")
            print("If you're in a development clone:")
            print("  1. Initialize submodules: git submodule update --init --recursive")
            print("  2. Build the frontend: ./build_frontend.sh")
        
        sys.exit(1)

    if daemon:
        # Run in background
        log_file = get_log_file()
        pid_file = get_pid_file()

        print(f"Starting StringSight UI in background...")
        print("Note: First startup can take 1–2 minutes (installing/loading components).")
        print(f"Access at: http://{host}:{port}")
        print(f"Logs: {log_file}")
        print(f"Workers: {workers}")
        if debug:
            print(f"Debug mode: enabled")

        # Create the app creation script
        app_script = f"""
import sys
from pathlib import Path
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import the main API app
from stringsight.api import app as api_app

# Use the API app directly
app = api_app

# Frontend dist path
dist_path = Path("{dist_path}")

# Serve static files
app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

# Add root route
@app.get("/")
async def root():
    return FileResponse(dist_path / "index.html")

# Exception handler to serve SPA for 404s on non-API routes
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    # If it's a 404 and not an API route, serve the SPA
    if exc.status_code == 404 and not request.url.path.startswith("/api"):
        # Check if it's a static file
        file_path = dist_path / request.url.path.lstrip("/")
        if file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        return FileResponse(dist_path / "index.html")

    # For API routes or other errors, return the original exception
    return JSONResponse(
        status_code=exc.status_code,
        content={{"detail": exc.detail}}
    )
"""

        # Write app to temp file
        app_file = get_config_dir() / "app.py"
        with open(app_file, 'w') as f:
            f.write(app_script)

        # Start uvicorn in background
        log_level = "debug" if debug else "info"
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app:app",
            "--host", host,
            "--port", str(port),
            "--log-level", log_level,
            "--workers", str(workers)
        ]

        # Open log file
        with open(log_file, 'a') as log:
            log.write(f"\n{'='*80}\n")
            log.write(f"Server started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write(f"{'='*80}\n\n")
            log.flush()

            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=str(get_config_dir()),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Detach from parent
            )

        # Write PID file
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        # Wait a bit to make sure it started
        time.sleep(2)

        # Check if still running
        if is_server_running():
            print(f"\n✓ Server started successfully (PID: {process.pid})")
            print(f"\nTo stop: stringsight stop")
            print(f"To view logs: stringsight logs")
        else:
            print(f"\n✗ Server failed to start. Check logs: {log_file}")
            sys.exit(1)
    else:
        # Run in foreground
        print(f"Starting StringSight UI...")
        print("Note: First startup can take 1–2 minutes (installing/loading components).")
        print(f"Access at: http://{host}:{port}")
        if debug:
            print(f"Debug mode: enabled")
        if workers > 1:
            print(f"Workers: {workers}")
        print("\nPress Ctrl+C to stop\n")

        # Create a combined FastAPI app that serves both API and static files
        import uvicorn
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        # Import the main API app
        from stringsight.api import app as api_app

        # Create a wrapper app that handles both API and frontend
        from fastapi import FastAPI, Request
        from fastapi.exceptions import HTTPException
        from starlette.exceptions import HTTPException as StarletteHTTPException

        # Create a fresh root app
        app = FastAPI()
        
        # Mount the backend API at /api so that /api/prompts -> api_app.get("/prompts")
        app.mount("/api", api_app)

        # Serve static files
        app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

        # Add root route for serving index.html
        @app.get("/")
        async def root():
            return FileResponse(dist_path / "index.html")

        # Exception handler to serve SPA for 404s on non-API routes
        @app.exception_handler(StarletteHTTPException)
        async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
            # If it's a 404 and not an API route, serve the SPA
            if exc.status_code == 404 and not request.url.path.startswith("/api"):
                # Check if it's a static file
                file_path = dist_path / request.url.path.lstrip("/")
                if file_path.is_file():
                    return FileResponse(file_path)
                # Otherwise serve index.html for SPA routing
                return FileResponse(dist_path / "index.html")

            # For API routes or other errors, return the original exception
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )

        # Start the server with appropriate log level
        log_level = "debug" if debug else "info"
        try:
            uvicorn.run(app, host=host, port=port, log_level=log_level, workers=workers)
        except KeyboardInterrupt:
            print("\nShutting down...")

def stop():
    """Stop the background server."""
    pid = is_server_running()
    if not pid:
        print("Server is not running")
        return

    print(f"Stopping server (PID: {pid})...")

    try:
        # Send SIGTERM
        os.kill(pid, signal.SIGTERM)

        # Wait for process to stop (max 10 seconds)
        for i in range(10):
            time.sleep(1)
            if not is_server_running():
                print("✓ Server stopped successfully")
                get_pid_file().unlink(missing_ok=True)
                return

        # If still running, force kill
        print("Server not responding, forcing shutdown...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        print("✓ Server stopped")
        get_pid_file().unlink(missing_ok=True)
    except ProcessLookupError:
        print("Server process not found")
        get_pid_file().unlink(missing_ok=True)
    except PermissionError:
        print(f"Error: No permission to stop process {pid}")
        sys.exit(1)

def status():
    """Check server status."""
    pid = is_server_running()
    if pid:
        print(f"✓ Server is running (PID: {pid})")
        print(f"Logs: {get_log_file()}")
        return 0
    else:
        print("✗ Server is not running")
        return 1

def logs(follow: bool = False, lines: int = 50):
    """View server logs."""
    log_file = get_log_file()

    if not log_file.exists():
        print("No log file found")
        return

    if follow:
        # Tail -f equivalent
        print(f"Following {log_file} (Ctrl+C to stop)...")
        try:
            with open(log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        print(line, end='')
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped following logs")
    else:
        # Show last N lines
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    print(line, end='')
        except Exception as e:
            print(f"Error reading log file: {e}")

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
    launch_parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in background"
    )
    launch_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, recommended: 4)"
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the background server"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check if server is running"
    )

    # Logs command
    logs_parser = subparsers.add_parser(
        "logs",
        help="View server logs"
    )
    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow log output"
    )
    logs_parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=50,
        help="Number of lines to show (default: 50)"
    )

    args = parser.parse_args()

    if args.command == "launch":
        launch(
            host=args.host,
            port=args.port,
            debug=args.debug,
            daemon=args.daemon,
            workers=args.workers
        )
    elif args.command == "stop":
        stop()
    elif args.command == "status":
        sys.exit(status())
    elif args.command == "logs":
        logs(follow=args.follow, lines=args.lines)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
