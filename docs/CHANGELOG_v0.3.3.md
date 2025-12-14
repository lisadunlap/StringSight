# StringSight v0.3.3 Release Notes

## New Features

### 🚀 Web UI with `stringsight launch` Command

Users can now launch the StringSight web interface with a simple command:

```bash
pip install stringsight
stringsight launch
```

The web UI provides an interactive interface for:
- Uploading and analyzing datasets
- Viewing extracted properties and clusters
- Exploring model behavior patterns
- Comparing models side-by-side

### 🔧 Daemon Mode for Background Operation

New daemon mode allows running the server in the background:

```bash
# Run in background with 4 workers
stringsight launch --daemon --workers 4

# Check status
stringsight status

# View logs
stringsight logs

# Stop server
stringsight stop
```

**Benefits:**
- Survives terminal disconnects
- Multiple workers for concurrent request handling
- Process management with PID files
- Logs saved to `~/.stringsight/logs/server.log`

### 📦 Frontend Included in Package

The web frontend is now included directly in the pip package. No need for:
- Separate frontend installation
- Docker (for simple use cases)
- Manual building (for end users)

### 🛠️ New CLI Commands

- `stringsight launch` - Start the web UI (foreground or background)
- `stringsight stop` - Stop background server
- `stringsight status` - Check if server is running
- `stringsight logs` - View server logs
  - `stringsight logs --follow` - Tail logs in real-time

## Technical Details

### Package Structure

- Frontend built files included in `stringsight/frontend_dist/`
- Single-process server combining FastAPI backend + static file serving
- Configuration stored in `~/.stringsight/`

### Architecture

**Foreground Mode:**
- Single uvicorn process
- Logs to stdout
- Ctrl+C to stop

**Daemon Mode:**
- Detached uvicorn process
- Multiple workers via `--workers N`
- PID file for process tracking
- Logs to file

### Dependencies

No new external dependencies required. All features work with:
- Python 3.8+
- FastAPI (already required)
- uvicorn (already required)

## Migration Guide

### For Existing Users

No changes needed! All existing code and APIs continue to work. The new CLI commands are additive.

### For New Users

The simplest way to get started is now:

```bash
pip install stringsight
stringsight launch
```

Open http://localhost:5180 and start analyzing!

## Deployment Options

### Simple (New Default)
```bash
stringsight launch --daemon --workers 4
```

### Docker (Still Supported)
```bash
docker compose up -d
```

### Python API (Still Supported)
```python
from stringsight import explain
result = explain(df, property="...")
```

## Files Changed

### New Files
- `stringsight/cli.py` - CLI implementation
- `stringsight/__main__.py` - Python module entry point
- `stringsight/frontend_dist/` - Built frontend (gitignored)
- `release.sh` - Automated release script
- `DEPLOYMENT_OPTIONS.md` - Deployment guide

### Modified Files
- `pyproject.toml` - Added CLI entry point, package data config
- `MANIFEST.in` - Include frontend dist files
- `README.md` - Added Quick Start, Web UI documentation
- `.gitignore` - Exclude frontend build artifacts

### Documentation
- `RELEASE.md` - Release process documentation
- `CHANGELOG_v0.3.3.md` - This file

## Known Issues

None currently. Please report issues at: https://github.com/lisadunlap/stringsight/issues

## What's Next (v0.4.0)

Potential features for next release:
- Auto-restart on crash
- Log rotation
- Configuration file support
- Custom frontend port
- SSL/TLS support
- Authentication for multi-user setups

## Credits

- Frontend: Separate repo at https://github.com/lisadunlap/stringsight-frontend
- Backend: Main StringSight repo

---

**Full Changelog**: https://github.com/lisadunlap/stringsight/compare/v0.3.2...v0.3.3
