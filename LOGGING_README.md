# Logging Configuration

StringSight now uses Python's logging module instead of print statements for better control over debug output.

## Usage

The logging system is automatically configured when you import any StringSight module. All logging output goes to stdout by default.

## Controlling Log Levels

You can control what level of logs are displayed using the `STRINGSIGHT_LOG_LEVEL` environment variable:

```bash
# Show only INFO and above (default)
export STRINGSIGHT_LOG_LEVEL=INFO
python your_script.py

# Show all debug messages
export STRINGSIGHT_LOG_LEVEL=DEBUG
python your_script.py

# Show only warnings and errors
export STRINGSIGHT_LOG_LEVEL=WARNING
python your_script.py

# Show only errors
export STRINGSIGHT_LOG_LEVEL=ERROR
python your_script.py
```

## Available Log Levels

- `DEBUG`: Detailed diagnostic information (formerly print statements with debug prefixes)
- `INFO`: General informational messages (formerly most print statements)
- `WARNING`: Warning messages about potential issues
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

## Custom Log Format

By default, StringSight uses a simple format that only shows the message (no timestamp or log level). You can customize this by setting the `STRINGSIGHT_LOG_FORMAT` environment variable:

```bash
# Add timestamps and log levels (useful for debugging)
export STRINGSIGHT_LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Show only log level and message
export STRINGSIGHT_LOG_FORMAT="[%(levelname)s] %(message)s"

# Show module name and message
export STRINGSIGHT_LOG_FORMAT="[%(name)s] %(message)s"
```

## Using Logging in Code

If you're extending StringSight, use the logging utility:

```python
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

## Third-Party Library Logs

By default, StringSight suppresses noisy logs from third-party libraries like LiteLLM, httpx, and openai. These are set to WARNING level to reduce clutter. If you need to see these logs for debugging, you can enable them:

```python
import logging

# Enable LiteLLM debug logs
logging.getLogger("LiteLLM").setLevel(logging.INFO)

# Enable httpx debug logs  
logging.getLogger("httpx").setLevel(logging.INFO)
```

## Migration Notes

- All print statements in core StringSight modules have been converted to logging statements
- Print statements with emoji prefixes like "🔍 DEBUG:" are now `logger.debug()` calls
- Print statements with "✅" or informational content are now `logger.info()` calls
- Print statements with "⚠️  WARNING:" are now `logger.warning()` calls
- Print statements with "❌" or error content are now `logger.error()` calls
- Embedding debug logs (`[emb-debug]`) are now `logger.debug()` calls (only visible with DEBUG level)

This change allows users to control verbosity without modifying code or using verbose flags.


