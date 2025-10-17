#!/usr/bin/env python3
"""
Run StringSight pipeline for Capabilities datasets via YAML config.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_from_config import main as run_from_config_main


DEFAULT_CONFIG = str(Path(__file__).parent / "dataset_configs" / "capabilities.yaml")


def main():
    parser = argparse.ArgumentParser(description="Run Capabilities analysis (config-driven)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config YAML")
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0], "--config", args.config, *unknown]
    return run_from_config_main()


if __name__ == "__main__":
    main()


