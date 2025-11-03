#!/usr/bin/env python3
"""
Script to run StringSight analysis on all JSON files in data/openai_exports
using the tony2.yaml config as a template.
"""

import os
import subprocess
import tempfile
import yaml
from pathlib import Path

def main():
    # Base config file
    base_config_path = "scripts/dataset_configs/tony2.yaml"
    
    # Directories
    data_dir = Path("data/openai_exports")
    results_base_dir = Path("results")
    
    # Read the base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Get only JSON files ending with __easy or __hard
    all_json_files = list(data_dir.glob("*.json"))
    json_files = [f for f in all_json_files if (f.stem.endswith("__easy") or f.stem.endswith("__hard"))]
    
    if not json_files:
        print("No JSON files found in data/openai_exports")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        # Create output directory name based on the JSON file name (without extension)
        output_dir_name = json_file.stem
        output_dir = results_base_dir / output_dir_name
        
        # Create modified config
        config = base_config.copy()
        config['data_path'] = str(json_file)
        config['output_dir'] = str(output_dir)
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            yaml.dump(config, temp_file, default_flow_style=False)
            temp_config_path = temp_file.name
        
        try:
            # Run the command
            cmd = [
                "python", "scripts/run_from_config.py", 
                "--config", temp_config_path
            ]
            
            # Set environment variable
            env = os.environ.copy()
            env['STRINGSIGHT_DISABLE_CACHE'] = '1'
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully processed {json_file.name}")
            else:
                print(f"✗ Error processing {json_file.name}")
                print(f"Error output: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Exception processing {json_file.name}: {e}")
            
        finally:
            # Clean up temporary config file
            os.unlink(temp_config_path)
    
    print(f"\nCompleted processing all {len(json_files)} files")

if __name__ == "__main__":
    main()
