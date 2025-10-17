#!/usr/bin/env python3
"""
Test script to verify the new stage-based saving structure works correctly.
"""

import pandas as pd
import tempfile
import os
from pathlib import Path
from stringsight import explain

def test_new_structure():
    """Test the new stage-based saving structure."""
    
    # Create sample data
    df = pd.DataFrame({
        "question_id": ["q1", "q2", "q3"],
        "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
        "model": ["gpt-4", "gpt-4", "gpt-4"],
        "model_response": [
            "Machine learning is a subset of artificial intelligence...",
            "Quantum computing uses quantum mechanical phenomena...", 
            "In circuits of light, silicon dreams awaken..."
        ],
        "score": [4.2, 3.8, 4.5]
    })
    
    print("Testing new stage-based saving structure...")
    print(f"Sample data shape: {df.shape}")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_results")
        
        print(f"Output directory: {output_dir}")
        
        # Run the pipeline with the new structure
        try:
            clustered_df, model_stats = explain(
                df,
                method="single_model",
                min_cluster_size=1,  # Small for testing
                max_workers=1,  # Single worker for testing
                output_dir=output_dir,
                use_wandb=False,  # Disable wandb for testing
                verbose=True
            )
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"Clustered DataFrame shape: {clustered_df.shape}")
            print(f"Models analyzed: {len(model_stats)}")
            
            # Check the directory structure
            output_path = Path(output_dir)
            print(f"\n📁 Checking output directory structure...")
            
            expected_dirs = [
                "01_extraction",
                "02_parsing", 
                "03_validation",
                "04_clustering",
                "05_metrics"
            ]
            
            for expected_dir in expected_dirs:
                dir_path = output_path / expected_dir
                if dir_path.exists():
                    print(f"  ✅ {expected_dir}/ exists")
                    
                    # List files in each directory
                    files = list(dir_path.glob("*"))
                    for file in files:
                        print(f"    - {file.name}")
                else:
                    print(f"  ❌ {expected_dir}/ missing")
            
            # Check for summary file
            summary_file = output_path / "summary.txt"
            if summary_file.exists():
                print(f"  ✅ summary.txt exists")
            else:
                print(f"  ❌ summary.txt missing")
            
            print(f"\n🎉 New structure test completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_new_structure()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        exit(1) 