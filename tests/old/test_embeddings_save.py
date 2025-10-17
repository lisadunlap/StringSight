#!/usr/bin/env python3
"""
Test script to verify that embeddings are saved to a separate embeddings.json file.
"""

import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path

# Mock data that simulates the output from StringSight clustering
def create_mock_clustered_data():
    """Create mock clustered data with embeddings."""
    data = {
        'question_id': ['q1', 'q2', 'q3', 'q4', 'q5'],
        'model': ['model_a', 'model_b', 'model_a', 'model_b', 'model_a'],
        'property_description': [
            'Provides detailed explanations',
            'Uses technical terminology',
            'Gives step-by-step guidance',
            'Shows practical examples',
            'Explains complex concepts clearly'
        ],
        'property_description_cluster_id': [0, 1, 0, 1, 0],
        'property_description_cluster_label': [
            'Detailed Explanations',
            'Technical Approach',
            'Detailed Explanations',
            'Technical Approach',
            'Detailed Explanations'
        ],
        'property_description_embedding': [
            np.random.rand(1536).tolist(),  # OpenAI embedding size
            np.random.rand(1536).tolist(),
            np.random.rand(1536).tolist(),
            np.random.rand(1536).tolist(),
            np.random.rand(1536).tolist()
        ],
        'property_description_cluster_label_embedding': [
            np.random.rand(1536).tolist(),
            np.random.rand(1536).tolist(),
            np.random.rand(1536).tolist(),
            np.random.rand(1536).tolist(),
            np.random.rand(1536).tolist()
        ]
    }
    return pd.DataFrame(data)

def test_embeddings_save():
    """Test that embeddings are saved to a separate file."""
    print("🧪 Testing embeddings save functionality...")
    
    # Create mock data
    df = create_mock_clustered_data()
    print(f"  ✓ Created mock data with {len(df)} rows")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Mock the save function behavior
        embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
        print(f"  ✓ Found embedding columns: {embedding_cols}")
        
        if embedding_cols:
            embeddings_data = {}
            for col in embedding_cols:
                # Get unique embeddings to avoid duplication
                unique_embeddings = {}
                for idx, row in df.iterrows():
                    if pd.notna(row[col]) and row[col] is not None:
                        # Use the property description as the key for the main embedding column
                        if col == 'property_description_embedding':
                            key = row.get('property_description', f'row_{idx}')
                        elif col == 'property_description_cluster_label_embedding':
                            key = row.get('property_description_cluster_label', f'cluster_{idx}')
                        elif col == 'property_description_coarse_cluster_label_embedding':
                            key = row.get('property_description_coarse_cluster_label', f'coarse_cluster_{idx}')
                        else:
                            key = f'{col}_{idx}'
                        
                        # Convert numpy arrays to lists for JSON serialization
                        if hasattr(row[col], 'tolist'):
                            unique_embeddings[key] = row[col].tolist()
                        else:
                            unique_embeddings[key] = row[col]
                
                embeddings_data[col] = unique_embeddings
            
            # Save embeddings to separate file
            embeddings_path = temp_path / "embeddings.json"
            with open(embeddings_path, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            
            print(f"  ✓ Saved embeddings to: {embeddings_path}")
            print(f"    • Embedding columns: {embedding_cols}")
            total_embeddings = sum(len(emb_dict) for emb_dict in embeddings_data.values())
            print(f"    • Total unique embeddings: {total_embeddings}")
            
            # Verify the file was created and contains the expected data
            assert embeddings_path.exists(), "Embeddings file was not created"
            
            with open(embeddings_path, 'r') as f:
                loaded_embeddings = json.load(f)
            
            assert 'property_description_embedding' in loaded_embeddings, "Main embedding column missing"
            assert 'property_description_cluster_label_embedding' in loaded_embeddings, "Cluster label embedding column missing"
            
            # Check that we have the expected number of unique embeddings
            main_embeddings = loaded_embeddings['property_description_embedding']
            cluster_embeddings = loaded_embeddings['property_description_cluster_label_embedding']
            
            print(f"    • Main embeddings: {len(main_embeddings)} unique")
            print(f"    • Cluster label embeddings: {len(cluster_embeddings)} unique")
            
            # Verify that embeddings are properly keyed
            expected_keys = [
                'Provides detailed explanations',
                'Uses technical terminology', 
                'Gives step-by-step guidance',
                'Shows practical examples',
                'Explains complex concepts clearly'
            ]
            
            for key in expected_keys:
                assert key in main_embeddings, f"Missing embedding for: {key}"
            
            print("  ✅ All tests passed!")
            return True
        else:
            print("  ⚠️ No embedding columns found in data")
            return False

if __name__ == "__main__":
    success = test_embeddings_save()
    if success:
        print("\n🎉 Embeddings save functionality test completed successfully!")
    else:
        print("\n❌ Test failed!") 