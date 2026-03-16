"""
Diagnostic script to check if role metadata is being properly added to clusters
"""
import json
import sys
from pathlib import Path

def check_cluster_metadata(results_dir: str):
    """Check if clusters have role metadata"""
    results_path = Path(results_dir)

    # Check clusters file
    clusters_file = results_path / "clusters.jsonl"
    if not clusters_file.exists():
        print(f"❌ Clusters file not found: {clusters_file}")
        return

    print(f"✅ Found clusters file: {clusters_file}")
    print()

    # Read first 5 clusters
    with open(clusters_file) as f:
        clusters = [json.loads(line) for line in f]

    print(f"Total clusters: {len(clusters)}")
    print()

    # Check metadata structure
    for i, cluster in enumerate(clusters[:5]):
        print(f"Cluster {i+1}: {cluster.get('label', 'NO LABEL')[:60]}...")
        meta = cluster.get('meta', {})
        print(f"  Metadata keys: {list(meta.keys())}")
        print(f"  group: {meta.get('group', 'MISSING')}")
        print(f"  role: {meta.get('role', 'MISSING')}")
        print(f"  behavior_type: {meta.get('behavior_type', 'MISSING')}")
        print()

    # Check if any clusters have role metadata
    clusters_with_role = [c for c in clusters if 'role' in c.get('meta', {})]
    print(f"Clusters with role metadata: {len(clusters_with_role)} / {len(clusters)}")

    if clusters_with_role:
        print("\n✅ Some clusters have role metadata!")
        example = clusters_with_role[0]
        print(f"Example: {example.get('label', '')[:60]}...")
        print(f"  Metadata: {json.dumps(example.get('meta', {}), indent=2)}")
    else:
        print("\n❌ No clusters have role metadata!")
        print("This suggests role-based extraction was not enabled or didn't work correctly.")

    # Check properties file for role field
    print("\n" + "="*80)
    print("Checking properties file...")
    print("="*80)

    properties_file = results_path / "extracted_properties.jsonl"
    if not properties_file.exists():
        print(f"❌ Properties file not found: {properties_file}")
        return

    with open(properties_file) as f:
        properties = [json.loads(line) for line in f]

    print(f"Total properties: {len(properties)}")

    # Check if properties have role field
    properties_with_role = [p for p in properties if 'role' in p]
    print(f"Properties with role field: {len(properties_with_role)} / {len(properties)}")

    if properties_with_role:
        print("\n✅ Some properties have role field!")
        example = properties_with_role[0]
        print(f"Example property:")
        print(f"  role: {example.get('role')}")
        print(f"  behavior_type: {example.get('behavior_type')}")
        print(f"  description: {example.get('property_description', '')[:80]}...")
    else:
        print("\n❌ No properties have role field!")
        print("This is the root cause - properties were extracted without role information.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_role_metadata.py <results_directory>")
        print()
        print("Example:")
        print("  python diagnose_role_metadata.py ./results/my_analysis")
        sys.exit(1)

    results_dir = sys.argv[1]
    check_cluster_metadata(results_dir)
