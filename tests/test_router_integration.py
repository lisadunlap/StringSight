#!/usr/bin/env python3
"""
Quick test to verify router integration works.

This script imports the api.py app and checks that routers are properly registered.
"""

from stringsight.api import app

def test_router_integration():
    """Check that all routers are registered."""

    # Get all registered routes
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append((route.path, route.methods))

    # Expected endpoints from our routers
    expected_endpoints = [
        # Health router
        '/health',
        '/api/health',
        '/embedding-models',
        '/debug',

        # DataFrame router
        '/df/select',
        '/df/groupby/preview',
        '/df/custom',

        # Prompts router
        '/prompts',
        '/prompt-text',
        '/label/prompt',

        # Explain router
        '/api/explain/side-by-side',
        '/explain/side-by-side',

        # Validation router
        '/detect-and-validate',
        '/conversations',
        '/read-path',
        '/list-path',
        '/results/load',

        # Extraction router
        '/label/run',
        '/extract/single',
        '/extract/batch',
        '/extract/jobs/start',
        '/extract/jobs/status',
        '/extract/jobs/result',
        '/extract/jobs/cancel',
        '/extract/stream',

        # Clustering router
        '/cluster/run',
        '/cluster/metrics',
    ]

    registered_paths = [r[0] for r in routes]

    print("=" * 60)
    print("ROUTER INTEGRATION TEST")
    print("=" * 60)
    print(f"\nTotal routes registered: {len(routes)}")

    # Check which expected endpoints are registered
    found = []
    missing = []
    duplicates = []

    for endpoint in expected_endpoints:
        count = registered_paths.count(endpoint)
        if count > 0:
            found.append(endpoint)
            if count > 1:
                duplicates.append((endpoint, count))
        else:
            missing.append(endpoint)

    print(f"\n✅ Found {len(found)}/{len(expected_endpoints)} expected endpoints")

    if duplicates:
        print(f"\n⚠️  WARNING: {len(duplicates)} endpoints have multiple registrations:")
        for path, count in duplicates:
            print(f"   - {path} ({count}x)")
        print("\n   This is expected during migration - old endpoints still exist in api.py")
        print("   The LAST registered handler will be used by FastAPI")

    if missing:
        print(f"\n❌ Missing {len(missing)} endpoints:")
        for path in missing:
            print(f"   - {path}")

    print("\n" + "=" * 60)

    if missing:
        print("❌ FAILED: Some endpoints are missing")
        return False
    else:
        print("✅ SUCCESS: All router endpoints are registered!")
        if duplicates:
            print("⚠️  Note: Some duplicates exist (old + new implementations)")
        return True


if __name__ == "__main__":
    success = test_router_integration()
    exit(0 if success else 1)
