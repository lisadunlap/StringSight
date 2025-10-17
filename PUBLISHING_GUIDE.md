# Publishing StringSight to PyPI

This guide walks you through publishing the StringSight package to PyPI.

## Prerequisites

1. **Create PyPI accounts:**
   - Production: https://pypi.org/account/register/
   - Test (recommended first): https://test.pypi.org/account/register/

2. **Install build tools:**
   ```bash
   pip install --upgrade pip
   pip install build twine
   ```

## Step-by-Step Publishing Process

### 1. Clean Previous Builds

Remove any old build artifacts:
```bash
rm -rf build/ dist/ *.egg-info
```

### 2. Build the Package

Build both source distribution and wheel:
```bash
python -m build
```

This creates:
- `dist/stringsight-0.1.0.tar.gz` (source distribution)
- `dist/stringsight-0.1.0-py3-none-any.whl` (wheel)

### 3. Test Your Package Locally (Optional but Recommended)

Install locally in editable mode to test:
```bash
pip install -e .
```

Or install from the built wheel:
```bash
pip install dist/stringsight-0.1.0-py3-none-any.whl
```

### 4. Upload to Test PyPI First

**Highly recommended** to test on Test PyPI before the real thing:

```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: your Test PyPI username (or `__token__` if using API token)
- Password: your Test PyPI password (or API token)

Test installation from Test PyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ stringsight
```

### 5. Upload to Production PyPI

Once you've verified everything works on Test PyPI:

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- Username: your PyPI username (or `__token__` if using API token)
- Password: your PyPI password (or API token)

### 6. Verify Installation

Test that users can install your package:
```bash
pip install stringsight
```

Or with extras:
```bash
pip install stringsight[viz]      # Visualization dependencies
pip install stringsight[ml]       # ML dependencies
pip install stringsight[full]     # All optional dependencies
pip install stringsight[dev]      # Development dependencies
```

## Using API Tokens (Recommended)

For security, use API tokens instead of passwords:

1. Go to PyPI Account Settings → API tokens
2. Create a new token with appropriate scope
3. Create a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

Then you can upload without being prompted:
```bash
python -m twine upload --repository testpypi dist/*  # For Test PyPI
python -m twine upload dist/*                         # For PyPI
```

## Version Updates

When releasing a new version:

1. **Update version number** in these files:
   - `pyproject.toml` (line 6)
   - `setup.py` (line 12)
   - `stringsight/__init__.py` (line 11)

2. **Update CHANGELOG.md** with new features/fixes

3. **Clean and rebuild:**
   ```bash
   rm -rf build/ dist/ *.egg-info
   python -m build
   ```

4. **Upload new version:**
   ```bash
   python -m twine upload dist/*
   ```

5. **Tag the release in git:**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

## Troubleshooting

### "File already exists" error
You cannot re-upload the same version. Increment your version number.

### Import errors after installation
Check that all subdirectories have `__init__.py` files and that `find_packages()` is finding them.

### Missing dependencies
Verify all required dependencies are listed in both `setup.py` and `pyproject.toml`.

### Package too large
PyPI has size limits. Consider:
- Excluding large data files
- Using `.gitignore` patterns
- Updating `MANIFEST.in` to exclude unnecessary files

## Best Practices

1. **Always test on Test PyPI first**
2. **Use semantic versioning:** MAJOR.MINOR.PATCH (e.g., 0.1.0 → 0.1.1 → 0.2.0)
3. **Keep a CHANGELOG.md** documenting changes
4. **Tag releases in git** matching version numbers
5. **Test installation** in a fresh virtual environment
6. **Document breaking changes** clearly
7. **Use API tokens** instead of passwords

## Quick Reference Commands

```bash
# Clean build
rm -rf build/ dist/ *.egg-info

# Build package
python -m build

# Check package before upload
twine check dist/*

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Test installation
pip install stringsight

# Install in development mode
pip install -e .
```

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PEP 517/518 - Build System](https://peps.python.org/pep-0517/)


