# Contributing to StringSight

Thank you for your interest in contributing to StringSight! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of Python and machine learning

### Setting Up Development Environment

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/stringsight.git
   cd stringsight
   ```

2. **Install Development Dependencies**
   ```bash
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   # Install pre-commit hooks
   pre-commit install
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### 2. Make Your Changes

- Write your code following the [Code Style](#code-style) guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the test suite
pytest

# Run with coverage
pytest --cov=stringsight

# Run linting
flake8 stringsight/
black stringsight/
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new evaluation metric"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template
5. Submit the PR

## Code Style

### Python Code

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Docstrings**: Google style
- **Type Hints**: Required for all public functions

### Example

```python
from typing import List, Dict, Optional

def evaluate_model(
    data: List[Dict],
    metrics: List[str] = ["accuracy"],
    config: Optional[Dict] = None
) -> Dict:
    """Evaluate model performance on given data.
    
    Args:
        data: List of dictionaries containing evaluation data
        metrics: List of metric names to compute
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing evaluation results
        
    Raises:
        EvaluationError: If evaluation fails
    """
    # Your implementation here
    pass
```

### Documentation

- All public functions must have docstrings
- Use Google style docstrings
- Include type hints
- Add examples for complex functions

### Testing

- Write tests for all new functionality
- Aim for at least 80% code coverage
- Use descriptive test names
- Test both success and failure cases

### Example Test

```python
import pytest
from stringsight.evaluation import evaluate_model

def test_evaluate_model_basic():
    """Test basic model evaluation functionality."""
    data = [
        {"question": "What is 2+2?", "answer": "4", "model_output": "4"}
    ]
    
    results = evaluate_model(data, metrics=["accuracy"])
    
    assert "accuracy" in results
    assert results["accuracy"] == 1.0

def test_evaluate_model_invalid_data():
    """Test evaluation with invalid data."""
    with pytest.raises(ValueError):
        evaluate_model([])
```

## Project Structure

```
stringsight/
├── stringsight/           # Main package
│   ├── __init__.py
│   ├── evaluation.py   # Core evaluation functions
│   ├── data.py         # Data loading and processing
│   ├── metrics.py      # Evaluation metrics
│   ├── visualization.py # Plotting and visualization
│   ├── config.py       # Configuration management
│   └── utils.py        # Utility functions
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Example scripts
└── requirements.txt    # Dependencies
```

## Adding New Features

### 1. New Metrics

To add a new evaluation metric:

1. Create the metric class in `stringsight/metrics.py`
2. Inherit from the `Metric` base class
3. Implement the `compute` method
4. Add tests in `tests/test_metrics.py`
5. Update documentation

### 2. New Data Formats

To add support for new data formats:

1. Add format detection in `stringsight/data.py`
2. Implement loading/saving functions
3. Add validation logic
4. Write tests
5. Update documentation

### 3. New Visualization Types

To add new visualization types:

1. Add plotting functions in `stringsight/visualization.py`
2. Follow the existing API patterns
3. Add configuration options
4. Write tests
5. Update documentation

## Bug Reports

When reporting bugs, please include:

1. **Environment**: Python version, OS, package versions
2. **Reproduction**: Steps to reproduce the issue
3. **Expected vs Actual**: What you expected vs what happened
4. **Error Messages**: Full error traceback
5. **Minimal Example**: Code that reproduces the issue

## Feature Requests

When requesting features, please include:

1. **Use Case**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: What other approaches have you considered?
4. **Implementation**: Any thoughts on implementation?

## Code Review Process

1. **Automated Checks**: All PRs must pass CI checks
2. **Review**: At least one maintainer must approve
3. **Tests**: All tests must pass
4. **Documentation**: Documentation must be updated
5. **Style**: Code must follow style guidelines

## Release Process

### For Maintainers

1. **Update Version**: Update version in `setup.py`
2. **Update Changelog**: Add release notes
3. **Create Release**: Tag and create GitHub release
4. **Publish**: Upload to PyPI

### Version Numbers

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check the docs first
- **Code**: Look at existing code for examples

## Recognition

Contributors will be recognized in:

- GitHub contributors list
- Release notes
- Documentation acknowledgments

## Next Steps

- Check out the [Testing Guide](testing.md) for detailed testing information
- Read the [API Reference](../api/core.md) to understand the codebase
- Look at [Basic Usage](../user-guide/basic-usage.md) for usage examples 