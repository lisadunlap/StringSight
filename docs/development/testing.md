# Testing Guide

Comprehensive guide to testing in StringSight.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_evaluation.py

# Run specific test function
pytest tests/test_evaluation.py::test_evaluate_model_basic
```

### Test Coverage

```bash
# Run with coverage report
pytest --cov=stringsight

# Generate HTML coverage report
pytest --cov=stringsight --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=stringsight --cov-report=xml
```

### Test Categories

```bash
# Run unit tests only
pytest -m "not integration"

# Run integration tests only
pytest -m integration

# Run slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"
```

## Writing Tests

### Test Structure

```python
import pytest
from stringsight.evaluation import evaluate_model

class TestEvaluation:
    """Test suite for evaluation functionality."""
    
    def test_basic_evaluation(self):
        """Test basic model evaluation."""
        # Arrange
        data = [{"question": "What is 2+2?", "answer": "4", "model_output": "4"}]
        
        # Act
        results = evaluate_model(data, metrics=["accuracy"])
        
        # Assert
        assert "accuracy" in results
        assert results["accuracy"] == 1.0
    
    def test_empty_data(self):
        """Test evaluation with empty data."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            evaluate_model([])
    
    @pytest.mark.parametrize("metric", ["accuracy", "bleu", "rouge"])
    def test_metric_computation(self, metric):
        """Test computation of different metrics."""
        data = [{"question": "Test", "answer": "answer", "model_output": "answer"}]
        results = evaluate_model(data, metrics=[metric])
        assert metric in results
```

### Test Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return [
        {"question": "What is 2+2?", "answer": "4", "model_output": "4"},
        {"question": "What is 3+3?", "answer": "6", "model_output": "6"}
    ]

@pytest.fixture
def evaluation_config():
    """Provide evaluation configuration."""
    return {
        "metrics": ["accuracy", "bleu"],
        "batch_size": 32,
        "save_results": False
    }

def test_evaluation_with_fixtures(sample_data, evaluation_config):
    """Test evaluation using fixtures."""
    results = evaluate_model(sample_data, config=evaluation_config)
    assert "accuracy" in results
    assert "bleu" in results
```

### Mocking

```python
from unittest.mock import patch, MagicMock

def test_external_api_call():
    """Test function that calls external API."""
    with patch('stringsight.external_api.call_api') as mock_api:
        mock_api.return_value = {"result": "success"}
        
        # Your test code here
        result = call_external_function()
        
        assert result == "success"
        mock_api.assert_called_once()
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation.

```python
def test_metric_computation():
    """Test metric computation logic."""
    from stringsight.metrics import AccuracyMetric
    
    metric = AccuracyMetric()
    predictions = ["4", "6", "8"]
    references = ["4", "6", "8"]
    
    score = metric.compute(predictions, references)
    assert score == 1.0
```

### Integration Tests

Test interactions between components.

```python
@pytest.mark.integration
def test_full_evaluation_pipeline():
    """Test complete evaluation pipeline."""
    # Load data
    data = load_test_dataset()
    
    # Run evaluation
    results = evaluate_model(data, metrics=["accuracy", "bleu"])
    
    # Save results
    save_results(results, "test_results.json")
    
    # Load and verify
    loaded_results = load_results("test_results.json")
    assert loaded_results == results
```

### Performance Tests

Test performance characteristics.

```python
@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance with large dataset."""
    import time
    
    # Generate large dataset
    large_data = generate_test_data(10000)
    
    start_time = time.time()
    results = evaluate_model(large_data, metrics=["accuracy"])
    end_time = time.time()
    
    # Should complete within reasonable time
    assert end_time - start_time < 60  # 60 seconds
```

## Test Data

### Creating Test Data

```python
def generate_test_data(num_samples: int = 100) -> List[Dict]:
    """Generate synthetic test data."""
    import random
    
    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain gravity",
        "What is photosynthesis?"
    ]
    
    data = []
    for i in range(num_samples):
        question = random.choice(questions)
        answer = f"Answer {i}"
        model_output = f"Model output {i}"
        
        data.append({
            "question": question,
            "answer": answer,
            "model_output": model_output,
            "metadata": {"id": i}
        })
    
    return data
```

### Test Data Files

Store test data in `tests/data/`:

```
tests/
├── data/
│   ├── sample.jsonl
│   ├── large_dataset.jsonl
│   └── edge_cases.jsonl
├── test_evaluation.py
└── test_metrics.py
```

## Assertions and Checks

### Basic Assertions

```python
def test_basic_assertions():
    """Test basic assertion patterns."""
    results = evaluate_model(sample_data)
    
    # Check key exists
    assert "accuracy" in results
    
    # Check value range
    assert 0.0 <= results["accuracy"] <= 1.0
    
    # Check type
    assert isinstance(results["accuracy"], float)
    
    # Check approximate equality
    assert results["accuracy"] == pytest.approx(0.85, rel=0.01)
```

### Custom Assertions

```python
def assert_valid_results(results: Dict):
    """Custom assertion for result validation."""
    required_keys = ["accuracy", "bleu", "rouge"]
    
    for key in required_keys:
        assert key in results, f"Missing key: {key}"
        assert isinstance(results[key], (int, float)), f"Invalid type for {key}"
        assert 0.0 <= results[key] <= 1.0, f"Value out of range for {key}"

def test_results_validation():
    """Test custom result validation."""
    results = evaluate_model(sample_data)
    assert_valid_results(results)
```

## Error Testing

### Testing Exceptions

```python
def test_invalid_input():
    """Test handling of invalid input."""
    with pytest.raises(ValueError, match="Data cannot be empty"):
        evaluate_model([])
    
    with pytest.raises(TypeError):
        evaluate_model("not a list")
    
    with pytest.raises(KeyError):
        evaluate_model([{"invalid": "data"}])
```

### Testing Warnings

```python
import warnings

def test_deprecation_warning():
    """Test deprecation warnings."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        deprecated_function()
```

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### conftest.py

```python
import pytest
import tempfile
import os

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set test environment variables
    os.environ["stringsight_TESTING"] = "true"
    yield
    # Cleanup
    if "stringsight_TESTING" in os.environ:
        del os.environ["stringsight_TESTING"]
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=stringsight --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Best Practices

1. **Test Naming**: Use descriptive test names
2. **Test Isolation**: Each test should be independent
3. **Fast Tests**: Keep unit tests fast (< 1 second)
4. **Coverage**: Aim for high code coverage
5. **Documentation**: Document complex test scenarios
6. **Maintenance**: Keep tests up to date with code changes

## Next Steps

- Check out [Contributing](contributing.md) for development guidelines
- Read the [API Reference](../api/core.md) to understand the codebase
- Look at [Basic Usage](../user-guide/basic-usage.md) for usage examples 