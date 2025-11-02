# Tests

This directory contains tests for the LLM training project.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_models.py           # Model architecture tests
├── test_config.py           # Configuration tests
├── test_errors.py           # Error handling tests
└── test_integration.py      # Integration tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_models.py
```

### Run specific test
```bash
pytest tests/test_models.py::TestGPTModel::test_model_creation
```

### Run with markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"
```

### Run with coverage
```bash
pytest --cov=. --cov-report=html
```

## Test Categories

### Unit Tests
- Fast, isolated tests
- Test individual components
- No external dependencies
- Marked with `@pytest.mark.unit`

### Integration Tests
- Test complete workflows
- May be slower
- Test component interactions
- Marked with `@pytest.mark.integration`

### Slow Tests
- Tests that take significant time
- Marked with `@pytest.mark.slow`
- Skip with: `pytest -m "not slow"`

### GPU Tests
- Tests that require GPU
- Marked with `@pytest.mark.gpu`
- Skip with: `pytest -m "not gpu"`

## Writing Tests

### Test Naming
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Using Fixtures
```python
def test_example(temp_dir, sample_config):
    # temp_dir and sample_config are fixtures from conftest.py
    pass
```

### Parametrized Tests
```python
@pytest.mark.parametrize("value", [1, 2, 3])
def test_with_params(value):
    assert value > 0
```

### Marking Tests
```python
@pytest.mark.unit
def test_fast():
    pass

@pytest.mark.slow
@pytest.mark.integration
def test_slow_integration():
    pass
```

## CI/CD Integration

Tests are automatically run in CI/CD pipeline on:
- Pull requests
- Commits to main branch
- Nightly builds

## Troubleshooting

### Tests fail with import errors
- Make sure you're in the project root
- Install dependencies: `pip install -r requirements.txt`
- Install test dependencies: `pip install pytest pytest-cov`

### GPU tests fail
- Skip GPU tests if no GPU available: `pytest -m "not gpu"`
- Or run on CPU: `CUDA_VISIBLE_DEVICES="" pytest`

### Slow tests timeout
- Increase timeout: `pytest --timeout=600`
- Or skip slow tests: `pytest -m "not slow"`
