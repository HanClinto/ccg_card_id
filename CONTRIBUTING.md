# Contributing to CCG Card ID

Thank you for your interest in contributing to CCG Card ID! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/HanClinto/ccg_card_id.git
cd ccg_card_id
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ccg_card_id --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## Code Style

We follow PEP 8 style guidelines. Use the following tools to maintain code quality:

```bash
# Format code with black
black ccg_card_id/ tests/ scripts/

# Sort imports with isort
isort ccg_card_id/ tests/ scripts/

# Check for style issues with flake8
flake8 ccg_card_id/ tests/ scripts/

# Type checking with mypy
mypy ccg_card_id/
```

## Adding New Features

### Adding a New Dataset Fetcher

1. Create a new fetcher class in `ccg_card_id/dataset/`
2. Implement the following methods:
   - `fetch_all_cards()`: Fetch all cards from the API
   - `download_card_images()`: Download card images
   - `save_metadata()`: Save card metadata
   - `fetch_sample_dataset()`: Fetch a sample dataset

3. Add the fetcher to `ccg_card_id/dataset/__init__.py`
4. Update the `download_dataset.py` script to support the new game
5. Add tests in `tests/test_dataset.py`

### Adding a New Model

1. Create a new model class in `ccg_card_id/models/`
2. Implement the following methods:
   - `build_gallery()`: Build a gallery of reference images
   - `find_matches()`: Find matching cards for a query
   - `compute_similarity_matrix()`: Compute similarity scores
   - `save_gallery()` / `load_gallery()`: Persistence

3. Add the model to `ccg_card_id/models/__init__.py`
4. Create a testing script in `scripts/test_<model_name>.py`
5. Add tests in `tests/test_models.py`

### Adding a New Benchmark Task

1. Create a new task class in `ccg_card_id/benchmark/tasks.py`
2. Implement:
   - Data preparation methods (e.g., `create_pairs()`, `create_query_gallery()`)
   - `evaluate()`: Compute metrics for the task

3. Add the task to `ccg_card_id/benchmark/__init__.py`
4. Add examples to `EXAMPLES.md`
5. Add tests in `tests/test_benchmark.py`

## Submitting Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Description of your changes"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Create a pull request on GitHub

## Pull Request Guidelines

- Include a clear description of the changes
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style
- Keep changes focused and atomic

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Error messages or logs

## Feature Requests

We welcome feature requests! Please:

- Check if the feature has already been requested
- Clearly describe the feature and its use case
- Explain how it fits into the project's goals

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment
- Report unacceptable behavior to the maintainers

## Questions?

If you have questions about contributing, feel free to:

- Open an issue on GitHub
- Check existing issues and discussions
- Review the documentation

Thank you for contributing to CCG Card ID!
