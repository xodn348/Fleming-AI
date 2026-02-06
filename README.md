# Fleming-AI

Intelligent research paper collection and analysis system powered by AI.

## Overview

Fleming-AI is a comprehensive system designed to automatically collect, filter, validate, and analyze research papers from various sources. It leverages AI to intelligently process academic content and organize it for further analysis.

## Features

- **Paper Collection**: Automated collection from multiple academic sources
- **Intelligent Filtering**: AI-powered filtering of relevant papers
- **Validation**: Comprehensive validation of paper metadata and content
- **Generation**: Generate summaries and insights from collected papers
- **Storage**: Efficient storage and retrieval of paper data
- **Scheduling**: Automated scheduling of collection and processing tasks

## Project Structure

```
Fleming-AI/
├── src/
│   ├── collectors/      # Paper collection modules
│   ├── filters/         # Filtering logic
│   ├── generators/      # Content generation
│   ├── validators/      # Validation logic
│   ├── storage/         # Data storage and retrieval
│   └── scheduler/       # Task scheduling
├── tests/               # Test suite
├── data/
│   ├── papers/          # Collected papers
│   ├── db/              # Database files
│   └── output/          # Generated outputs
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

## Installation

### Prerequisites
- Python 3.10 or higher
- uv (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/xodn348/Fleming-AI.git
cd Fleming-AI

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_example.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Configuration

Create a `.env` file in the project root:

```env
# Example configuration
DEBUG=True
LOG_LEVEL=INFO
```

## Usage

```python
from src.collectors import PaperCollector
from src.filters import PaperFilter

# Initialize collector
collector = PaperCollector()

# Collect papers
papers = collector.collect()

# Filter papers
filter = PaperFilter()
filtered = filter.apply(papers)
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a feature branch
2. Make your changes
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Authors

Fleming-AI Team

## Support

For issues and questions, please open an issue on GitHub.
