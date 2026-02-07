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
- Ollama (for LLM inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/xodn348/Fleming-AI.git
cd Fleming-AI

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .

# Install and start Ollama (if not already installed)
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
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

### Running Fleming-AI

**Continuous Mode** (runs forever with automatic cycles):
```bash
python main.py
```

**Test Mode** (single short cycle for testing):
```bash
python main.py --test
```

**Custom Configuration**:
```bash
# Run with custom cycle delay (in seconds)
python main.py --cycle-delay 7200

# Run with custom retry settings
python main.py --max-retries 5
```

### Command Line Options

- `--test`: Run in test mode (single cycle with minimal data)
- `--cycle-delay SECONDS`: Time between cycles (default: 3600)
- `--max-retries N`: Maximum retry attempts on failure (default: 3)

### Pipeline Stages

Fleming-AI runs through 5 stages in each cycle:

1. **Paper Collection**: Fetches recent papers from arXiv
2. **Hypothesis Generation**: Uses LLM to generate hypotheses from papers
3. **Validation**: Validates hypotheses using computational methods
4. **Storage**: Stores results in SQLite database
5. **Synchronization**: Syncs data (placeholder for cloud backup)

### Programmatic Usage

```python
from src.scheduler.runner import FlemingRunner
import asyncio

async def main():
    runner = FlemingRunner(
        cycle_delay=3600,
        max_retries=3,
        test_mode=False,
    )
    
    # Run single cycle
    success = await runner.run_once()
    
    # Or run continuously
    await runner.run_forever()
    
    # Cleanup
    await runner.cleanup()

asyncio.run(main())
```

### Component Usage

**Collect Papers from arXiv**:
```python
from src.collectors.arxiv_client import ArxivClient

with ArxivClient() as client:
    papers = client.search(
        query="cat:cs.AI",
        max_results=10,
    )
```

**Generate Hypotheses**:
```python
from src.llm.ollama_client import OllamaClient
from src.generators.hypothesis import HypothesisGenerator
from src.storage.vectordb import VectorDB
from src.filters.quality import QualityFilter

async with OllamaClient() as ollama:
    vectordb = VectorDB()
    quality_filter = QualityFilter()
    
    generator = HypothesisGenerator(
        ollama_client=ollama,
        vectordb=vectordb,
        quality_filter=quality_filter,
    )
    
    hypotheses = await generator.generate_hypotheses(
        query="machine learning",
        k=10,
    )
```

**Validate Hypotheses**:
```python
from src.validators.pipeline import ValidationPipeline
from src.storage.hypothesis_db import HypothesisDatabase

async with OllamaClient() as ollama:
    with HypothesisDatabase() as db:
        pipeline = ValidationPipeline(
            ollama_client=ollama,
            hypothesis_db=db,
        )
        
        result = await pipeline.validate(hypothesis)
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
