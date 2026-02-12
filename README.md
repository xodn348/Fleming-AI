# Fleming-AI

자동으로 논문을 읽고 가설을 생성하여 연구 논문을 작성하는 AI 시스템

## Quick Start (빠른 시작)

### 1. 설치
```bash
git clone https://github.com/xodn348/Fleming-AI.git
cd Fleming-AI
uv sync  # 또는 pip install -e .
```

### 2. API 키 설정
`.env` 파일 생성:
```bash
# Claude 세션 키 (추천 - 무료!)
# claude.ai 로그인 → F12 → Application → Cookies → sessionKey 복사
CLAUDE_SESSION_KEY=sk-ant-sid01-...

# 또는 무료 대안
GOOGLE_API_KEY=...      # Gemini (무료 1500 req/day)
GROQ_API_KEY=...        # Groq (무료 30 req/min)
OPENROUTER_API_KEY=...  # OpenRouter (무료 200 req/day)

# 로컬 임베딩 (필수)
# Ollama 설치: curl -fsSL https://ollama.com/install.sh | sh
# ollama pull nomic-embed-text
```

### 3. 전체 연구 파이프라인 실행
```bash
python scripts/run_full_research.py
```

**결과물**: `runs/[timestamp]/paper.pdf` - NeurIPS 형식 논문 (4페이지)

**파이프라인**:
1. 가설 생성 (vision-only, structured)
2. 실행가능성 검증
3. Alex 리뷰 (가설 품질 평가)
4. 실험 실행 (DeiT-Small vs ResNet-34 on Flowers102/CIFAR-10)
5. 결과 분석 및 그래프 생성
6. 논문 작성 및 PDF 컴파일

**예상 시간**: 약 12분

---

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

# Optional: TAMU Library Access (for subscription journals)
# Enables downloading papers from subscription-based journals (Nature, Science, etc.)
# via TAMU's EZProxy service
TAMU_NETID=your_netid
TAMU_PASSWORD=your_password
```

### TAMU Library Proxy Setup

To access subscription-based papers (Nature, Science, etc.), configure TAMU credentials:

1. **Get your TAMU NetID and password** from your TAMU account
2. **Add to `.env` file**:
   ```env
   TAMU_NETID=your_netid
   TAMU_PASSWORD=your_password
   ```
3. **How it works**:
   - When downloading papers via DOI, the system checks for TAMU credentials
   - If available, it authenticates with TAMU's EZProxy service
   - DOI URLs are transformed to proxy URLs for authenticated access
   - If proxy fails, falls back to direct download (works for open access papers)

**Note**: TAMU credentials are optional. Without them, the system will still download:
- All arXiv papers (open access)
- Papers available through direct DOI access
- Papers from other open access sources

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
