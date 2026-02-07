# Fleming-AI End-to-End Test Suite

## Overview

`test_e2e.py` is a comprehensive end-to-end test script for Fleming-AI that validates the complete pipeline from paper collection through hypothesis generation.

## Features

✓ **Quality Scoring Tests**
- Validates quality score calculation (0-100 scale)
- Tests venue tier detection (Tier 1, 2, 3)
- Verifies citation velocity scoring

✓ **Quality Filter Tests**
- Tests hypothesis quality scoring (0-1 scale)
- Validates filter initialization

✓ **Paper Collector Tests**
- Tests collector initialization
- Validates paper quality scoring
- Gracefully handles missing API keys

✓ **Hypothesis Database Tests**
- Database initialization and schema
- Hypothesis insertion and retrieval
- Status updates and scoring
- Full-text search functionality
- Top hypothesis ranking

✓ **VectorDB Tests**
- ChromaDB initialization
- Collection creation and management

✓ **Hypothesis Generation Tests**
- Component imports and initialization
- ConceptPair creation
- ABC pattern detection (Swanson's model)

## Running the Tests

### Basic Usage

```bash
cd ~/Fleming-AI
python scripts/test_e2e.py
```

### Expected Output

```
======================================================================
Fleming-AI End-to-End Test Suite
======================================================================
Started: 2026-02-07 01:05:09

📊 Testing Quality Scoring...
  ✓ Quality score calculation
    → Score: 88.5/100 for NeurIPS paper with 150 citations
  ✓ Venue tier scoring
    → Tested 3 different venue types
...
======================================================================
✓ ALL TESTS PASSED (22/22)
======================================================================
```

## Test Coverage

### 1. Quality Scoring (2 tests)
- Composite quality score calculation
- Venue tier-based scoring

### 2. Quality Filter (4 tests)
- Filter initialization
- Hypothesis quality scoring (3 test cases)

### 3. Paper Collector (2 tests)
- Collector initialization (skipped if API keys missing)
- Paper quality scoring

### 4. Hypothesis Database (11 tests)
- Database initialization
- Empty database count
- Hypothesis insertion
- Hypothesis retrieval
- Count after insertion
- Get all hypotheses
- Get top hypotheses
- Update hypothesis status
- Verify status update
- Search hypotheses

### 5. VectorDB (2 tests)
- VectorDB initialization
- Collection creation

### 6. Hypothesis Generation (2 tests)
- HypothesisGenerator import
- ConceptPair creation

## Exit Codes

- **0**: All tests passed
- **1**: One or more tests failed

## Test Markers

- ✓ **Passed**: Test completed successfully
- ✗ **Failed**: Test failed with error
- ⊘ **Skipped**: Test skipped (e.g., missing API keys)

## Dependencies

The test script requires:
- Python 3.8+
- Fleming-AI source code
- SQLite3
- ChromaDB
- All dependencies from `pyproject.toml`

## Notes

### API Keys
The paper collector test is skipped if OpenAlex API credentials are not configured. This is expected behavior in test mode.

### Database Cleanup
The test script automatically cleans up test databases after execution.

### No Production Data Modified
All tests use isolated test databases and do not modify production data.

## Troubleshooting

### Test Timeout
If tests hang, check:
1. Ollama server status (if testing LLM features)
2. Network connectivity for API calls
3. Disk space for database operations

### Import Errors
Ensure you're running from the Fleming-AI root directory:
```bash
cd ~/Fleming-AI
python scripts/test_e2e.py
```

### Database Errors
If you see database errors, ensure the `data/db/` directory exists:
```bash
mkdir -p ~/Fleming-AI/data/db
```

## Integration with CI/CD

To use in CI/CD pipelines:

```bash
#!/bin/bash
cd ~/Fleming-AI
python scripts/test_e2e.py
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "✗ Tests failed"
fi
exit $exit_code
```

## Future Enhancements

- [ ] Add async/await tests for Ollama integration
- [ ] Add paper collection tests with mock API responses
- [ ] Add hypothesis generation tests with sample papers
- [ ] Add performance benchmarking
- [ ] Add test coverage reporting
