# Dual Process Mode - Paper Collection & Hypothesis Generation

## Overview
Two independent continuous processes running in parallel:
1. **Paper Collector** - Runs until 1000 papers collected, then stops
2. **Hypothesis Generator** - Runs forever, continuously generating/validating hypotheses

## Files Created

### 1. `scripts/collect_papers_1000.py`
- **Purpose**: Collect papers until reaching 1000 total
- **Behavior**: Runs collection cycles, checks count after each cycle, exits when count >= 1000
- **Logging**: `~/Fleming-AI/logs/paper_collection.log`
- **Exit Condition**: Automatic exit after 1000 papers collected

### 2. `scripts/generate_hypotheses_continuous.py`
- **Purpose**: Continuously generate and validate hypotheses
- **Behavior**: Infinite loop that generates hypotheses from paper samples and validates pending ones
- **Logging**: `~/Fleming-AI/logs/hypothesis_generation.log`
- **Exit Condition**: Manual termination only (runs forever)

### 3. `scripts/start_dual_process.sh`
- **Purpose**: Launcher script for both processes
- **Behavior**: 
  - Kills any existing instances
  - Starts paper collector in background
  - Starts hypothesis generator in background
  - Displays PIDs and monitoring instructions

## Usage

### Start Both Processes
```bash
cd /Users/jnnj92/Fleming-AI
./scripts/start_dual_process.sh
```

### Monitor Paper Collection
```bash
tail -f ~/Fleming-AI/logs/paper_collection.log
```

### Monitor Hypothesis Generation
```bash
tail -f ~/Fleming-AI/logs/hypothesis_generation.log
```

### Check Process Status
```bash
ps aux | grep -E 'collect_papers|generate_hypotheses' | grep -v grep
```

### Stop Paper Collector
```bash
pkill -f "collect_papers_1000.py"
```

### Stop Hypothesis Generator
```bash
pkill -f "generate_hypotheses_continuous.py"
```

### Stop Both Processes
```bash
pkill -f "collect_papers_1000.py"
pkill -f "generate_hypotheses_continuous.py"
```

## Process Details

### Paper Collector Cycle
1. Check current paper count in database
2. Log progress (count/1000)
3. If count >= 1000: exit with success message
4. Otherwise: run collection cycle
5. Brief pause (1 second)
6. Repeat

### Hypothesis Generator Cycle
1. Increment cycle counter
2. Get all paper IDs from vector database
3. If < 2 papers: wait 60 seconds and retry
4. Sample 5 papers (or fewer if not enough available)
5. Generate hypotheses from sampled papers (300s timeout)
6. Store generated hypotheses in database
7. Validate up to 5 pending hypotheses
8. Log statistics (total, pending, validated)
9. Brief pause (1 second)
10. Repeat forever

## Logging

Both processes log to:
- **Console**: Real-time output to stdout
- **File**: Persistent logs in `~/Fleming-AI/logs/`

Log format: `TIMESTAMP - LEVEL - MESSAGE`

### Log Files
- `paper_collection.log` - Paper collector activity
- `hypothesis_generation.log` - Hypothesis generator activity

## Environment Variables

Both processes require:
- `OPENROUTER_API_KEY` - Set in `start_dual_process.sh`

## Exit Behavior

### Paper Collector
- **Normal Exit**: When 1000 papers collected
- **Error Exit**: On unrecoverable errors (logs error and continues)
- **Manual Exit**: `pkill -f "collect_papers_1000.py"`

### Hypothesis Generator
- **Normal Exit**: Never (runs forever)
- **Manual Exit**: `pkill -f "generate_hypotheses_continuous.py"`

## Troubleshooting

### Paper Collector Not Starting
1. Check logs: `tail -f ~/Fleming-AI/logs/paper_collection.log`
2. Verify database path: `data/db/papers.db`
3. Check FlemingRunner initialization

### Hypothesis Generator Not Starting
1. Check logs: `tail -f ~/Fleming-AI/logs/hypothesis_generation.log`
2. Verify Ollama is running
3. Verify OpenRouter API key is set
4. Check vector database connectivity

### Both Processes Stuck
1. Check system resources (CPU, memory, disk)
2. Verify network connectivity
3. Check API rate limits
4. Review error logs for specific failures

## Performance Notes

- Paper collector: Minimal overhead, runs until target reached
- Hypothesis generator: Continuous operation, may consume resources
- Both processes use 1-second pause between cycles to prevent CPU spinning
- Hypothesis generation has 300-second timeout per cycle
