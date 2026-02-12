#!/bin/bash
# Start both processes with environment variables loaded

cd "$(dirname "$0")/.."

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Start paper collector
nohup python scripts/collect_papers_1000.py > /dev/null 2>&1 &
COLLECTOR_PID=$!
echo "Paper collector started: PID $COLLECTOR_PID"

# Start hypothesis generator  
nohup python scripts/generate_hypotheses_continuous.py > /dev/null 2>&1 &
GENERATOR_PID=$!
echo "Hypothesis generator started: PID $GENERATOR_PID"

echo "Both processes running. Check logs:"
echo "  tail -f logs/paper_collection.log"
echo "  tail -f logs/hypothesis_generation.log"
