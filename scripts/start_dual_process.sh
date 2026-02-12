#!/bin/bash

cd "$(dirname "$0")/.."

echo "================================================================"
echo "🚀 Starting Fleming-AI Dual Process Mode"
echo "================================================================"
echo ""

# Kill any existing processes
pkill -f "collect_papers_1000.py" 2>/dev/null
pkill -f "generate_hypotheses_continuous.py" 2>/dev/null
sleep 2

# Start paper collector
echo "📚 Starting paper collector (until 1000 papers)..."
OPENROUTER_API_KEY="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" \
nohup python scripts/collect_papers_1000.py > logs/paper_collection.log 2>&1 &
PAPER_PID=$!
echo "   PID: $PAPER_PID"

sleep 2

# Start hypothesis generator  
echo "💡 Starting hypothesis generator (runs forever)..."
OPENROUTER_API_KEY="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" \
nohup python scripts/generate_hypotheses_continuous.py > logs/hypothesis_generation.log 2>&1 &
HYPO_PID=$!
echo "   PID: $HYPO_PID"

echo ""
echo "================================================================"
echo "✅ Both processes started"
echo "================================================================"
echo ""
echo "Monitor:"
echo "  Paper collection:    tail -f logs/paper_collection.log"
echo "  Hypothesis generation: tail -f logs/hypothesis_generation.log"
echo ""
echo "Check status:"
echo "  ps aux | grep -E 'collect_papers|generate_hypotheses' | grep -v grep"
echo ""
