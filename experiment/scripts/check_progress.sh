#!/bin/bash
# Quick progress check script

echo "=========================================="
echo "EXPERIMENT PROGRESS CHECK"
echo "=========================================="
echo ""

# Check if process is running
if ps aux | grep -q "[0-9] python experiment/scripts/02_run_experiments.py"; then
    echo "✅ Experiments are RUNNING"
    PID=$(ps aux | grep "[0-9] python experiment/scripts/02_run_experiments.py" | awk '{print $2}')
    echo "   PID: $PID"
else
    echo "❌ Experiments are NOT running"
fi

echo ""

# Count completed experiments
if [ -f experiment/results/all_results.jsonl ]; then
    COUNT=$(wc -l < experiment/results/all_results.jsonl)
    echo "📊 Completed: $COUNT/120 ($(echo "scale=1; $COUNT*100/120" | bc)%)"
    
    # Show latest result
    echo ""
    echo "Latest result:"
    tail -1 experiment/results/all_results.jsonl | python3 -c "import sys, json; r=json.load(sys.stdin); print(f\"  {r['arch']}, pretrained={r['pretrained']}, {r['dataset']}, {r['eval_method']}, seed={r['seed']}\"); print(f\"  Accuracy: {r['accuracy']:.2%}\")"
else
    echo "📊 Completed: 0/120"
fi

echo ""

# Show recent log
echo "Recent log (last 5 lines):"
tail -5 experiment/results/experiment.log | sed 's/^/  /'

echo ""
echo "=========================================="
