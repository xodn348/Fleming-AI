#!/bin/bash

clear
echo "================================================================"
echo "🎯 Fleming-AI - 1000개 가설 생성 미션"
echo "================================================================"
echo ""
echo "시작 시간: $(date)"
echo "PID: $(ps aux | grep continuous_collection | grep -v grep | awk '{print $2}')"
echo ""

# 현재 상태 표시
cd /Users/jnnj92/Fleming-AI
CURRENT=$(python -c "from src.storage.hypothesis_db import HypothesisDatabase; db=HypothesisDatabase(); print(db.count_hypotheses())" 2>/dev/null || echo "확인중...")
echo "현재 가설 수: ${CURRENT}/1000"
echo ""
echo "================================================================"
echo "실시간 로그 (Ctrl+C로 중지):"
echo "================================================================"
echo ""

tail -f logs/continuous_collection.log | grep --line-buffered -E "Progress:|Generated|Stored|Cycle complete|TARGET REACHED" | while read line; do
    echo "[$(date '+%H:%M:%S')] $line"
    
    # 1000개 달성 체크
    if echo "$line" | grep -q "TARGET REACHED"; then
        echo ""
        echo "================================================================"
        echo "🎉🎉🎉 목표 달성! 1000개 가설 생성 완료! 🎉🎉🎉"
        echo "================================================================"
        break
    fi
done
