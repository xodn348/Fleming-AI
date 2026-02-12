#!/bin/bash

echo "================================================================"
echo "Fleming-AI 24시간 모니터링 시작"
echo "시작 시간: $(date)"
echo "PID: $(ps aux | grep continuous_collection | grep -v grep | awk '{print $2}')"
echo "================================================================"
echo ""
echo "로그 위치: /Users/jnnj92/Fleming-AI/logs/continuous_collection.log"
echo ""
echo "실시간 모니터링 (Ctrl+C로 중지):"
echo "================================================================"

tail -f /Users/jnnj92/Fleming-AI/logs/continuous_collection.log | while read line; do
    echo "[$(date '+%H:%M:%S')] $line"
done
