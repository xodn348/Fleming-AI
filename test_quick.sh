#!/bin/bash
# 빠른 시스템 테스트

echo "🧪 Fleming-AI 빠른 테스트"
echo ""

# Python 모듈 로드 테스트
python -c "
from src.utils.scoring import calculate_quality_score
from src.storage.database import PaperDatabase
from src.storage.vectordb import VectorDB
print('✓ 모든 모듈 임포트 성공')
"

if [ $? -eq 0 ]; then
    echo "✓ 시스템 준비 완료"
    echo ""
    echo "다음 명령어로 실행:"
    echo "  python main.py collect --limit 10 --test-mode"
else
    echo "✗ 모듈 로드 실패"
    exit 1
fi
