#!/bin/bash
# Fleming-AI 자동 수집 서비스 시작

echo "🚀 Fleming-AI 자동 수집 서비스 시작"
echo ""

# Load API key from zshrc if not already set
if [ -z "$OPENALEX_API_KEY" ] && [ -z "$OPENALEX_EMAIL" ]; then
    if [ -f ~/.zshrc ]; then
        source ~/.zshrc
    fi
fi

# API 키 확인
if [ -z "$OPENALEX_EMAIL" ] && [ -z "$OPENALEX_API_KEY" ]; then
    echo "⚠️  OpenAlex API 키/이메일이 설정되지 않았습니다."
    echo "다음 명령어로 설정하세요:"
    echo "  export OPENALEX_EMAIL='your-email@example.com'"
    echo ""
    echo "또는 ~/.zshrc에 추가:"
    echo "  echo 'export OPENALEX_EMAIL=\"your-email@example.com\"' >> ~/.zshrc"
    echo ""
    exit 1
fi

echo "✓ OpenAlex API 키 확인됨"

# 디렉토리 확인
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "✓ logs 디렉토리 생성"
fi

# 서비스 시작 (일간 수집 + 가설 생성)
echo "일간 자동 수집 & 가설 생성 서비스 시작 중..."
echo "목표: 1000개 고품질 논문 수집 후 자동 중지"
echo ""

# Export all API keys for subprocess
export OPENALEX_API_KEY
export OPENALEX_EMAIL
export CLAUDE_SESSION_KEY
export KIMI_API_KEY

nohup python scripts/schedule_collection.py --frequency daily > logs/scheduler.log 2>&1 &
PID=$!
echo "✓ 서비스 시작됨 (PID: $PID)"
echo ""
echo "로그 확인: tail -f logs/scheduler.log"
echo "논문 개수 확인: sqlite3 ~/Fleming-AI/data/db/papers.db 'SELECT COUNT(*) FROM great_papers;'"
echo "중지: kill $PID"
