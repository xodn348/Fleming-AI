# 🚀 Fleming-AI 듀얼 시스템 실행 중

## 시작 시간
**2026-02-07 15:22**

---

## 📊 현재 상태

### 프로세스 1: 📚 논문 수집기
- **목표**: 1000개 논문 수집
- **시작 논문 수**: 154개
- **필요 수집**: 846개
- **PID**: 81419
- **로그**: `logs/paper_collection.log`
- **종료 조건**: 1000개 도달 시 자동 정지

**작동 방식:**
```
OpenAlex API → 50개 후보 발견
   ↓
Semantic Scholar → 인용 데이터 보강
   ↓
Quality Filter → 고품질 논문만 선택
   ↓
Database 저장 → 반복
```

### 프로세스 2: 💡 가설 생성기
- **목표**: 무한 실행 (계속 생성/검증)
- **PID**: 81436
- **로그**: `logs/hypothesis_generation.log`
- **종료 조건**: 없음 (수동 종료만 가능)

**작동 방식:**
```
VectorDB → 5개 논문 샘플링
   ↓
개념 추출 → Ollama qwen2.5:14b (병렬)
   ↓
ABC 패턴 탐색 → 최대 50개
   ↓
가설 생성 → Trinity Large Preview (병렬)
   ↓
검증 → Ollama qwen2.5:14b
   ↓
Database 저장 → 반복 (1초 쿨다운)
```

---

## 🎯 AI 모델 구성

| 작업 | 모델 | 위치 |
|------|------|------|
| 가설 생성 | Trinity Large Preview (400B) | OpenRouter 무료 |
| 개념 추출 | Ollama qwen2.5:14b (8.7GB) | 로컬 |
| 가설 검증 | Ollama qwen2.5:14b | 로컬 |
| 임베딩 | Ollama nomic-embed-text | 로컬 |

---

## 📈 예상 소요 시간

### 논문 수집 (846개 필요)
- **사이클당 저장**: ~5-10개
- **필요 사이클**: 85-170회
- **사이클당 시간**: ~60초
- **예상 총 시간**: **1.5~3시간**

### 가설 생성 (무한)
- **사이클당 시간**: ~80초
- **사이클당 생성**: 0-3개 (데이터에 따라 변동)
- **계속 실행**: 영구적

---

## 🔍 모니터링 방법

### 실시간 로그 확인
```bash
# 논문 수집 모니터
tail -f /Users/jnnj92/Fleming-AI/logs/paper_collection.log | grep "Progress:"

# 가설 생성 모니터
tail -f /Users/jnnj92/Fleming-AI/logs/hypothesis_generation.log | grep -E "Generated|Total:"
```

### 현재 상태 확인
```bash
cd /Users/jnnj92/Fleming-AI

# 논문 수
python -c "from src.storage.database import PaperDatabase; db=PaperDatabase('data/db/papers.db'); print(f'{len(db.get_all_papers())}/1000 papers')"

# 가설 수
python -c "from src.storage.hypothesis_db import HypothesisDatabase; db=HypothesisDatabase(); print(f'{db.count_hypotheses()} hypotheses')"

# 프로세스 확인
ps aux | grep -E "collect_papers|generate_hypotheses" | grep -v grep
```

### 통합 대시보드
```bash
cd /Users/jnnj92/Fleming-AI
./scripts/monitor_dual.sh  # 실시간 통합 모니터링
```

---

## 🎉 자동 종료

### 논문 수집기
```
✅ Progress: 1000/1000 papers
🎉 TARGET REACHED: 1000 papers collected!
Paper collection complete. Exiting.
```
→ **자동 정지**

### 가설 생성기
```
계속 실행 (종료 없음)
```
→ **수동 정지 필요**: `pkill -f generate_hypotheses_continuous.py`

---

## ⚠️ 중요 사항

### API Rate Limits
- **Semantic Scholar**: 429 에러 발생 시 자동 재시도 (3초 대기)
- **OpenRouter**: 무료 티어 제한 (시간당 제한 있을 수 있음)
- **OpenAlex**: 무제한 (polite pool)

### 데이터베이스 안전
- ✅ SQLite WAL 모드 활성화
- ✅ 동시 쓰기 지원
- ✅ 두 프로세스 간 충돌 없음

### 디스크 공간
- **논문 1000개**: ~100MB
- **가설 계속 증가**: 시간당 ~5-10MB
- **로그 파일**: 시간당 ~1-2MB

---

## 📂 주요 파일

| 파일 | 용도 |
|------|------|
| `scripts/collect_papers_1000.py` | 논문 수집 스크립트 |
| `scripts/generate_hypotheses_continuous.py` | 가설 생성 스크립트 |
| `scripts/start_dual_process.sh` | 통합 실행 스크립트 |
| `logs/paper_collection.log` | 논문 수집 로그 |
| `logs/hypothesis_generation.log` | 가설 생성 로그 |
| `data/db/papers.db` | 논문 데이터베이스 |
| `data/db/hypotheses.db` | 가설 데이터베이스 |

---

## 🛠 문제 해결

### 프로세스가 죽은 경우
```bash
cd /Users/jnnj92/Fleming-AI
./scripts/start_dual_process.sh
```

### 논문 수집만 재시작
```bash
pkill -f collect_papers_1000.py
OPENROUTER_API_KEY="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" \
nohup python scripts/collect_papers_1000.py > logs/paper_collection.log 2>&1 &
```

### 가설 생성만 재시작
```bash
pkill -f generate_hypotheses_continuous.py
OPENROUTER_API_KEY="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" \
nohup python scripts/generate_hypotheses_continuous.py > logs/hypothesis_generation.log 2>&1 &
```

---

**🎯 두 프로세스가 독립적으로 실행 중입니다!**
**논문 수집은 1000개 도달 시 자동 정지, 가설 생성은 계속 실행됩니다!**
