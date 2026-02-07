# Fleming-AI: 자가 학습하는 논문 발견 시스템

## 🎯 프로젝트 목표

**위대한 논문을 끊임없이 발견하고, 스스로 개선하는 AI 시스템**

- ✅ 고품질 논문만 자동 수집 (질적 접근)
- ✅ Literature-Based Discovery로 새로운 가설 생성
- ✅ 자가 학습: 성능 데이터 기반 자동 개선

## 📊 현재 상태

| 항목 | 상태 |
|------|------|
| **논문 DB** | 106개 (86개 인용 데이터 포함) |
| **VectorDB** | 231 chunks (10개 논문) |
| **자동 수집** | ✅ 주간 스케줄러 구현 |
| **자가 개선** | ✅ AdaptiveCollector 구현 |
| **시스템 상태** | ✅ 완전 작동 |

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# API 키 설정 (둘 중 하나)
export OPENALEX_EMAIL="your-email@example.com"
# OR
export OPENALEX_API_KEY="your-key"
```

### 2. 첫 논문 수집
```bash
cd ~/Fleming-AI
python main.py collect --limit 10 --test-mode
```

### 3. 자동 수집 시작
```bash
# 주 1회 자동 실행
nohup python scripts/schedule_collection.py --frequency weekly &
```

## 🧠 핵심 기능

### 1. 지능형 논문 수집
- **연령 정규화**: 최신 논문 30 인용/년, 오래된 논문 200+ 인용
- **학회 등급**: Tier 1 (NeurIPS, ICML) 우선
- **종합 점수**: 인용 속도 + 학회 + 영향력 + 수상 + 개념

### 2. 자가 개선 시스템 (NEW!)
```python
from src.intelligence.adaptive_collector import AdaptiveCollector

collector = AdaptiveCollector()
result = await collector.collect_with_learning()

# 자동으로:
# - 성공률 분석
# - 임계값 조정
# - 학회 가중치 최적화
# - A/B 테스트
```

### 3. 지속적인 발견
- 주간 자동 수집
- 품질 필터: 60점 이상만 저장
- 중복 제거: DOI/arXiv ID/제목 해시

## 📈 성능 지표

| 지표 | 목표 |
|------|------|
| 필터 통과율 | 10-30% |
| 가설 검증률 | 40-60% |
| 수집 주기 | 주 1회 |
| 논문 성장 | 월 20-50개 |

## 🛠️ 주요 파일

```
Fleming-AI/
├── src/
│   ├── collectors/paper_collector.py      # 논문 수집 엔진
│   ├── intelligence/adaptive_collector.py # 자가 개선 시스템 (NEW!)
│   ├── utils/scoring.py                   # 품질 점수 계산
│   └── scheduler/runner.py                # 자동화 스케줄러
├── scripts/
│   ├── schedule_collection.py             # 주간 자동 수집
│   ├── enrich_papers.py                   # 인용 데이터 추가
│   └── test_e2e.py                        # 통합 테스트
├── data/
│   ├── db/papers.db                       # 논문 메타데이터
│   ├── db/metrics.db                      # 성능 지표 (NEW!)
│   └── db/chromadb/                       # VectorDB 임베딩
└── DEPLOYMENT.md                          # 배포 가이드
```

## 🧪 테스트

```bash
# 전체 시스템 테스트
python scripts/test_e2e.py

# 개별 컴포넌트 테스트
python -c "from src.utils.scoring import calculate_quality_score; print('OK')"
```

## 📊 모니터링

```bash
# 수집 통계
sqlite3 data/db/metrics.db "SELECT * FROM collection_cycles ORDER BY cycle_id DESC LIMIT 5;"

# 성능 추이
sqlite3 data/db/metrics.db "SELECT * FROM threshold_history ORDER BY changed_at DESC;"

# 학회 성과
sqlite3 data/db/metrics.db "SELECT venue_name, success_rate FROM venue_performance;"
```

## 🎓 연구 배경

### Don Swanson's ABC Model
```
논문 A: X → Y 연결
논문 B: Y → Z 연결
가설: X → Z 새로운 발견 가능!
```

### 필요한 논문 수
- **최소**: 106개로 가능 (Swanson 원래 연구: 489개)
- **권장**: 500개 (안정적인 패턴)
- **핵심**: 개념 다양성 > 논문 수

## 💡 사용 예시

### 수동 수집
```bash
# 소규모 테스트
python main.py collect --limit 10 --test-mode

# 프로덕션 수집
python main.py collect --limit 100
```

### 자동 수집 (백그라운드)
```bash
# 데몬으로 실행
nohup python scripts/schedule_collection.py --frequency weekly > logs/scheduler.log 2>&1 &

# 상태 확인
ps aux | grep schedule_collection
tail -f logs/scheduler.log
```

### Cron Job
```bash
# crontab -e
0 2 * * 1 cd ~/Fleming-AI && python scripts/schedule_collection.py --once
```

## 🔧 문제 해결

### API 키 오류
```bash
export OPENALEX_EMAIL="your-email@example.com"
echo 'export OPENALEX_EMAIL="your-email@example.com"' >> ~/.zshrc
```

### VectorDB 비어있음
```bash
python scripts/embed_papers.py
```

### 성능 조정
```python
collector = PaperCollector(config={
    'min_citations': 200,      # 기본: 100
    'quality_threshold': 70     # 기본: 60
})
```

## 🎯 다음 단계

1. **환경 설정**: API 키 등록
2. **테스트 실행**: `python main.py collect --limit 10 --test-mode`
3. **자동화 시작**: 주간 스케줄러 활성화
4. **모니터링**: metrics.db 확인
5. **개선**: AdaptiveCollector 피드백 루프 활용

## 📚 상세 문서

- [DEPLOYMENT.md](DEPLOYMENT.md) - 배포 가이드
- [scripts/SCHEDULER_README.md](scripts/SCHEDULER_README.md) - 스케줄러 설정

---

**시스템 준비 완료!** 자고 일어나면 자동으로 논문을 수집하고 스스로 개선됩니다. 🌙
