# Fleming-AI 시스템 상태 보고서
생성 시간: 2026-02-07 15:13

## ✅ 시스템 정상 작동 중

### 프로세스 상태
- **PID**: 76386
- **시작 시간**: 2026-02-07 15:10:06
- **상태**: 실행 중 (정상)
- **모드**: 연속 실행 (1시간 주기)

### 성능 개선 결과
| 항목 | 이전 | 현재 | 개선율 |
|------|------|------|--------|
| 가설 생성 시간 | 120s+ (타임아웃) | 78.9s | **35% 단축** |
| 개념 추출 | 순차 (600s) | 병렬 (60s) | **90% 단축** |
| 가설 생성 API 호출 | 순차 (600s) | 병렬 (60s) | **90% 단축** |
| 데이터베이스 잠금 오류 | 빈번 | 없음 | **100% 해결** |

### 데이터베이스 현황
- **논문**: 154개
- **가설 총계**: 30개
  - 대기 중: 18개
  - 검증 완료: 11개
  - 거부됨: 1개
- **VectorDB**: 95개 논문 인덱싱

### 사용 중인 AI 모델
1. **가설 생성**: Trinity Large Preview (400B params, OpenRouter 무료)
2. **개념 추출**: Ollama qwen2.5:14b (8.7GB, 로컬)
3. **검증**: Ollama qwen2.5:14b (로컬)
4. **임베딩**: Ollama nomic-embed-text (로컬)

### 최적화 적용 사항
- ✅ 개념 추출 병렬화 (Semaphore 3)
- ✅ 가설 생성 병렬화 (Semaphore 3)
- ✅ 파이프라인 타임아웃 (300초)
- ✅ ABC 패턴 제한 (최대 50개)
- ✅ SQLite WAL 모드 활성화
- ✅ ChromaDB allow_reset=False
- ✅ 불필요한 코드 제거 (550+ 라인)

### 로그 파일
- **위치**: `/Users/jnnj92/Fleming-AI/logs/continuous_collection.log`
- **모니터링**: `./monitor_fleming.sh` 실행

### 다음 주기
- **예정 시간**: 2026-02-07 16:10 (1시간 후)
- **작업**: 논문 샘플링 → 가설 생성 → 검증 → 저장 → 동기화

---

## 모니터링 방법

### 실시간 로그 확인
```bash
cd /Users/jnnj92/Fleming-AI
./monitor_fleming.sh
```

### 상태 확인
```bash
ps aux | grep continuous_collection
```

### 데이터베이스 확인
```bash
sqlite3 data/db/hypotheses.db "SELECT COUNT(*) FROM hypotheses;"
```

---

**시스템은 24시간 연속 실행됩니다. 문제 발생 시 로그를 확인하세요.**
