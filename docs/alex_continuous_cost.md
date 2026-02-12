# Alex Continuous Operation Cost: 계속 돌리면 얼마?

**핵심 질문**: "아니 계속 돌리면 얼마가 나오냐고"  
**답**: **월 $30-100 나올 수 있습니다** (솔직히)

---

## 🔄 Full Pipeline 1회 비용

### Fleming-Alex 전체 사이클

```
Stage 1: Hypothesis Review
  - Alex 리뷰: $0.0045
  - Fleming 수정
  - Alex 재리뷰: $0.0045
  - 수렴 (평균 2-3 turns)
  소계: $0.009-0.0135

Stage 2: Experiment Design Review
  - 동일 패턴
  소계: $0.009-0.0135

Stage 3: Results Review
  - 동일 패턴
  소계: $0.009-0.0135

Stage 4: Paper Review
  - 동일 패턴
  소계: $0.009-0.0135

────────────────────────────────
Full Pipeline 1회: $0.036-0.054 (약 50-70원)
```

---

## ⏱️ 처리 속도 (Bottleneck)

### 시간 분석

```
1. Hypothesis (Alex review): 10초
2. Experiment Design (Alex review): 10초
3. 실험 실행 (Fleming): ⏰ 15-20분 ← BOTTLENECK
4. Results (Alex review): 10초
5. Paper Generation (Fleming): 30초
6. Paper Review (Alex review): 15초

────────────────────────────────
Total per paper: ~20-25분
시간당 처리량: 2-3 papers
```

**Key Point**: 실험 실행 시간이 bottleneck이라 API 호출은 생각보다 느림

---

## 💰 24시간 Continuous 비용

### 시나리오 A: 보수적 (시간당 2 papers)

```
시간당: 2 papers × $0.045 = $0.09
일당: 24시간 × $0.09 = $2.16
월당: 30일 × $2.16 = $64.80

━━━━━━━━━━━━━━━━━━━━━━━━━━━
월 비용: ~$65 (약 87,000원)
```

### 시나리오 B: 공격적 (시간당 3 papers)

```
시간당: 3 papers × $0.045 = $0.135
일당: 24시간 × $0.135 = $3.24
월당: 30일 × $3.24 = $97.20

━━━━━━━━━━━━━━━━━━━━━━━━━━━
월 비용: ~$97 (약 130,000원)
```

### 시나리오 C: 현실 (8시간/day, 주중만)

```
일당: 8시간 × 2 papers × $0.045 = $0.72
월당: 22일(주중) × $0.72 = $15.84

━━━━━━━━━━━━━━━━━━━━━━━━━━━
월 비용: ~$16 (약 21,000원)
```

---

## ⚠️ 실제 비용 경고

### 최악의 경우 (24/7 full throttle)

```
GPT-5 mini:
월 비용: $60-100 (80,000-130,000원)

GPT-5.2:
월 비용: $420-700 (560,000-930,000원)

GPT-5.2 pro:
월 비용: $5,000-8,400 (6,700,000-11,200,000원) ⚠️⚠️⚠️
```

**Yes, 계속 돌리면 비쌉니다.**

---

## 🛡️ 비용 통제 방법

### 1. Budget Limit 설정 (필수!)

```python
# OpenAI Dashboard → Settings → Billing → Usage limits
Monthly budget: $20 (또는 원하는 금액)

→ 초과 시 자동 중단
```

**추천 설정**:
- 테스트: $5/month
- 일반 사용: $20/month
- 연구실: $50/month

### 2. Rate Limiting (코드 수준)

```python
# src/pipeline/orchestrator.py
class FlemingAlexOrchestrator:
    def __init__(self, groq_client, max_reviews_per_day=50):
        self.max_reviews_per_day = max_reviews_per_day
        self.daily_count = 0
        
    async def run_full_pipeline(self, hypothesis):
        if self.daily_count >= self.max_reviews_per_day:
            raise RateLimitError("Daily review limit reached")
        
        result = await super().run_full_pipeline(hypothesis)
        self.daily_count += 1
        return result
```

**효과**: 일 50개 × $0.045 = $2.25/day = **$67/month cap**

### 3. Selective Review (권장!)

```python
# 모든 논문을 리뷰하지 말고, 중요한 것만

# Option A: 수동 트리거
if user_requests_review:
    await orchestrator.run_full_pipeline(hypothesis)

# Option B: Quality threshold
quality_score = quick_check(hypothesis)
if quality_score > 0.7:  # 괜찮은 것만 리뷰
    await orchestrator.run_full_pipeline(hypothesis)

# Option C: Sampling (10%만)
if random.random() < 0.1:
    await orchestrator.run_full_pipeline(hypothesis)
```

**효과**: 비용 90% 절감

### 4. Hybrid Model (최적!)

```python
# Stage 1-2: 로컬 모델 (free, unlimited)
hypothesis_review = await local_alex.review_hypothesis(hyp)
design_review = await local_alex.review_design(design)

# Stage 3-4: GPT-5 mini (critical stages only)
results_review = await gpt5_alex.review_results(results)
paper_review = await gpt5_alex.review_paper(paper)
```

**효과**: 비용 50% 절감, 품질 유지

### 5. Caching (OpenAI native)

```python
# 동일 프롬프트 재사용 시 자동 캐싱
# Input cost: $0.25/1M → $0.025/1M (10x cheaper)

# Alex 프롬프트는 고정이므로:
# 2번째 리뷰부터 input cost 90% 절감
```

**효과**: 장기적으로 40-50% 절감

---

## 📊 비용 비교 (월 사용량별)

| 사용 패턴 | Papers/월 | GPT-5 mini | GPT-5.2 | 추천 |
|----------|-----------|-----------|---------|------|
| **가끔 사용** | 10-20 | $0.45-0.90 | $3.15-6.30 | ✅ |
| **주간 사용** | 50-100 | $2.25-4.50 | $15.75-31.50 | ✅ |
| **매일 사용** | 200-300 | $9-13.50 | $63-94.50 | ⚠️ |
| **24/7 자동** | 1500+ | $67+ | $470+ | ❌ |

---

## 🎯 실용적 권장사항

### Plan A: 선택적 사용 (추천)

```python
# 중요한 논문만 full review
# 일반 논문은 quick check

월 비용: $2-10 (2,700-13,000원)
```

### Plan B: Hybrid (로컬 + API)

```python
# Stage 1-2: Local (free)
# Stage 3-4: API (critical)

월 비용: $5-20 (6,700-27,000원)
```

### Plan C: Budget Cap

```python
# OpenAI dashboard에서 $20/month 설정
# 초과 시 자동 중단

월 비용: 최대 $20 (보장)
```

---

## 💡 현실적 사용 시나리오

### 시나리오 1: 개인 연구자

```
- 주 2-3 papers review
- 월 10-15 papers
- GPT-5 mini 사용

월 비용: $0.45-0.68 (600-900원) ✅
```

### 시나리오 2: 연구실

```
- 일 1-2 papers review (주중)
- 월 40-50 papers
- GPT-5 mini + selective

월 비용: $1.80-2.25 (2,400-3,000원) ✅
```

### 시나리오 3: 자동 시스템 (위험)

```
- 24/7 continuous
- 월 1500+ papers
- GPT-5 mini

월 비용: $67+ (89,000원+) ⚠️
```

---

## ⚠️ 비용 폭탄 방지 체크리스트

### 1. Budget Limit 설정 (필수!)
```
OpenAI Dashboard → Usage limits → $20/month
```

### 2. 알림 설정
```
$10 도달 시 이메일 알림
$15 도달 시 경고
$20 도달 시 차단
```

### 3. 코드 레벨 제한
```python
MAX_REVIEWS_PER_DAY = 50  # 하드 캡
```

### 4. 모니터링
```python
# 일일 사용량 로깅
logger.info(f"Today's API cost: ${daily_cost:.2f}")
```

---

## 📈 비용 추이 예측

### 첫 달 (테스트)
```
Week 1: 5 papers × $0.045 = $0.23
Week 2: 10 papers × $0.045 = $0.45
Week 3: 15 papers × $0.045 = $0.68
Week 4: 20 papers × $0.045 = $0.90

Total: $2.26 ✅
```

### 정착 후 (regular use)
```
월 평균: 50-100 papers
비용: $2.25-4.50 (3,000-6,000원) ✅
```

### 만약 자동화 (조심!)
```
월 평균: 1500+ papers
비용: $67+ (89,000원+) ⚠️
```

---

## 🎯 결론

### "계속 돌리면 얼마가 나오냐고"

**솔직한 답**:
- **24/7 자동**: 월 $60-100 (80,000-130,000원)
- **선택적 사용**: 월 $2-10 (2,700-13,000원)
- **Budget cap**: 월 최대 $20 (안전)

**추천**:
1. ✅ Budget limit $20 설정 (필수)
2. ✅ 선택적 리뷰 (중요한 것만)
3. ✅ Hybrid (로컬 + API)
4. ❌ 24/7 자동은 비용 주의

**Bottom Line**:
- 제어하면: 월 $2-20 (관리 가능) ✅
- 방치하면: 월 $60-100+ (비쌀 수 있음) ⚠️

---

## 🚀 시작 가이드

### 안전하게 시작하기

```bash
# 1. Budget limit 설정
OpenAI Dashboard → $20/month cap

# 2. 테스트 with $5 credit
export OPENAI_API_KEY=...
python scripts/test_alex_with_budget.py

# 3. 모니터링
python scripts/check_daily_cost.py

# 4. 선택적 사용
# 자동 활성화 말고 수동 트리거
```

**First month target**: $5 이하 (안전 테스트)

---

**다음**: Budget limit 설정하고 테스트해볼까요?
