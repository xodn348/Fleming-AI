# Alex Model Selection: 모델이 제일 중요하다

**핵심 질문**: "모델이 제일 중요하지 않아?"  
**답**: **맞습니다.** Base model의 ceiling이 최종 품질을 결정합니다.

---

## 📊 Model Quality Hierarchy (현실)

```
Tier 1 (Best): GPT-4, Claude Opus, Gemini Ultra
  → API 필요 ($20-40/month)
  
Tier 2 (Good): Claude 3.5 Sonnet, GPT-4o-mini, Llama 3.3 70B
  → 무료 tier 있음 (rate-limited)
  
Tier 3 (Okay): Llama 3.2 8B fine-tuned, Qwen 14B
  → 로컬 무제한
  
Tier 4 (Weak): Llama 3.2 8B base, GPT-3.5
  → 논문 리뷰는 부족
```

---

## 🎯 당신의 현재 상황

### 가지고 있는 것:
- ✅ Claude Max (web only, API 없음)
- ✅ OpenAI Plus (web only, API 없음)
- ✅ Groq 무료 tier (Llama 70B, **rate-limited** ← 문제)
- ✅ M1 Pro (로컬 8B 가능)

### 문제:
- Groq 70B: **37분 rate limit** (방금 겪음)
- Web subscriptions: API 없어서 자동화 불가
- Local 8B: 괜찮지만 70B보다 약함

---

## 💡 실용적 해결책 (우선순위)

### Option A: OpenRouter (무료 tier) ⭐ **추천**

**무엇**:
- Claude 3.5 Sonnet (free tier)
- GPT-4o-mini (free tier)
- Llama 70B (free tier)

**장점**:
- ✅ 무료
- ✅ API 있음 (자동화 가능)
- ✅ Rate limit이 Groq보다 관대
- ✅ 여러 모델 선택 가능

**Rate Limits** (무료):
- Claude 3.5 Sonnet: 200 req/day
- GPT-4o-mini: 200 req/day
- Llama 70B: unlimited (slow)

**구현**:
```python
# src/llm/openrouter_client.py 이미 있음!
from src.llm.openrouter_client import OpenRouterClient

# Just change one line in orchestrator:
async with OpenRouterClient() as client:  # ← Groq → OpenRouter
    alex = Alex(client)
    result = await alex.review_paper(paper)
```

**예상 품질**: Groq 70B와 비슷하거나 더 좋음 (Claude 3.5 Sonnet)

---

### Option B: API 구독 ($20/month)

**OpenAI API** ($20/month):
- GPT-4o: 논문 리뷰 충분
- Rate limit: 넉넉함 (5000 req/min)
- **Cost per review**: ~$0.02-0.05 (싸다)

**Anthropic API** ($20/month Claude Pro가 아닌 API 별도):
- Claude Opus: 최고 품질
- Rate limit: 넉넉함
- **Cost per review**: ~$0.10-0.20

**Trade-off**:
- 돈 들지만 품질 확실
- Alex 리뷰 1회 = $0.02-0.20 (커피보다 싸다)
- 월 100개 논문 리뷰 = $2-20

---

### Option C: Local Fine-tuned 8B (원래 계획)

**현실 체크**:
```
Fine-tuned Llama 8B vs Base Llama 70B:
- 8B fine-tuned: 도메인 강함, 전반적으론 약함
- 70B base: 도메인 약함, 전반적으론 강함

결과: 70B가 대부분 이김 (fine-tuning으로 못 메꿈)
```

**언제 8B가 나은가**:
- 매우 specific task (예: "CVPR 스타일로만 리뷰")
- 70B API 없을 때
- Cost/latency 극한 최적화

**논문 리뷰**는:
- 광범위한 지식 필요 (statistics, ML, writing)
- 추론 능력 필요 (논리적 약점 찾기)
- → 큰 모델이 유리

---

### Option D: Ensemble (최고 품질)

**What**:
```python
# 3개 모델에게 리뷰 요청
reviews = [
    await alex_gpt4o.review(paper),
    await alex_claude.review(paper),
    await alex_llama70b.review(paper)
]

# Aggregate
final_review = merge_reviews(reviews)  # Vote, average scores
```

**Pros**: 
- 최고 품질 (3 models > 1 model)
- 한 모델이 놓친 거 다른 모델이 잡음

**Cons**:
- 3배 느림
- 3배 비쌈 (API 쓴다면)

**When to use**: 최종 submission 전 한 번만

---

## 🎯 제 추천

### **Short-term** (지금 바로):

**1단계: OpenRouter 무료 tier 시도**
- 설치: 5분 (이미 코드 있음)
- Cost: $0
- Quality: Groq 70B와 동등 or 더 좋음
- Rate limit: 200/day (충분)

```bash
# .env에 추가
OPENROUTER_API_KEY=your_key_here

# Test
python -c "
import asyncio
from src.llm.openrouter_client import OpenRouterClient
from src.reviewers.alex import Alex

async def test():
    async with OpenRouterClient() as client:
        alex = Alex(client)
        result = await alex.review_hypothesis('Test hypothesis')
        print(f'Verdict: {result.verdict}')

asyncio.run(test())
"
```

**2단계: 결과 평가**
- OpenRouter Claude 3.5 품질 만족? → 계속 사용
- 부족함? → API 구독 고려

### **Long-term** (1-2주 후):

만약 OpenRouter로 만족 못하면:
- **OpenAI API 구독** ($20/month)
- GPT-4o로 Alex 리뷰
- Monthly cost: ~$20-30 (100-200 papers)

---

## 📊 비교표

| Option | Quality | Speed | Cost/month | Rate Limit | Setup |
|--------|---------|-------|------------|------------|-------|
| **OpenRouter (free)** | 8.5/10 | Medium | $0 | 200/day | 5 min |
| **OpenAI API** | 9/10 | Fast | $20-30 | 5000/min | 10 min |
| **Anthropic API** | 9.5/10 | Fast | $30-50 | 1000/min | 10 min |
| Fine-tuned 8B local | 6.5/10 | Slow | $0 | Unlimited | 1 day |
| Groq 70B (current) | 8/10 | Fast | $0 | **30 req/min** ⚠️ | Done |

---

## 🚀 실행 계획

### **Plan A: 무료로 해결** (추천)

```bash
# 1. OpenRouter 가입 (5분)
#    https://openrouter.ai/

# 2. API key 받기
#    Settings → Keys → Create Key

# 3. .env에 추가
echo "OPENROUTER_API_KEY=sk-or-..." >> .env

# 4. Test (Alex with OpenRouter)
python scripts/test_alex_openrouter.py

# 5. 실제 논문 리뷰
python scripts/review_vit_paper_openrouter.py
```

**Expected**: 
- 5분 setup
- Claude 3.5 Sonnet 품질
- 200 reviews/day (충분)
- $0

### **Plan B: 품질 최우선** ($20/month 투자 가치 있음)

```bash
# OpenAI API 구독
# → GPT-4o
# → $0.02-0.05 per review
# → 월 500 reviews = $10-25
```

---

## 💬 솔직한 조언

**"모델이 제일 중요하지 않아?"** → **맞습니다.**

Fine-tuned 8B는:
- ✅ 학습 경험으로 좋음
- ✅ 완전 무료 + unlimited
- ❌ **하지만 70B 품질은 못 따라감**

**현실적 선택**:
1. **OpenRouter 무료 (Claude 3.5)** ← 지금 바로 이거
2. 만족 못하면 → **OpenAI API** ($20/month, 확실함)
3. Fine-tuning은 → 나중에 (학습용/실험용)

**Bottom line**: 
- Base model ceiling > fine-tuning gains
- 70B base > 8B fine-tuned (대부분의 경우)
- **OpenRouter 무료부터 시작**이 정답

---

## ❓ 지금 뭐 할까요?

**A. OpenRouter 무료 tier 시도** (5분 setup, 즉시 리뷰 가능)  
**B. OpenAI API 구독 결정** ($20/month, 품질 확실)  
**C. 둘 다 테스트** (OpenRouter → 안되면 OpenAI)

어떤 거 해볼까요?
