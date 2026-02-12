# Google AI Studio 설정 가이드

## Option 1: 개인 Google AI Studio (즉시 사용)

### 1. API Key 발급 (5분)

1. **Google AI Studio 접속**
   - URL: https://aistudio.google.com
   - **중요**: 개인 Gmail 계정 사용 (TAMU Gmail은 조직 정책으로 차단됨)
   - TAMU 계정 에러 발생 시 → 다른 계정으로 로그인

2. **API Key 생성**
   - 왼쪽 사이드바: "Get API key" 클릭
   - "Create API key" 버튼 클릭
   - 새 프로젝트 생성 또는 기존 프로젝트 선택
   - API key 복사 (예: `AIzaSy...`)

3. **Fleming-AI에 설정**
   ```bash
   cd /Users/jnnj92/Fleming-AI
   
   # .env 파일 생성/수정
   echo "GOOGLE_API_KEY=AIzaSy..." >> .env
   ```

### 2. 무료 Tier 제한

| 모델 | 무료 할당량 | 품질 | 속도 |
|------|-----------|------|------|
| **Gemini 2.5 Pro** | 10 RPM, 50 RPD | 9.5/10 | 느림 |
| **Gemini 2.5 Flash** | 15 RPM, 1500 RPD | 8.5/10 | 빠름 |

**전략**: Pro 먼저 시도 → rate limit 걸리면 Flash로 fallback

### 3. 사용 예시

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Gemini 2.5 Pro (최고 성능)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

response = model.generate_content("Review this ML paper...")
print(response.text)
```

---

## ~~Option 2: TAMU Google AI Studio 신청~~ (차단됨)

**업데이트**: TAMU organization이 AI Studio 접근을 차단했습니다.
- 에러: "Your account is managed by an organization that has this service turned off"
- **해결책**: 개인 Gmail 계정 사용 (Option 1)

---

## Option 2 (대안): TAMU AI Development Platform 신청 (예산 필요)

### 1. 신청 절차

1. **Help Desk 티켓 제출**
   - URL: https://service.tamu.edu/TDClient/36/Portal/Home/
   - 카테고리: "AI Services" > "Google AI Studio Request"
   - 또는 직접 이메일: helpdesk@tamu.edu

2. **요청 내용 (템플릿)**
   ```
   Subject: Google AI Studio Access Request for Research

   Name: [Your Name]
   NetID: [Your NetID]
   Department: Computer Science
   
   Purpose: Academic research - ML paper writing assistant (Fleming-AI)
   Use case: Automated literature review and paper quality assessment
   Expected usage: 50-100 API requests/day
   Duration: Ongoing research project
   
   I understand TAMU AI Studio is approved for University-Confidential 
   data or lower, and will comply with data classification policies.
   ```

3. **승인 대기**
   - 예상 시간: 1-3 영업일
   - 승인되면 API key 또는 접근 방법 안내

### 2. TAMU vs 개인 비교

| 항목 | 개인 AI Studio | TAMU AI Studio |
|------|---------------|---------------|
| **비용** | 무료 (제한적) | 무료/할인 (TAMU 예산) |
| **Quota** | 10 RPM (Pro) | 더 높을 가능성 |
| **승인** | 즉시 | 1-3일 |
| **데이터** | 개인 책임 | TAMU 보호 |
| **지원** | Google 문서 | TAMU Help Desk |

**추천**: 개인으로 즉시 시작 → TAMU 승인되면 전환

---

## 다음 단계

1. ✅ API key 발급 완료
2. ⏭️ `src/llm/gemini_client.py` 구현
3. ⏭️ `src/pipeline/orchestrator.py` 통합
4. ⏭️ ViT paper 실제 리뷰 테스트

---

## 참고 자료

- Google AI Studio: https://aistudio.google.com
- API 문서: https://ai.google.dev/gemini-api/docs
- TAMU AI Services: https://www.it.tamu.edu/ai-services/
- Pricing: https://ai.google.dev/pricing
