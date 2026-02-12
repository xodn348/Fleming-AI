# Claude 세션 키 설정 가이드

## 📋 요약

- **Alex 리뷰어**: Claude Opus 사용 ✅
- **Fleming 가설 생성**: Claude Opus 사용 ✅
- **같은 세션 키 공유**: 가능 ✅
- **백업 모델**: 없음 (Claude 실패 시 에러)

---

## 🔑 1. 세션 키 설정

### 방법 1: 새 세션 키 발급 (권장)

```bash
# 1. https://claude.ai 접속
# 2. 로그인
# 3. F12 (개발자 도구) → Application → Cookies
# 4. "sessionKey" 값 복사 (sk-ant-sid01-로 시작)
```

### 방법 2: 기존 세션 키 재사용

OpenCode와 같은 세션 키 사용 가능 (동시 실행 시 rate limit 공유됨)

---

## ⚙️ 2. .env 파일 설정

```bash
cd ~/Fleming-AI

# .env 파일 생성/수정
cat > .env << 'EOF'
# Claude 세션 키 (필수!)
CLAUDE_SESSION_KEY=sk-ant-sid01-YOUR_SESSION_KEY_HERE

# 다른 API 키는 불필요 (Claude만 사용)
# GOOGLE_API_KEY=  # 사용 안 함
# GROQ_API_KEY=     # 사용 안 함
# OPENROUTER_API_KEY=  # 사용 안 함
EOF
```

**중요**: `YOUR_SESSION_KEY_HERE`를 실제 세션 키로 교체하세요!

---

## 🎯 3. Opus 4.5/4.6 사용 설정

### 세션 키 방식의 모델 선택

세션 키는 **claude.ai 웹사이트에서 선택한 모델**을 사용합니다.

**설정 방법**:
1. https://claude.ai 접속
2. 새 대화 시작
3. 모델 선택 드롭다운 클릭
4. **"Claude Opus 4.6"** 또는 **"Claude Opus 4.5"** 선택
5. 메시지 1개 보내기 (모델 활성화)
6. 이 상태로 세션 키 사용 → Opus 사용됨!

**확인 방법**:
- claude.ai에서 대화 시작 시 모델 이름 확인
- "Claude Opus 4.6" 표시되면 OK

---

## ✅ 4. 설정 확인

```bash
cd ~/Fleming-AI

# 세션 키 확인
cat .env | grep CLAUDE_SESSION_KEY

# 실행 테스트
python scripts/run_full_research.py
```

**성공 로그**:
```
BackendSwitcher initialized with Claude only (Opus 4.5/4.6)
ClaudeClient initialized with session key authentication
✓ Success with claude
```

**실패 시**:
```
✗ claude failed: Invalid session key
```
→ 세션 키가 잘못되었거나 만료됨

---

## 🚨 5. 실패 시 대처 방법

### 에러: "Invalid session key"

**원인**: 세션 키 만료 또는 잘못된 키

**해결**:
1. claude.ai에서 로그아웃
2. 다시 로그인
3. 새 세션 키 복사
4. `.env` 파일 업데이트

```bash
# .env 파일 수정
nano .env  # 또는 vim .env

# CLAUDE_SESSION_KEY 값을 새 키로 교체
```

### 에러: "Cannot send a request, as the client has been closed"

**원인**: 병렬 호출 버그 (이미 수정됨)

**해결**:
```bash
# 최신 코드로 업데이트
cd ~/Fleming-AI
git pull

# 다시 실행
python scripts/run_full_research.py
```

### 에러: "All backends failed"

**원인**: Claude 세션 키가 설정되지 않음

**해결**:
```bash
# .env 파일 확인
cat .env

# CLAUDE_SESSION_KEY가 없으면 추가
echo "CLAUDE_SESSION_KEY=sk-ant-sid01-..." >> .env
```

---

## 🗑️ 6. 세션 키 제거 방법

### 임시 비활성화 (파일 유지)

```bash
cd ~/Fleming-AI

# .env 파일에서 주석 처리
sed -i.bak 's/^CLAUDE_SESSION_KEY=/#CLAUDE_SESSION_KEY=/' .env

# 확인
cat .env
```

### 완전 제거

```bash
cd ~/Fleming-AI

# .env 파일에서 해당 줄 삭제
grep -v "CLAUDE_SESSION_KEY" .env > .env.tmp && mv .env.tmp .env

# 또는 .env 파일 전체 삭제
rm .env
```

### 세션 키 무효화 (보안)

```bash
# 1. https://claude.ai 접속
# 2. 로그아웃
# 3. 다시 로그인
# → 이전 세션 키는 자동으로 무효화됨
```

---

## 💡 7. 팁

### OpenCode와 동시 사용

**옵션 1**: 같은 세션 키 사용 (rate limit 공유)
- Fleming 실행 중에도 OpenCode 사용 가능
- 단, 둘 다 느려질 수 있음

**옵션 2**: 별도 세션 키 사용 (권장)
- 다른 브라우저(Safari, Firefox)에서 claude.ai 로그인
- 새 세션 키 발급
- Fleming에만 새 키 사용

### 세션 키 유효 기간

- 일반적으로 **수일~수주** 유효
- 로그아웃하면 즉시 무효화
- 만료 시 새로 발급 필요

### 보안 주의사항

- `.env` 파일을 Git에 커밋하지 마세요
- 세션 키를 공개 저장소에 올리지 마세요
- 사용 후 로그아웃하면 키 무효화됨

---

## 📞 문제 해결

### 로그 확인

```bash
cd ~/Fleming-AI

# 최근 실행 로그 확인
tail -100 runs/*/pipeline.log | grep -i "claude\|error\|failed"
```

### 디버그 모드

```bash
# 환경 검증만 실행 (dry-run)
python scripts/run_full_research.py --dry-run
```

### 세션 키 테스트

```bash
# Python으로 직접 테스트
python -c "
import asyncio
import os
os.environ['CLAUDE_SESSION_KEY'] = 'sk-ant-sid01-YOUR_KEY_HERE'
from src.llm.claude_client import ClaudeClient

async def test():
    client = ClaudeClient()
    result = await client.generate('Hello', max_tokens=10)
    print('Success:', result)

asyncio.run(test())
"
```

---

## ✅ 체크리스트

설정 완료 전 확인:

- [ ] claude.ai에서 Opus 4.6 모델 선택
- [ ] 세션 키 복사 (sk-ant-sid01-로 시작)
- [ ] `.env` 파일에 `CLAUDE_SESSION_KEY` 설정
- [ ] `python scripts/run_full_research.py` 실행 성공
- [ ] 로그에 "Claude only (Opus 4.5/4.6)" 표시 확인

모두 체크되면 준비 완료! 🎉
