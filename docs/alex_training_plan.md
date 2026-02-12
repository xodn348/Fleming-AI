# Alex Training Plan: Making Alex Smarter

**Goal**: Train Alex to provide higher-quality ML paper reviews

---

## 📊 Current State

**Alex v1.0**:
- Base: Llama 3.3 70B (Groq API, rate-limited)
- Knowledge: Prompt engineering (4 stage prompts, ~4K chars each)
- Performance: 0.72 overall score on test case
- Limitations: 
  - Generic reviews
  - No domain-specific expertise
  - Rate limit issues
  - No learning from feedback

---

## 🚀 Training Options (Ranked by Feasibility)

### Option 1: Local Fine-Tuning (LoRA) ⭐ **RECOMMENDED**

**What**: Fine-tune smaller open model (7B-14B) with LoRA on review data

**Why**:
- ✅ You control everything (no API limits)
- ✅ Fast: ~2-4 hours on M1 Pro (with Unsloth)
- ✅ Cheap: 100% free
- ✅ Iterative: retrain as you collect more data

**How**:
```bash
# 1. Collect data (500-1000 reviews)
python scripts/collect_openreview_data.py

# 2. Fine-tune with Unsloth
python scripts/train_alex_lora.py \
  --model "unsloth/llama-3.2-8b" \
  --data reviews_dataset.json \
  --epochs 3 \
  --lora_rank 16

# 3. Merge & export
python scripts/export_alex_model.py

# 4. Use in Fleming-AI
# Update GroqClient → LocalModel in orchestrator.py
```

**Data Sources**:
1. **OpenReview** (NeurIPS, ICLR, ICML) - 100K+ reviews
2. **ArXiv papers + reviews** - pairs available
3. **Your own reviews** - start with 10-20 manual examples

**Cost**: $0 (uses local compute)
**Time**: 
- Data collection: 2-4 hours
- Training: 2-4 hours (M1 Pro)
- Integration: 1 hour
- **Total**: ~1 day

**Quality Gain**: +20-30% (estimated, based on similar fine-tuning results)

---

### Option 2: Prompt + RAG (No Training) ⚡ **FASTEST**

**What**: Add retrieval of similar reviews (no model training)

**Why**:
- ✅ Immediate (implement in 2-3 hours)
- ✅ No training needed
- ✅ Improves quality +10-15%

**How**:
```python
# In alex.py review method:
# 1. Search OpenReview for similar papers
similar_reviews = search_openreview(paper_title, paper_abstract)

# 2. Add to prompt
prompt = f"""
{REVIEW_PROMPT}

REFERENCE REVIEWS (similar papers):
{similar_reviews[:3]}  # Top 3 matches

NOW REVIEW THIS PAPER:
{paper_text}
"""
```

**Implementation**:
- Use OpenReview API (free)
- Simple BM25/TF-IDF search (no embeddings needed)
- 2-3 hours to implement

**Cost**: $0
**Time**: 2-3 hours
**Quality Gain**: +10-15%

---

### Option 3: Distillation from GPT-4/Claude

**What**: Use your Claude Max to generate training data, then fine-tune small model

**Why**:
- Better than pure prompt engineering
- Leverages your Claude Max subscription
- Creates reusable training data

**How**:
```python
# 1. Generate synthetic reviews with Claude
for paper in papers_to_review:
    review = claude_max_web_interface.review(paper)
    save_to_dataset(paper, review)

# 2. Fine-tune small model on synthetic data
train_lora(model="llama-3.2-8b", data=claude_reviews)

# 3. Result: Small model that mimics Claude Max quality
```

**Limitation**: Claude Max is web-only (no API), so manual process

**Cost**: $0 (use existing subscription)
**Time**: 
- Manual data gen: 3-5 hours (20-50 reviews)
- Training: 2-4 hours
- **Total**: 1 day

**Quality Gain**: +30-40% (Claude-quality reviews at local speed)

---

### Option 4: Bigger Model (No Training)

**What**: Switch to larger/better model

**Options**:
- Groq Llama 3.3 70B → Groq Llama 90B (when available)
- Add OpenRouter (free tier: Claude 3.5 Sonnet, GPT-4o)
- Local Qwen 72B (slow on M1 Pro, but possible)

**Pros**: Simple (just change API endpoint)
**Cons**: Still rate-limited, not "yours"

---

## 📁 Data Collection (For Fine-Tuning)

### Source 1: OpenReview (Best Quality)

**Script**:
```python
# scripts/collect_openreview_data.py
import requests
from pathlib import Path

def collect_openreview_reviews(conference="NeurIPS.cc/2024", limit=1000):
    """Collect paper+review pairs from OpenReview."""
    reviews = []
    
    # OpenReview API
    base_url = "https://api.openreview.net/notes"
    params = {
        "invitation": f"{conference}/Conference/-/Blind_Submission",
        "limit": limit
    }
    
    papers = requests.get(base_url, params=params).json()["notes"]
    
    for paper in papers:
        paper_id = paper["id"]
        
        # Get reviews for this paper
        review_params = {
            "forum": paper_id,
            "invitation": f"{conference}/Conference/Paper.*/-/Official_Review"
        }
        paper_reviews = requests.get(base_url, params=review_params).json()["notes"]
        
        for review in paper_reviews:
            reviews.append({
                "paper_title": paper["content"]["title"],
                "paper_abstract": paper["content"]["abstract"],
                "paper_text": paper["content"].get("pdf", ""),
                "review": review["content"],
                "rating": review["content"].get("rating", ""),
                "confidence": review["content"].get("confidence", "")
            })
    
    # Save to JSONL
    output = Path("data/openreview_reviews.jsonl")
    output.parent.mkdir(exist_ok=True)
    
    with open(output, "w") as f:
        for review in reviews:
            f.write(json.dumps(review) + "\n")
    
    return reviews

# Run
reviews = collect_openreview_reviews("NeurIPS.cc/2024", limit=1000)
print(f"Collected {len(reviews)} reviews")
```

**Expected Output**: 1000 review pairs in ~10 minutes

---

### Source 2: Your Own Manual Reviews

**Bootstrap Strategy**:
```python
# Start with 10-20 manual high-quality reviews
manual_reviews = [
    {
        "paper": "ViT Pre-training paper",
        "review": alex_review_vit_paper.json,  # Today's review
        "quality": "high"
    },
    # ... add 9-19 more
]

# Use these as few-shot examples + fine-tuning seed data
```

---

## 🛠️ Training Script (LoRA Fine-Tuning)

**Using Unsloth** (optimized for M1/M2/M3):

```python
# scripts/train_alex_lora.py
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

# 1. Load base model (8B fits on M1 Pro)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-8b-instruct",
    max_seq_length=4096,
    dtype=None,  # Auto-detect
    load_in_4bit=True  # 4-bit quantization for memory
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)

# 3. Load review dataset
dataset = load_dataset("json", data_files="data/openreview_reviews.jsonl")

def format_review_prompt(example):
    """Convert review pair to instruction format."""
    return {
        "text": f"""<|im_start|>system
You are Alex, an expert ML paper reviewer trained on NeurIPS standards.<|im_end|>
<|im_start|>user
Review this paper:

Title: {example['paper_title']}
Abstract: {example['paper_abstract']}

Provide a structured review with: strengths, weaknesses, questions, rating.<|im_end|>
<|im_start|>assistant
{example['review']}<|im_end|>"""
    }

dataset = dataset.map(format_review_prompt)

# 4. Train
from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        output_dir="outputs/alex_lora",
        logging_steps=10,
        save_steps=100,
    ),
)

trainer.train()

# 5. Save
model.save_pretrained("models/alex_finetuned")
tokenizer.save_pretrained("models/alex_finetuned")
```

**Expected Time on M1 Pro**:
- 500 reviews: ~2 hours
- 1000 reviews: ~4 hours
- 5000 reviews: ~12 hours

---

## 📊 Expected Quality Improvements

| Method | Time | Cost | Quality Gain | Notes |
|--------|------|------|--------------|-------|
| **LoRA Fine-tuning** | 1 day | $0 | +20-30% | Best long-term |
| **RAG (no training)** | 3 hours | $0 | +10-15% | Quick win |
| **Claude distillation** | 1 day | $0 | +30-40% | Highest quality |
| **Bigger model** | 1 hour | $ varies | +5-10% | Simple but limited |

---

## 🎯 Recommended Path

**Week 1** (Quick wins):
1. ✅ Implement RAG (2-3 hours) → +10% quality
2. ✅ Collect OpenReview data (4 hours) → 1000 reviews
3. ✅ Add to prompts as few-shot → +5% quality

**Week 2** (Real training):
4. ✅ Fine-tune Llama 3.2 8B with LoRA (4 hours) → +20% quality
5. ✅ Integrate into Fleming-AI (2 hours)
6. ✅ Test on 5-10 real papers

**Week 3** (Iterate):
7. ✅ Collect feedback (what Alex misses)
8. ✅ Add to training data
9. ✅ Retrain → +10% more

**Total**: ~3 weeks to 2x Alex quality

---

## 💻 Hardware Requirements

**Your M1 Pro**:
- ✅ Can train 7-8B models with 4-bit LoRA
- ✅ 2-4 hours per training run (500-1000 reviews)
- ✅ Inference: 5-10 tokens/sec (acceptable)

**Alternative** (if M1 is slow):
- Google Colab (free): ~1-2 hours training (but need to export)
- RunPod ($0.30/hour): Faster, pay-as-you-go

---

## 🚀 Next Steps

1. **Choose path**: LoRA fine-tuning (recommended) or RAG (fastest)
2. **Collect data**: Run OpenReview scraper
3. **Train**: Use provided script
4. **Integrate**: Replace GroqClient with LocalModel
5. **Evaluate**: Review 5-10 papers, compare scores

---

## 📝 Want to Start Now?

**Option A**: Implement RAG first (3 hours, immediate improvement)
**Option B**: Collect data + train LoRA (1 day, bigger improvement)
**Option C**: Both (RAG while training happens)

**Which path do you want to take?**
