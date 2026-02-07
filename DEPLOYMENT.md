# Fleming-AI Deployment Guide

## 🚀 Quick Start

### Prerequisites
```bash
# 1. Set up API keys (choose one)
export OPENALEX_EMAIL="your-email@example.com"  # Polite pool access
# OR
export OPENALEX_API_KEY="your-api-key"

# Optional but recommended
export S2_API_KEY="your-semantic-scholar-key"
```

### First Time Setup
```bash
cd ~/Fleming-AI

# 1. Verify installation
python -c "from src.storage.vectordb import VectorDB; print('✓ Dependencies OK')"

# 2. Check database status
sqlite3 data/db/papers.db "SELECT COUNT(*) FROM great_papers;"
# Should show: 106

# 3. Verify enrichment
sqlite3 data/db/papers.db "SELECT COUNT(*) FROM great_papers WHERE citations > 0;"
# Should show: 86 (81% enriched)
```

## 📋 Usage

### Manual Collection
```bash
# Test mode (limit=10, faster)
python main.py collect --limit 10 --test-mode

# Production mode (limit=50)
python main.py collect --limit 50
```

### Automated Weekly Collection
```bash
# Start background scheduler
nohup python scripts/schedule_collection.py --frequency weekly > logs/scheduler.log 2>&1 &

# Check status
ps aux | grep schedule_collection

# View logs
tail -f logs/scheduler.log
```

### Cron Job Alternative
```cron
# Add to crontab (crontab -e)
# Runs every Monday at 2 AM
0 2 * * 1 cd ~/Fleming-AI && python scripts/schedule_collection.py --once >> logs/collection.log 2>&1
```

## 🧠 Self-Improvement Mode

### Using AdaptiveCollector
```python
import asyncio
from src.intelligence.adaptive_collector import AdaptiveCollector

async def run():
    collector = AdaptiveCollector()
    
    # Collect with automatic threshold optimization
    result = await collector.collect_with_learning()
    
    print(f"Discovered: {result['discovered']}")
    print(f"Stored: {result['stored']}")
    
    # Get performance report
    report = await collector.get_performance_report()
    print(f"Success Rate: {report['success_rate']:.1%}")
    print(f"Recommendations: {report['recommendations']}")

asyncio.run(run())
```

### A/B Testing
```python
# Test two strategies
test_id = await collector.start_ab_test(
    "high_quality",  # Higher quality threshold
    "high_volume"    # More papers, lower threshold
)

# After some collection cycles...
results = await collector.end_ab_test(test_id)
print(f"Winner: {results['winner']}")
```

## 🧪 Testing

### Run E2E Tests
```bash
python scripts/test_e2e.py
```

### Manual Component Testing
```bash
# Test scoring
python -c "from src.utils.scoring import calculate_quality_score; print(calculate_quality_score({'year': 2020, 'citations': 500, 'venue': 'NeurIPS'}))"

# Test database
python -c "from src.storage.database import PaperDatabase; db = PaperDatabase('data/db/papers.db'); print(len(db.get_all_papers()))"

# Test VectorDB
python -c "from src.storage.vectordb import VectorDB; db = VectorDB(); print(f'{db.count()} chunks')"
```

## 📊 Monitoring

### Check Collection Metrics
```bash
# View recent collections
sqlite3 data/db/metrics.db "SELECT * FROM collection_cycles ORDER BY cycle_id DESC LIMIT 5;"

# View threshold adjustments
sqlite3 data/db/metrics.db "SELECT * FROM threshold_history ORDER BY changed_at DESC LIMIT 10;"

# View venue performance
sqlite3 data/db/metrics.db "SELECT venue_name, success_rate FROM venue_performance ORDER BY success_rate DESC LIMIT 10;"
```

### System Health
```bash
# Check paper count growth
sqlite3 data/db/papers.db "SELECT COUNT(*), source FROM great_papers GROUP BY source;"

# Check hypothesis quality
sqlite3 data/db/hypotheses.db "SELECT status, COUNT(*) FROM hypotheses GROUP BY status;"
```

## 🔧 Troubleshooting

### "OpenAlex requires API key or email"
```bash
# Solution: Set environment variable
export OPENALEX_EMAIL="your-email@example.com"
# OR add to ~/.zshrc for persistence
echo 'export OPENALEX_EMAIL="your-email@example.com"' >> ~/.zshrc
```

### "Rate limit exceeded"
- Semantic Scholar: Wait 3 seconds between requests (automatic)
- OpenAlex: Use email for polite pool (5x higher limit)
- Solution: Already implemented in code

### "No papers in VectorDB"
```bash
# Re-embed papers
python scripts/embed_papers.py
```

## 🎯 Performance Tuning

### Adjust Collection Frequency
```bash
# Daily (aggressive)
python scripts/schedule_collection.py --frequency daily

# Monthly (conservative)
python scripts/schedule_collection.py --frequency monthly
```

### Adjust Quality Thresholds
```python
# In your collection script
collector = PaperCollector(config={
    'min_citations': 200,      # Default: 100
    'quality_threshold': 70     # Default: 60
})
```

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| Papers/collection | 50-100 candidates → 5-15 stored |
| Filter pass rate | 10-30% |
| API calls/minute | ~20 (rate limited) |
| Collection time | ~5-10 minutes |
| Success rate | 40-60% (hypotheses validated) |

## 🛡️ Best Practices

1. **Start Conservative**: Use test mode first
2. **Monitor Metrics**: Check `metrics.db` regularly
3. **Review Logs**: Look for API errors or anomalies
4. **Validate Hypotheses**: Run hypothesis validation to feed back to adaptive system
5. **Backup Database**: `cp data/db/papers.db data/db/papers.db.backup`

## 🆘 Support

If issues persist:
1. Check logs: `tail -f logs/collection.log`
2. Verify API keys: `env | grep -E "(OPENALEX|S2)"`
3. Test components individually (see Testing section)
4. Check system resources: `df -h`, `free -m`

---

**System is ready! Start with:**
```bash
export OPENALEX_EMAIL="your-email@example.com"
python main.py collect --limit 10 --test-mode
```
