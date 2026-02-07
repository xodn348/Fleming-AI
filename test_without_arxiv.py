"""
arXiv 없이 Fleming-AI 테스트
- DB에 있는 106개 논문으로 가설 생성
- 검증 파이프라인 테스트
- Google Drive 동기화
"""
import asyncio
from src.llm.ollama_client import OllamaClient
from src.storage.vectordb import VectorDB
from src.filters.quality import QualityFilter
from src.generators.hypothesis import HypothesisGenerator
from src.validators.pipeline import ValidationPipeline
from src.storage.hypothesis_db import HypothesisDatabase
from src.storage.gdrive import sync_to_drive

async def main():
    print("=== Fleming-AI (arXiv 없이 작동) ===\n")
    
    # 1. VectorDB에서 가설 생성
    print("1. Generating hypotheses from VectorDB...")
    async with OllamaClient() as ollama:
        vectordb = VectorDB()
        qf = QualityFilter()
        hg = HypothesisGenerator(ollama, vectordb, qf)
        
        hypotheses = await hg.generate_hypotheses(
            query="deep learning transformers",
            k=5
        )
        print(f"   ✓ Generated {len(hypotheses)} hypotheses")
        
        # 2. 첫 가설 검증
        if hypotheses:
            print("\n2. Validating first hypothesis...")
            with HypothesisDatabase() as db:
                pipeline = ValidationPipeline(ollama, db)
                result = await pipeline.validate(hypotheses[0])
                print(f"   ✓ Status: {result.status}")
                print(f"   ✓ Classification: {result.classification}")
        
        # 3. Google Drive 동기화
        print("\n3. Syncing to Google Drive...")
        sync_result = sync_to_drive("data/db", "Fleming-AI/db")
        if sync_result:
            print("   ✓ Synced to Google Drive")
    
    print("\n✅ Full pipeline works without arXiv!")

if __name__ == "__main__":
    asyncio.run(main())
