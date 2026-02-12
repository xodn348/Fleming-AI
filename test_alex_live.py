#!/usr/bin/env python3
"""
Live test of Alex reviewer with ViT paper using BackendSwitcher
Tests Gemini (if available) -> Groq -> OpenRouter fallback
"""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from src.llm.backend_switcher import BackendSwitcher
from src.reviewers.alex import Alex

load_dotenv()


async def main():
    print("=" * 80)
    print("LIVE ALEX REVIEW TEST - ViT Paper")
    print("=" * 80)

    paper_path = Path("experiment/paper/paper.tex")

    if not paper_path.exists():
        print(f"Error: {paper_path} not found")
        return

    with open(paper_path) as f:
        paper_content = f.read()

    print(f"\n📄 Paper loaded: {len(paper_content)} chars")
    print(f"📝 First 500 chars:\n{paper_content[:500]}...\n")

    async with BackendSwitcher() as switcher:
        alex = Alex(switcher)

        print("🤖 Initializing Alex reviewer with BackendSwitcher...")
        print(f"   Available backends: Gemini -> Groq -> OpenRouter\n")

        print("🔍 Starting paper review (stage: paper)...")
        print("   This will take 30-60 seconds...\n")

        try:
            review_result = await alex.review(
                stage="paper", artifact=paper_content[:8000], conversation_history=[]
            )

            active_backend = await switcher.get_active_backend()
            backend_status = switcher.get_backend_status()

            print("\n" + "=" * 80)
            print("REVIEW COMPLETE")
            print("=" * 80)
            print(f"\n✓ Backend used: {active_backend}")
            print(f"✓ Backend status: {backend_status}")
            print(f"\n📊 Review Result:")
            print(f"   Verdict: {review_result.verdict}")
            print(f"   Scores: {review_result.scores}")
            print(f"\n💪 Strengths ({len(review_result.strengths)}):")
            for i, strength in enumerate(review_result.strengths[:3], 1):
                print(f"   {i}. {strength}")
            if len(review_result.strengths) > 3:
                print(f"   ... and {len(review_result.strengths) - 3} more strengths")
            print(f"\n⚠️ Weaknesses ({len(review_result.weaknesses)}):")
            for i, weakness in enumerate(review_result.weaknesses[:3], 1):
                print(f"   {i}. {weakness}")
            if len(review_result.weaknesses) > 3:
                print(f"   ... and {len(review_result.weaknesses) - 3} more weaknesses")

            output_path = Path("experiment/alex_review_live.json")
            output_data = {
                "backend": active_backend,
                "backend_status": backend_status,
                "verdict": review_result.verdict,
                "scores": review_result.scores,
                "strengths": review_result.strengths,
                "weaknesses": review_result.weaknesses,
                "questions": review_result.questions,
                "suggestions": review_result.suggestions,
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\n💾 Full review saved to: {output_path}")

        except Exception as e:
            print(f"\n❌ Review failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
