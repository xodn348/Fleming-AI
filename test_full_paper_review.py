#!/usr/bin/env python3
"""
Full paper review test using SingleCallReviewer with complete 27K paper.
Tests consolidated review with BackendSwitcher fallback (Gemini -> Groq -> OpenRouter).
"""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from src.pipeline.single_call_gate import SingleCallReviewer
from src.llm.backend_switcher import BackendSwitcher

load_dotenv()


async def main():
    print("=" * 80)
    print("FULL PAPER REVIEW TEST - SingleCallReviewer")
    print("=" * 80)

    paper_path = Path("experiment/paper/paper.tex")

    if not paper_path.exists():
        print(f"Error: {paper_path} not found")
        return

    # Load FULL paper (no truncation)
    paper_content = paper_path.read_text()
    print(f"\n📄 Loaded paper: {len(paper_content)} chars")
    print(f"📝 First 300 chars:\n{paper_content[:300]}...\n")

    async with BackendSwitcher() as switcher:
        reviewer = SingleCallReviewer(switcher)

        print("🤖 Initializing SingleCallReviewer with BackendSwitcher...")
        print(f"   Available backends: Gemini -> Groq -> OpenRouter\n")

        print("🔍 Starting full paper review (consolidated)...")
        print("   This will take 30-60 seconds...\n")

        try:
            # Use review_paper_full for consolidated review
            review_result = await reviewer.review_paper_full(paper_content)

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
            print(f"\n⚠️  Weaknesses ({len(review_result.weaknesses)}):")
            for i, weakness in enumerate(review_result.weaknesses[:2], 1):
                print(f"   {i}. {weakness}")
            if len(review_result.weaknesses) > 2:
                print(f"   ... and {len(review_result.weaknesses) - 2} more weaknesses")

            # Save output
            output_path = Path("experiment/alex_review_full_paper.json")
            output_data = {
                "review_id": review_result.review_id,
                "verdict": review_result.verdict,
                "strengths": review_result.strengths,
                "weaknesses": review_result.weaknesses,
                "questions": review_result.questions,
                "suggestions": review_result.suggestions,
                "scores": review_result.scores,
                "timestamp": review_result.timestamp,
            }

            output_path.write_text(json.dumps(output_data, indent=2))
            print(f"\n💾 Full review saved to: {output_path}")

        except Exception as e:
            print(f"\n❌ Review failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
