#!/usr/bin/env python3
"""
Live test of Alex reviewing the actual ViT paper.
Tests Alex's ability to identify weaknesses in a real research paper.
"""

import asyncio
from pathlib import Path
from src.reviewers.alex import Alex
from src.llm.groq_client import GroqClient


async def test_alex_on_real_paper():
    paper_path = Path("experiment/paper/paper.tex")

    if not paper_path.exists():
        print(f"ERROR: Paper not found at {paper_path}")
        return

    paper_text = paper_path.read_text()
    paper_preview = paper_text[:12000]

    print("=" * 80)
    print("LIVE TEST: Alex reviewing ViT paper")
    print("=" * 80)
    print(f"\nPaper length: {len(paper_text)} chars")
    print(f"Using preview: {len(paper_preview)} chars")
    print("\nCalling Alex...\n")

    async with GroqClient() as groq:
        alex = Alex(groq)
        result = await alex.review_paper(paper_preview)

        print("=" * 80)
        print("ALEX REVIEW RESULTS")
        print("=" * 80)
        print(f"\nVerdict: {result.verdict}")
        print(f"\nScores:")
        for score_name, score_value in result.scores.items():
            print(f"  {score_name}: {score_value:.2f}")

        print(f"\nStrengths ({len(result.strengths)}):")
        for i, strength in enumerate(result.strengths, 1):
            print(f"  {i}. {strength}")

        print(f"\nWeaknesses ({len(result.weaknesses)}):")
        for i, weakness in enumerate(result.weaknesses, 1):
            print(f"  {i}. {weakness}")

        print(f"\nQuestions ({len(result.questions)}):")
        for i, question in enumerate(result.questions, 1):
            print(f"  {i}. {question}")

        print(f"\nSuggestions ({len(result.suggestions)}):")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"  {i}. {suggestion}")

        print("\n" + "=" * 80)
        print("TEST VALIDATION")
        print("=" * 80)

        avg_score = sum(result.scores.values()) / len(result.scores) if result.scores else 0

        print(f"\n✓ Review completed successfully")
        print(f"✓ Weaknesses found: {len(result.weaknesses)}")
        print(f"✓ Average score: {avg_score:.2f}")

        if len(result.weaknesses) >= 2:
            print(f"✓ PASS: Found ≥2 weaknesses (expected for live test)")
        else:
            print(f"⚠ WARN: Found <2 weaknesses (expected ≥2)")

        missing_limitations_mentioned = any(
            "limitation" in w.lower() for w in result.weaknesses + result.suggestions
        )

        overclaiming_mentioned = any(
            "claim" in w.lower() or "overstate" in w.lower()
            for w in result.weaknesses + result.suggestions
        )

        statistical_mentioned = any(
            "statistic" in w.lower() or "significance" in w.lower() or "effect size" in w.lower()
            for w in result.strengths + result.weaknesses + result.suggestions
        )

        print(f"\nContent analysis:")
        print(f"  - Mentions limitations: {missing_limitations_mentioned}")
        print(f"  - Mentions overclaiming: {overclaiming_mentioned}")
        print(f"  - Mentions statistics: {statistical_mentioned}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_alex_on_real_paper())
