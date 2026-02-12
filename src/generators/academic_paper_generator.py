"""
Academic Paper Generator for Fleming-AI
Generates research paper sections using Groq LLM
"""

import asyncio
import logging
from typing import Any

from src.generators.hypothesis import Hypothesis

logger = logging.getLogger(__name__)


class AcademicPaperGenerator:
    """Generate academic paper sections from validated hypotheses using Groq LLM"""

    def __init__(self, groq_client: Any, hypotheses_db: Any):
        """
        Initialize paper generator

        Args:
            groq_client: GroqClient instance
            hypotheses_db: HypothesisDatabase instance
        """
        self.groq = groq_client
        self.db = hypotheses_db

    def _calculate_statistics(self, hypotheses: list[Hypothesis]) -> dict[str, Any]:
        """Calculate statistics from hypotheses"""
        if not hypotheses:
            return {
                "total": 0,
                "avg_confidence": 0.0,
                "avg_quality": 0.0,
                "domains": [],
            }

        total = len(hypotheses)
        avg_conf = sum(h.confidence for h in hypotheses) / total
        avg_qual = sum(h.quality_score for h in hypotheses) / total

        # Extract domains from connections
        domains = set()
        for h in hypotheses:
            if h.connection:
                domains.add(h.connection.get("concept_a", "Unknown")[:20])
                domains.add(h.connection.get("concept_b", "Unknown")[:20])

        return {
            "total": total,
            "avg_confidence": avg_conf,
            "avg_quality": avg_qual,
            "domains": list(domains)[:5],  # Top 5 domains
        }

    async def generate_abstract(self, hypotheses: list[Hypothesis]) -> str:
        """Generate 200-250 word abstract"""
        stats = self._calculate_statistics(hypotheses)

        prompt = f"""Write an academic abstract for a research paper presenting {stats["total"]} novel scientific hypotheses generated via Literature-Based Discovery (LBD).

Key statistics:
- Total validated hypotheses: {stats["total"]}
- Average confidence score: {stats["avg_confidence"]:.2f}
- Average quality score: {stats["avg_quality"]:.2f}
- Research domains: {", ".join(stats["domains"])}
- Source papers analyzed: 1,011

The paper describes:
1. A computational LBD system using Swanson's ABC model
2. Automated concept extraction and pattern matching
3. LLM-based hypothesis generation and validation
4. Novel cross-domain connections discovered

Write in formal academic style, 200-250 words. Focus on methodology and key findings."""

        await asyncio.sleep(2)  # Rate limit
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=400)

    async def generate_introduction(self) -> str:
        """Generate introduction section"""
        prompt = """Write the Introduction section for an academic paper on Literature-Based Discovery (LBD).

Structure:
1. Background on LBD and its importance
2. Don Swanson's ABC model (1986) - fish oil and Raynaud's disease example
3. Challenges in manual LBD (information overload, cognitive limits)
4. Our contribution: Automated LBD using LLMs and vector databases
5. Paper organization overview

Length: 800-1000 words
Style: Formal academic, cite Swanson (1986)
Tone: Motivate the problem, highlight innovation"""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.4, max_tokens=1500)

    async def generate_methods(self) -> str:
        """Generate methods section"""
        prompt = """Write the Methods section for a computational Literature-Based Discovery system.

Subsections:
1. Data Collection: OpenAlex API, 1,011 PDFs, Citation > 50
2. Concept Extraction: Groq Llama 3.3 70B, temperature=0.3
3. ABC Pattern Detection: Concept graph, bridging concepts
4. Hypothesis Generation: Groq Llama 3.3 70B, temperature=0.5
5. Validation Pipeline: Novelty, plausibility, quality checks

Length: 1200-1500 words
Style: Detailed, reproducible, technical"""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=2000)

    async def generate_results(self, hypotheses: list[Hypothesis]) -> str:
        """Generate results section"""
        stats = self._calculate_statistics(hypotheses)
        top_10 = sorted(
            hypotheses, key=lambda h: (h.confidence + h.quality_score) / 2, reverse=True
        )[:10]

        # Format top hypotheses
        top_text = "\n".join(
            [
                f"{i + 1}. {h.hypothesis_text[:100]}... (Conf: {h.confidence:.2f}, Qual: {h.quality_score:.2f})"
                for i, h in enumerate(top_10)
            ]
        )

        prompt = f"""Write the Results section for a Literature-Based Discovery paper.

Statistics:
- Total hypotheses: {stats["total"]}
- Avg confidence: {stats["avg_confidence"]:.2f}
- Avg quality: {stats["avg_quality"]:.2f}

Top 10 hypotheses:
{top_text}

Structure:
1. Overview of hypothesis generation statistics
2. Validation rate and quality metrics
3. Domain distribution analysis
4. Detailed presentation of top 10 hypotheses

Length: 1500-2000 words
Style: Objective, data-driven"""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=2500)

    async def generate_discussion(self, hypotheses: list[Hypothesis]) -> str:
        """Generate discussion section"""
        high_conf = [h for h in hypotheses if h.confidence > 0.7]

        prompt = f"""Write the Discussion section for a Literature-Based Discovery paper.

Key findings:
- {len(high_conf)} high-confidence hypotheses (>0.7)
- Novel cross-domain connections discovered

Structure:
1. Novel discoveries and significance
2. Comparison with manual LBD
3. Limitations: English only, LLM-dependent, no experimental validation
4. Future work: Expand dataset, integrate experiments, collaborate with experts

Length: 1000-1200 words
Style: Critical analysis, balanced"""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.4, max_tokens=1800)

    async def generate_conclusion(self, hypotheses: list[Hypothesis]) -> str:
        """Generate conclusion section"""
        total = len(hypotheses)

        prompt = f"""Write the Conclusion section for a Literature-Based Discovery paper.

Key points:
- Generated {total} validated hypotheses from 1,011 papers
- Demonstrated AI-assisted scientific discovery feasibility
- Identified non-obvious cross-domain connections
- Provided testable hypotheses for validation

Length: 300-400 words
Style: Concise, impactful"""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=600)

    def generate_appendix(self, hypotheses: list[Hypothesis]) -> str:
        """Generate appendix with all hypotheses"""
        lines = []
        for i, h in enumerate(hypotheses, 1):
            lines.append(f"\\subsection{{Hypothesis {i}}}")
            lines.append(f"\\textbf{{Confidence:}} {h.confidence:.2f} | ")
            lines.append(f"\\textbf{{Quality:}} {h.quality_score:.2f}\n")
            lines.append(f"\\textbf{{Statement:}} {h.hypothesis_text}\n")

            if h.connection:
                lines.append(f"\\textbf{{Connection:}}")
                lines.append(f"\\begin{{itemize}}")
                lines.append(f"\\item Concept A: {h.connection.get('concept_a', 'N/A')}")
                lines.append(f"\\item Bridging: {h.connection.get('bridging_concept', 'N/A')}")
                lines.append(f"\\item Concept B: {h.connection.get('concept_b', 'N/A')}")
                lines.append(f"\\end{{itemize}}\n")

        return "\n".join(lines)

    async def generate_bibliography(self, source_papers: list[dict]) -> str:
        """Generate BibTeX bibliography"""
        entries = []

        # Swanson reference
        entries.append(
            """@article{swanson1986,
  author = {Swanson, Don R.},
  title = {Fish oil, Raynaud's syndrome, and undiscovered public knowledge},
  journal = {Perspectives in Biology and Medicine},
  volume = {30},
  number = {1},
  pages = {7--18},
  year = {1986}
}"""
        )

        # Source papers (sample)
        for i, paper in enumerate(source_papers[:50]):
            title = (paper.get("title") or "Untitled").replace("{", "").replace("}", "")
            authors = (paper.get("authors") or "Unknown").replace("{", "").replace("}", "")
            year = paper.get("year", 2024)
            doi = paper.get("doi") or ""

            entry = f"""@article{{paper{i},
  author = {{{authors}}},
  title = {{{title}}},
  year = {{{year}}},
  doi = {{{doi}}}
}}"""
            entries.append(entry)

        return "\n\n".join(entries)
