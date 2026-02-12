"""
Hypothesis Generator for Fleming-AI
Literature-Based Discovery using Don Swanson's ABC Model
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.pipeline.capabilities import CAPABILITIES
from src.pipeline.hypothesis_spec import HypothesisSpec

logger = logging.getLogger(__name__)

MAX_PATTERNS = 50


@dataclass
class Hypothesis:
    """Represents a generated hypothesis from Literature-Based Discovery."""

    id: str
    hypothesis_text: str
    source_papers: list[str]
    connection: dict[str, str]  # {concept_a, concept_b, bridging_concept}
    confidence: float
    quality_score: float
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, validated, rejected

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "hypothesis_text": self.hypothesis_text,
            "source_papers": self.source_papers,
            "connection": self.connection,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        """Create Hypothesis from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            hypothesis_text=data["hypothesis_text"],
            source_papers=data.get("source_papers", []),
            connection=data.get("connection", {}),
            confidence=data.get("confidence", 0.0),
            quality_score=data.get("quality_score", 0.0),
            created_at=created_at,
            status=data.get("status", "pending"),
        )


@dataclass
class ConceptPair:
    """Represents a pair of concepts with their bridging concept."""

    concept_a: str
    concept_b: str
    bridging_concept: str
    paper_a_id: str  # Paper linking A to bridging
    paper_b_id: str  # Paper linking bridging to B
    strength: float = 0.0


class HypothesisGenerator:
    """
    Generate hypotheses using Literature-Based Discovery (LBD).

    Implements Don Swanson's ABC Model:
    - Paper A: Concept X → Y connection
    - Paper B: Concept Y → Z connection
    - Hypothesis: X → Z connection (undiscovered)
    """

    def __init__(
        self,
        llm_client: Any,
        vectordb: Any,
        quality_filter: Any,
    ):
        """
        Initialize HypothesisGenerator with dependencies.

        Args:
            llm_client: LLM client for concept extraction and hypothesis generation (e.g., GroqClient)
            vectordb: VectorDB for paper retrieval
            quality_filter: QualityFilter for scoring hypotheses
        """
        self.llm = llm_client
        self.vectordb = vectordb
        self.quality_filter = quality_filter

    async def extract_concepts(self, text: str) -> list[str]:
        """
        Extract key concepts from paper text using LLM.

        Args:
            text: Paper text to extract concepts from

        Returns:
            List of extracted concepts
        """
        prompt = f"""Extract the key scientific concepts from this research paper text.
Return ONLY a JSON array of concept strings, no explanation.
Focus on:
- Key scientific terms and phenomena
- Methods and techniques
- Biological/chemical entities (if applicable)
- Diseases, conditions, or outcomes (if applicable)

Text:
{text[:3000]}

Return format: ["concept1", "concept2", ...]"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500,
            )
            # Parse JSON array from response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            concepts = json.loads(response)
            return concepts if isinstance(concepts, list) else []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to extract concepts: {e}")
            return []

    async def find_concept_connections(
        self,
        papers: list[dict[str, str]],
    ) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[str]]]:
        """
        Build a graph of concept connections from papers.

        Args:
            papers: List of paper data with 'paper_id' and 'text'

        Returns:
            Dict mapping concepts to lists of (connected_concept, paper_id)
        """
        concept_graph: dict[str, list[tuple[str, str]]] = {}
        paper_concepts: dict[str, list[str]] = {}

        sem = asyncio.Semaphore(1)  # Serial requests to avoid rate limits

        async def extract_for_paper(paper):
            async with sem:
                paper_id = paper.get("paper_id", paper.get("id", "unknown"))
                text = paper.get("text", paper.get("abstract", ""))
                concepts = await self.extract_concepts(text)
                return paper_id, concepts

        results = await asyncio.gather(
            *(extract_for_paper(p) for p in papers),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Concept extraction failed: {result}")
                continue
            paper_id, concepts = result
            paper_concepts[paper_id] = concepts

            # Build connections: concepts in same paper are connected
            for i, concept_a in enumerate(concepts):
                if concept_a not in concept_graph:
                    concept_graph[concept_a] = []
                for concept_b in concepts[i + 1 :]:
                    concept_graph[concept_a].append((concept_b, paper_id))
                    if concept_b not in concept_graph:
                        concept_graph[concept_b] = []
                    concept_graph[concept_b].append((concept_a, paper_id))

        return concept_graph, paper_concepts

    async def find_abc_patterns(
        self,
        concept_graph: dict[str, list[tuple[str, str]]],
        paper_concepts: dict[str, list[str]],
    ) -> list[ConceptPair]:
        """
        Find ABC patterns: A-B in one paper, B-C in another, but no A-C.

        Args:
            concept_graph: Graph of concept connections
            paper_concepts: Mapping of paper_id to concepts

        Returns:
            List of ConceptPair representing potential discoveries
        """
        abc_patterns = []
        processed = set()

        for bridging_concept, connections in concept_graph.items():
            if len(abc_patterns) >= MAX_PATTERNS:
                break
            # Group connections by paper
            papers_with_concept: dict[str, list[str]] = {}
            for connected_concept, paper_id in connections:
                if paper_id not in papers_with_concept:
                    papers_with_concept[paper_id] = []
                papers_with_concept[paper_id].append(connected_concept)

            paper_ids = list(papers_with_concept.keys())
            # Need at least 2 different papers
            if len(paper_ids) < 2:
                continue

            # Look for A-B-C patterns across different papers
            for i, paper_a_id in enumerate(paper_ids):
                for paper_b_id in paper_ids[i + 1 :]:
                    concepts_a = papers_with_concept[paper_a_id]
                    concepts_b = papers_with_concept[paper_b_id]

                    for concept_a in concepts_a:
                        for concept_b in concepts_b:
                            # Skip if same concept
                            if concept_a == concept_b:
                                continue

                            # Check if A-C connection already exists
                            pair_key = tuple(sorted([concept_a, concept_b]))
                            if pair_key in processed:
                                continue

                            # Check if A and C are directly connected in any paper
                            a_connections = {c for c, _ in concept_graph.get(concept_a, [])}
                            if concept_b not in a_connections:
                                # Found ABC pattern: A-B in paper_a, B-C in paper_b
                                abc_patterns.append(
                                    ConceptPair(
                                        concept_a=concept_a,
                                        concept_b=concept_b,
                                        bridging_concept=bridging_concept,
                                        paper_a_id=paper_a_id,
                                        paper_b_id=paper_b_id,
                                        strength=0.5,  # Default strength
                                    )
                                )
                                processed.add(pair_key)

        return abc_patterns

    async def generate_hypothesis_text(
        self,
        concept_pair: ConceptPair,
        paper_a_text: str,
        paper_b_text: str,
    ) -> dict[str, Any]:
        """
        Generate hypothesis using LLM in HypothesisSpec format.

        Args:
            concept_pair: The ABC pattern to generate hypothesis for
            paper_a_text: Text from paper A (A-B connection)
            paper_b_text: Text from paper B (B-C connection)

        Returns:
            Dict containing HypothesisSpec fields (hypothesis, confidence, task, dataset, baseline, variant, metric, expected_effect)
        """
        prompt = f"""CRITICAL CONSTRAINTS - Vision Experiments ONLY:
- Task: {CAPABILITIES["task"]} ONLY (image classification)
- Available models: {", ".join(CAPABILITIES["models"])}
- Available datasets: {", ".join(CAPABILITIES["datasets"])}
- Available metrics: {", ".join(CAPABILITIES["metrics"])}
- Do NOT propose text classification, NLP, or any non-vision tasks
- All models and datasets MUST be from the lists above

Based on Literature-Based Discovery (Swanson's ABC model):

Paper 1 discusses: "{concept_pair.concept_a}" and "{concept_pair.bridging_concept}"
Context: {paper_a_text[:1000]}

Paper 2 discusses: "{concept_pair.bridging_concept}" and "{concept_pair.concept_b}"
Context: {paper_b_text[:1000]}

The bridging concept "{concept_pair.bridging_concept}" connects these two research areas.

Generate a scientific hypothesis comparing two vision models on an image classification task.
The hypothesis should relate to the concepts above, but MUST use only available resources.

Requirements:
1. The hypothesis MUST use image classification tasks ONLY
2. Choose models from: {", ".join(CAPABILITIES["models"])}
3. Choose dataset from: {", ".join(CAPABILITIES["datasets"])}
4. Choose metric from: {", ".join(CAPABILITIES["metrics"])}
5. Rate your confidence (0.0-1.0) based on the strength of evidence

Return JSON with ALL 8 fields (HypothesisSpec format):
{{
    "hypothesis": "Natural language description relating to the concepts above",
    "confidence": 0.X,
    "task": "image_classification",
    "dataset": "one of available datasets",
    "baseline": {{"model": "one of available models", "pretrain": "imagenet"}},
    "variant": {{"model": "different available model", "pretrain": "imagenet"}},
    "metric": "one of available metrics",
    "expected_effect": {{"direction": "increase or decrease", "min_delta_points": X.X}}
}}"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=800,
            )
            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            result = json.loads(response)
            confidence = float(result.get("confidence", 0.5))
            result["confidence"] = min(max(confidence, 0.0), 1.0)
            return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to generate hypothesis: {e}")
            fallback_hypothesis = (
                f"There may be a connection between {concept_pair.concept_a} and "
                f"{concept_pair.concept_b} mediated by {concept_pair.bridging_concept}."
            )
            return {
                "hypothesis": fallback_hypothesis,
                "confidence": 0.3,
                "task": CAPABILITIES["task"],
                "dataset": CAPABILITIES["datasets"][0],
                "baseline": {"model": CAPABILITIES["models"][0], "pretrain": "imagenet"},
                "variant": {
                    "model": CAPABILITIES["models"][1]
                    if len(CAPABILITIES["models"]) > 1
                    else CAPABILITIES["models"][0],
                    "pretrain": "imagenet",
                },
                "metric": CAPABILITIES["metrics"][0],
                "expected_effect": {"direction": "increase", "min_delta_points": 1.0},
            }

    async def generate_from_papers(
        self,
        paper_ids: list[str],
    ) -> list[Hypothesis]:
        """
        Generate hypotheses from a list of paper IDs.

        Uses Literature-Based Discovery:
        - Paper A: Concept X → Y connection
        - Paper B: Concept Y → Z connection
        - Hypothesis: X → Z connection possible?

        Args:
            paper_ids: List of paper IDs to analyze

        Returns:
            List of generated Hypothesis objects
        """
        # Retrieve papers from VectorDB
        papers = []
        for paper_id in paper_ids:
            paper_data = self.vectordb.get_paper(paper_id)
            if paper_data and paper_data.get("chunks"):
                # Combine chunks into full text
                text = "\n".join(chunk["text"] for chunk in paper_data["chunks"])
                papers.append(
                    {
                        "paper_id": paper_id,
                        "text": text,
                        "title": paper_data["chunks"][0].get("metadata", {}).get("title", ""),
                    }
                )

        if len(papers) < 2:
            logger.warning("Need at least 2 papers for LBD")
            return []

        # Build concept graph
        concept_graph, paper_concepts = await self.find_concept_connections(papers)

        # Find ABC patterns
        abc_patterns = await self.find_abc_patterns(concept_graph, paper_concepts)

        # Generate hypotheses for top patterns
        sem = asyncio.Semaphore(1)  # Serial requests to avoid rate limits

        async def gen_one(pattern):
            async with sem:
                paper_a = next((p for p in papers if p["paper_id"] == pattern.paper_a_id), None)
                paper_b = next((p for p in papers if p["paper_id"] == pattern.paper_b_id), None)

                if not paper_a or not paper_b:
                    return None

                hypothesis_spec = await self.generate_hypothesis_text(
                    pattern,
                    paper_a.get("text", ""),
                    paper_b.get("text", ""),
                )

                if not hypothesis_spec or not hypothesis_spec.get("hypothesis"):
                    return None

                hypothesis_text = hypothesis_spec["hypothesis"]
                confidence = hypothesis_spec["confidence"]

                quality_score = self.quality_filter.score(hypothesis_text)

                return Hypothesis(
                    id=str(uuid.uuid4()),
                    hypothesis_text=hypothesis_text,
                    source_papers=[pattern.paper_a_id, pattern.paper_b_id],
                    connection={
                        "concept_a": pattern.concept_a,
                        "concept_b": pattern.concept_b,
                        "bridging_concept": pattern.bridging_concept,
                    },
                    confidence=confidence,
                    quality_score=quality_score,
                )

        results = await asyncio.gather(
            *(gen_one(p) for p in abc_patterns[:10]),
            return_exceptions=True,
        )

        hypotheses = [r for r in results if isinstance(r, Hypothesis)]
        hypotheses.sort(
            key=lambda h: (h.confidence + h.quality_score) / 2,
            reverse=True,
        )

        return hypotheses

    async def generate_hypotheses(
        self,
        query: str,
        k: int = 10,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses based on a research query.

        Steps:
        1. Search VectorDB for relevant papers
        2. Extract concepts from papers
        3. Find unconnected concept pairs (ABC patterns)
        4. Generate hypotheses using LLM
        5. Score with Quality Filter

        Args:
            query: Research query to find related papers
            k: Number of papers to retrieve

        Returns:
            List of generated Hypothesis objects
        """
        # 1. Search for relevant papers
        search_results = self.vectordb.search(query, k=k)

        if not search_results:
            logger.warning(f"No papers found for query: {query}")
            return []

        # Extract unique paper IDs
        paper_ids = list(
            set(
                result["metadata"].get("paper_id", result["id"].split("_")[0])
                for result in search_results
            )
        )

        logger.info(f"Found {len(paper_ids)} unique papers for query: {query}")

        # 2-5. Generate hypotheses from papers
        return await self.generate_from_papers(paper_ids)
