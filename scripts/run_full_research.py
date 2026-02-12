#!/usr/bin/env python3
"""
Run full Fleming-AI research pipeline end-to-end.

Pipeline stages:
1. Environment validation
2. Hypothesis generation (Swanson ABC + VectorDB)
3. Hypothesis selection + Alex review
4. Hypothesis -> experiment config translation
5. Smoke test (1 epoch)
6. Full training (3-5 epochs)
7. Results analysis + figure generation
8. Paper generation
9. Alex paper review + Fleming revisions (max 3 turns)
10. PDF compilation

Artifacts are saved under runs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Add project root to import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment.src.datasets import get_num_classes, get_transforms, load_dataset
from experiment.src.models import extract_features, load_model, load_model_unfrozen
from experiment.src.train import knn_evaluate, train_from_scratch, train_linear_probe
from experiment.src.utils import set_seed
from src.filters.quality import QualityFilter
from src.generators.hypothesis import Hypothesis, HypothesisGenerator
from src.llm.claude_client import ClaudeClient
from src.llm.groq_client import GroqClient
from src.pipeline.capabilities import CAPABILITIES
from src.pipeline.experiment_paper_generator import ExperimentPaperGenerator
from src.pipeline.experiment_translator import ExperimentTranslator
from src.pipeline.feasibility_checker import check_hypothesis_feasible
from src.pipeline.review_gates import revise_hypothesis, revise_paper
from src.reviewers.alex import Alex
from src.reviewers.conversation import ConversationManager
from src.storage.vectordb import VectorDB


MAX_REVIEW_TURNS = 3
MAX_HYPOTHESES = 5
MIN_HYPOTHESES = 3
MIN_VECTORDB_CHUNKS = 1000
PIPELINE_TIMEOUT_SECONDS = 6 * 60 * 60
SMOKE_TEST_EPOCHS = 1
FULL_TRAIN_MIN_EPOCHS = 3
FULL_TRAIN_MAX_EPOCHS = 5
PDF_COMPILE_TIMEOUT_SECONDS = 300
PDF_MAX_DEBUG_SECONDS = 30 * 60

VISION_KEYWORDS = {
    "vision",
    "image",
    "visual",
    "transformer",
    "cnn",
    "resnet",
    "deit",
    "classification",
    "cifar",
    "flowers",
    "pet",
    "stl",
    "object",
}

MODEL_DISPLAY_TO_KEY = {
    "deit-small": "deit_small",
    "deitsmall": "deit_small",
    "deit_small": "deit_small",
    "deit": "deit_small",
    "visiontransformer": "deit_small",
    "vit": "deit_small",
    "resnet-34": "resnet34",
    "resnet34": "resnet34",
    "resnet": "resnet34",
}

MODEL_KEY_TO_DISPLAY = {
    "deit_small": "DeiT-Small",
    "resnet34": "ResNet-34",
}

DATASET_DISPLAY_TO_KEY = {
    "cifar-10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar100": "cifar100",
    "stl-10": "stl10",
    "stl10": "stl10",
    "flowers102": "flowers102",
    "flowers-102": "flowers102",
    "flowers": "flowers102",
    "oxford-pets": "oxford_pets",
    "oxfordpets": "oxford_pets",
    "oxford_pets": "oxford_pets",
    "pets": "oxford_pets",
}

DATASET_KEY_TO_DISPLAY = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "stl10": "STL-10",
    "flowers102": "Flowers102",
    "oxford_pets": "Oxford-Pets",
}

SUPPORTED_TRAINING_MODES = {"linear_probe", "knn_evaluate", "train_from_scratch"}


@dataclass
class StageRecord:
    stage: str
    status: str
    started_at: str
    ended_at: str
    duration_seconds: float
    message: str = ""


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclass_is_instance(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def configure_logging(log_path: Path) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    while root_logger.handlers:
        root_logger.handlers.pop()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logger = logging.getLogger("run_full_research")
    logger.info("Logging initialized: %s", log_path)
    return logger


class PipelineExperimentTranslator(ExperimentTranslator):
    """
    Adapter that extracts HypothesisSpec from text using LLM, then uses
    ExperimentTranslator's deterministic mapping.
    """

    def __init__(self, llm_client: GroqClient):
        super().__init__()
        self._llm = llm_client

    async def translate(self, hypothesis_text: str) -> dict[str, Any]:
        prompt = self._build_spec_extraction_prompt(hypothesis_text)

        try:
            raw_response = await self._llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=700,
            )
            spec = self._extract_json_config(raw_response)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "LLM spec extraction failed: %s; using heuristic fallback.",
                exc,
            )
            spec = self._extract_spec_heuristic(hypothesis_text)

        return await super().translate(spec)

    def _build_spec_extraction_prompt(self, hypothesis_text: str) -> str:
        return f"""Extract a structured hypothesis specification from the following text.

Hypothesis: {hypothesis_text}

Available models: {", ".join(MODEL_KEY_TO_DISPLAY.values())}
Available datasets: {", ".join(DATASET_KEY_TO_DISPLAY.values())}

Extract and output ONLY a JSON object with these fields:
- hypothesis: the hypothesis text (verbatim)
- confidence: estimated confidence (0.0-1.0)
- task: "image_classification"
- dataset: one of the available datasets
- baseline: {{"model": model_name, "pretrain": "imagenet"}}
- variant: {{"model": model_name, "pretrain": "imagenet"}}
- metric: "top1_accuracy" or "accuracy"
- expected_effect: {{"direction": "increase" or "decrease", "min_delta_points": 2}}

Output ONLY valid JSON, no explanation."""

    def _extract_spec_heuristic(self, hypothesis_text: str) -> dict[str, Any]:
        hypothesis_lower = hypothesis_text.lower()

        baseline_model = "resnet34"
        variant_model = "deit_small"

        if "resnet" in hypothesis_lower and "deit" in hypothesis_lower:
            if hypothesis_lower.index("resnet") < hypothesis_lower.index("deit"):
                baseline_model = "resnet34"
                variant_model = "deit_small"
            else:
                baseline_model = "deit_small"
                variant_model = "resnet34"
        elif "resnet" in hypothesis_lower:
            baseline_model = "resnet34"
            variant_model = "deit_small"
        elif "deit" in hypothesis_lower or "vit" in hypothesis_lower:
            baseline_model = "resnet34"
            variant_model = "deit_small"

        dataset = "cifar10"
        for ds_name in DATASET_KEY_TO_DISPLAY.keys():
            if ds_name.replace("_", "").replace("-", "") in hypothesis_lower.replace(
                "_", ""
            ).replace("-", ""):
                dataset = ds_name
                break
        else:
            if any(kw in hypothesis_lower for kw in ["transfer", "fine-grained", "flower", "pet"]):
                dataset = "flowers102"

        return {
            "hypothesis": hypothesis_text,
            "confidence": 0.7,
            "task": "image_classification",
            "dataset": dataset,
            "baseline": {"model": baseline_model, "pretrain": "imagenet"},
            "variant": {"model": variant_model, "pretrain": "imagenet"},
            "metric": "top1_accuracy",
            "expected_effect": {"direction": "increase", "min_delta_points": 2},
        }

    @staticmethod
    def _extract_json_config(raw_response: str) -> dict[str, Any]:
        text = raw_response.strip()

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        if not text.startswith("{"):
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                text = text[first_brace : last_brace + 1]

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Translator response is not a JSON object")
        return parsed


class FullResearchPipeline:
    def __init__(self, run_dir: Path, dry_run: bool):
        self.run_dir = run_dir
        self.figures_dir = run_dir / "figures"
        self.pipeline_log_path = run_dir / "pipeline.log"
        self.review_log_path = run_dir / "review_log.json"
        self.stage_log_path = run_dir / "stage_log.json"
        self.partial_path = run_dir / "partial_results.json"

        self.dry_run = dry_run
        self.logger = logging.getLogger("run_full_research")

        self.groq_llm: GroqClient | None = None
        self.claude_llm: ClaudeClient | None = None
        self.vectordb: VectorDB | None = None
        self.alex: Alex | None = None

        self.stage_records: list[StageRecord] = []
        self.review_log: dict[str, list[dict[str, Any]]] = {
            "hypothesis_review": [],
            "paper_review": [],
        }
        self.artifacts: dict[str, str] = {}

    def _register_artifact(self, name: str, path: Path) -> None:
        self.artifacts[name] = str(path)

    async def _run_stage(self, stage_name: str, stage_fn: Any, critical: bool) -> Any:
        started_at = now_utc_iso()
        start_time = time.time()

        self.logger.info("=" * 80)
        self.logger.info("Starting stage: %s", stage_name)
        self.logger.info("=" * 80)

        try:
            result = stage_fn()
            if inspect.isawaitable(result):
                result = await result

            duration = time.time() - start_time
            record = StageRecord(
                stage=stage_name,
                status="success",
                started_at=started_at,
                ended_at=now_utc_iso(),
                duration_seconds=round(duration, 2),
            )
            self.stage_records.append(record)
            write_json(self.stage_log_path, [asdict(item) for item in self.stage_records])

            self.logger.info("Stage completed: %s (%.2fs)", stage_name, duration)
            return result

        except Exception as exc:
            duration = time.time() - start_time
            message = f"{exc}\n{traceback.format_exc()}"
            record = StageRecord(
                stage=stage_name,
                status="failed",
                started_at=started_at,
                ended_at=now_utc_iso(),
                duration_seconds=round(duration, 2),
                message=str(exc),
            )
            self.stage_records.append(record)
            write_json(self.stage_log_path, [asdict(item) for item in self.stage_records])

            self.logger.error("Stage failed: %s", stage_name)
            self.logger.error("%s", message)
            self._save_partial_results(reason=f"{stage_name} failed: {exc}")

            if critical:
                raise
            return None

    def _save_partial_results(self, reason: str) -> None:
        payload = {
            "run_dir": str(self.run_dir),
            "reason": reason,
            "timestamp": now_utc_iso(),
            "stages": [asdict(item) for item in self.stage_records],
            "artifacts": self.artifacts,
            "review_log": self.review_log,
        }
        write_json(self.partial_path, payload)

    async def validate_environment(self) -> dict[str, Any]:
        if self.groq_llm is None or self.claude_llm is None or self.vectordb is None:
            raise RuntimeError("Pipeline dependencies are not initialized")

        status: dict[str, Any] = {}

        mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        status["mps_available"] = mps_available
        status["mps_built"] = torch.backends.mps.is_built()
        if not mps_available:
            raise RuntimeError("MPS is required but not available")

        status["llm_backends"] = {"groq": True, "claude": True}
        self.logger.info("LLM backends: Alex=Claude, Fleming=Groq")

        chunk_count = self.vectordb.count()
        status["vectordb_chunks"] = chunk_count
        if chunk_count < MIN_VECTORDB_CHUNKS:
            raise RuntimeError(
                f"VectorDB not ready: expected >= {MIN_VECTORDB_CHUNKS} chunks, got {chunk_count}"
            )

        pdflatex_check = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        status["pdflatex_available"] = pdflatex_check.returncode == 0
        status["pdflatex_version"] = pdflatex_check.stdout.splitlines()[0].strip()

        env_path = self.run_dir / "environment_validation.json"
        write_json(env_path, status)
        self._register_artifact("environment_validation", env_path)

        self.logger.info("Environment validation passed")
        self.logger.info("MPS available: %s", status["mps_available"])
        self.logger.info("VectorDB chunks: %d", chunk_count)
        self.logger.info("LLM backends: %s", backend_status)

        return status

    async def generate_hypotheses(self) -> list[Hypothesis]:
        if self.llm is None or self.vectordb is None:
            raise RuntimeError("Pipeline dependencies are not initialized")

        quality_filter = QualityFilter()
        generator = HypothesisGenerator(
            llm_client=self.groq_llm,
            vectordb=self.vectordb,
            quality_filter=quality_filter,
        )

        queries = [
            "vision transformer transfer learning cnn comparison",
            "image classification representation learning pretraining",
            "self-attention vs convolution inductive bias vision datasets",
        ]

        candidates: list[Hypothesis] = []
        seen_texts: set[str] = set()

        for query in queries:
            if len(candidates) >= MAX_HYPOTHESES:
                break

            self.logger.info("Generating hypotheses from query: %s", query)
            papers = self._collect_papers_for_query(query, k=10)
            if len(papers) < 2:
                self.logger.warning("Insufficient papers for query '%s'", query)
                continue

            concept_graph, paper_concepts = await generator.find_concept_connections(papers)
            abc_patterns = await generator.find_abc_patterns(concept_graph, paper_concepts)

            paper_text_lookup = {item["paper_id"]: item["text"] for item in papers}

            for pattern in abc_patterns:
                if len(candidates) >= MAX_HYPOTHESES:
                    break

                paper_a_text = paper_text_lookup.get(pattern.paper_a_id, "")
                paper_b_text = paper_text_lookup.get(pattern.paper_b_id, "")
                if not paper_a_text or not paper_b_text:
                    continue

                spec_dict = await generator.generate_hypothesis_text(
                    pattern,
                    paper_a_text,
                    paper_b_text,
                )
                hypothesis_text = spec_dict["hypothesis"]
                confidence = spec_dict["confidence"]

                normalized = hypothesis_text.strip().lower()
                if not normalized or normalized in seen_texts:
                    continue
                if not self._looks_like_vision_hypothesis(hypothesis_text):
                    continue

                seen_texts.add(normalized)
                quality_score = quality_filter.score(hypothesis_text)
                candidates.append(
                    Hypothesis(
                        id=str(uuid.uuid4()),
                        hypothesis_text=hypothesis_text.strip(),
                        source_papers=[pattern.paper_a_id, pattern.paper_b_id],
                        connection={
                            "concept_a": pattern.concept_a,
                            "concept_b": pattern.concept_b,
                            "bridging_concept": pattern.bridging_concept,
                        },
                        confidence=confidence,
                        quality_score=quality_score,
                    )
                )

            self.logger.info("Current hypotheses collected: %d", len(candidates))

        if len(candidates) < MIN_HYPOTHESES:
            raise RuntimeError(
                f"Generated only {len(candidates)} hypotheses; expected at least {MIN_HYPOTHESES}"
            )

        ranked = sorted(candidates, key=self._rank_hypothesis, reverse=True)[:MAX_HYPOTHESES]

        candidates_payload = [
            {
                **item.to_dict(),
                "ranking_score": self._rank_hypothesis(item),
            }
            for item in ranked
        ]
        candidates_path = self.run_dir / "hypothesis_candidates.json"
        write_json(candidates_path, candidates_payload)
        self._register_artifact("hypothesis_candidates", candidates_path)

        self.logger.info("Generated %d ranked hypotheses", len(ranked))
        return ranked

    async def check_hypothesis_feasibility(self, hypothesis_text: str) -> dict[str, Any]:
        """
        Check if a hypothesis is feasible given available capabilities.

        This stage validates that the hypothesis can be executed with available
        models, datasets, and metrics. Infeasible hypotheses are rejected early
        before expensive review and experiment stages.

        Args:
            hypothesis_text: The hypothesis text to validate

        Returns:
            Dictionary with feasibility check results

        Raises:
            ValueError: If hypothesis fails feasibility check
        """
        # Extract hypothesis spec from text using the translator's heuristic
        translator = PipelineExperimentTranslator(llm_client=self.groq_llm)
        spec = translator._extract_spec_heuristic(hypothesis_text)

        # Check feasibility against capabilities
        is_feasible, errors = check_hypothesis_feasible(spec, CAPABILITIES)

        if not is_feasible:
            self.logger.error("Hypothesis failed feasibility check:")
            for error in errors:
                self.logger.error("  - %s", error)
            raise ValueError(f"Infeasible hypothesis: {'; '.join(errors)}")

        self.logger.info("Hypothesis passed feasibility check")
        self.logger.info("  Task: %s", spec.get("task"))
        self.logger.info("  Dataset: %s", spec.get("dataset"))
        self.logger.info("  Baseline model: %s", spec.get("baseline", {}).get("model"))
        self.logger.info("  Variant model: %s", spec.get("variant", {}).get("model"))
        self.logger.info("  Metric: %s", spec.get("metric"))

        return {
            "hypothesis_text": hypothesis_text,
            "spec": spec,
            "is_feasible": True,
            "checked_at": now_utc_iso(),
        }

    def _collect_papers_for_query(self, query: str, k: int) -> list[dict[str, str]]:
        if self.vectordb is None:
            raise RuntimeError("VectorDB is not initialized")

        search_results = self.vectordb.search(query, k=k)
        unique_ids: list[str] = []
        seen_ids: set[str] = set()

        for result in search_results:
            metadata = result.get("metadata", {})
            paper_id = metadata.get("paper_id")
            if not paper_id or paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            unique_ids.append(paper_id)

        papers: list[dict[str, str]] = []
        for paper_id in unique_ids:
            paper_data = self.vectordb.get_paper(paper_id)
            chunks = paper_data.get("chunks", [])
            if not chunks:
                continue
            text = "\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text"))
            if len(text) < 100:
                continue
            papers.append(
                {
                    "paper_id": paper_id,
                    "text": text,
                }
            )

        return papers

    def _looks_like_vision_hypothesis(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in VISION_KEYWORDS)

    def _rank_hypothesis(self, hypothesis: Hypothesis) -> float:
        text = hypothesis.hypothesis_text.strip()
        text_lower = text.lower()

        length = len(text)
        if 120 <= length <= 320:
            length_score = 1.0
        elif 80 <= length < 120 or 320 < length <= 420:
            length_score = 0.7
        else:
            length_score = 0.4

        specificity_tokens = [
            "when",
            "under",
            "compared",
            "dataset",
            "accuracy",
            "cifar",
            "flowers",
            "resnet",
            "deit",
            "transformer",
            "cnn",
        ]
        specificity_hits = sum(token in text_lower for token in specificity_tokens)
        specificity_score = min(1.0, specificity_hits / 6.0)

        vision_hits = sum(keyword in text_lower for keyword in VISION_KEYWORDS)
        vision_score = min(1.0, vision_hits / 5.0)

        rank_score = (
            0.35 * float(hypothesis.quality_score)
            + 0.25 * float(hypothesis.confidence)
            + 0.20 * length_score
            + 0.20 * (0.6 * specificity_score + 0.4 * vision_score)
        )
        return round(rank_score, 4)

    async def review_hypothesis(self, hypotheses: list[Hypothesis]) -> dict[str, Any]:
        if self.llm is None or self.alex is None:
            raise RuntimeError("Pipeline dependencies are not initialized")

        ranked = sorted(hypotheses, key=self._rank_hypothesis, reverse=True)
        selected = ranked[0]

        conversation = ConversationManager(max_turns=MAX_REVIEW_TURNS)
        current_text = selected.hypothesis_text
        fallback_pool = [item.hypothesis_text for item in ranked[1:]]
        restart_used = False

        while not conversation.is_converged():
            review = await self.alex.review_hypothesis(
                hypothesis_text=current_text,
                conversation_history=conversation.state.turns,
            )
            conversation.add_turn("alex", review.verdict, review.__dict__)

            self.review_log["hypothesis_review"].append(
                {
                    "timestamp": now_utc_iso(),
                    "verdict": review.verdict,
                    "strengths": review.strengths,
                    "weaknesses": review.weaknesses,
                    "suggestions": review.suggestions,
                    "scores": review.scores,
                    "artifact_preview": current_text[:400],
                }
            )

            if review.verdict == "PASS":
                break

            if review.verdict == "RESTART_PIPELINE":
                raise RuntimeError("Alex requested RESTART_PIPELINE during hypothesis review")

            if review.verdict == "RESTART_STAGE":
                if not restart_used and fallback_pool:
                    restart_used = True
                    current_text = fallback_pool.pop(0)
                    conversation.add_turn(
                        "fleming",
                        "Switched to next-ranked hypothesis candidate.",
                        {"artifact": current_text},
                    )
                    continue

                self.logger.warning(
                    "Repeated RESTART_STAGE or no fallback candidate; proceeding with current hypothesis"
                )
                break

            revised = await revise_hypothesis(self.groq_llm, current_text, review)
            current_text = revised.strip()
            conversation.add_turn(
                "fleming",
                "Revised hypothesis.",
                {"artifact": current_text},
            )

        output = {
            "hypothesis_text": current_text,
            "selected_hypothesis_id": selected.id,
            "selected_connection": selected.connection,
            "source_papers": selected.source_papers,
            "candidate_count": len(ranked),
            "review_turns": len(conversation.state.turns),
            "ranked_candidates": [
                {
                    "id": item.id,
                    "hypothesis_text": item.hypothesis_text,
                    "ranking_score": self._rank_hypothesis(item),
                    "confidence": item.confidence,
                    "quality_score": item.quality_score,
                }
                for item in ranked
            ],
        }

        hypothesis_path = self.run_dir / "hypothesis.json"
        write_json(hypothesis_path, output)
        self._register_artifact("hypothesis", hypothesis_path)

        write_json(self.review_log_path, self.review_log)
        self._register_artifact("review_log", self.review_log_path)

        return output

    async def translate_experiment(self, hypothesis_text: str) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("Pipeline dependencies are not initialized")

        translator = PipelineExperimentTranslator(llm_client=self.groq_llm)
        translated = await translator.translate(hypothesis_text)
        normalized = self._normalize_experiment_config(translated)

        config_payload = {
            "translated": translated,
            "normalized": normalized,
            "generated_at": now_utc_iso(),
        }

        config_path = self.run_dir / "experiment_config.json"
        write_json(config_path, config_payload)
        self._register_artifact("experiment_config", config_path)

        return normalized

    def _normalize_experiment_config(self, config: dict[str, Any]) -> dict[str, Any]:
        model_keys: list[str] = []
        dataset_keys: list[str] = []

        for model in config.get("models", []):
            model_key = self._normalize_model_key(model)
            if model_key and model_key not in model_keys:
                model_keys.append(model_key)

        for dataset in config.get("datasets", []):
            dataset_key = self._normalize_dataset_key(dataset)
            if dataset_key and dataset_key not in dataset_keys:
                dataset_keys.append(dataset_key)

        if not model_keys:
            model_keys = ["deit_small"]
        if not dataset_keys:
            dataset_keys = ["cifar10"]

        model_keys = model_keys[:2]
        dataset_keys = dataset_keys[:2]

        training_mode = str(config.get("training_mode", "linear_probe")).strip().lower()
        if training_mode not in SUPPORTED_TRAINING_MODES:
            training_mode = "linear_probe"

        epochs = int(config.get("epochs", FULL_TRAIN_MIN_EPOCHS))
        epochs = max(FULL_TRAIN_MIN_EPOCHS, min(epochs, FULL_TRAIN_MAX_EPOCHS))

        batch_size = int(config.get("batch_size", 32))
        batch_size = max(8, min(batch_size, 256))

        learning_rate = float(config.get("learning_rate", 0.001))
        learning_rate = max(1e-5, min(learning_rate, 0.1))

        metrics_to_track = config.get("metrics_to_track", ["accuracy", "loss"])
        if not isinstance(metrics_to_track, list) or not metrics_to_track:
            metrics_to_track = ["accuracy", "loss"]

        return {
            "models": [MODEL_KEY_TO_DISPLAY[key] for key in model_keys],
            "model_keys": model_keys,
            "datasets": [DATASET_KEY_TO_DISPLAY[key] for key in dataset_keys],
            "dataset_keys": dataset_keys,
            "training_mode": training_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "metrics_to_track": metrics_to_track,
            "seed": 42,
            "device": "mps",
        }

    def _normalize_model_key(self, model: Any) -> str | None:
        if not isinstance(model, str):
            return None
        cleaned = model.lower().replace(" ", "").replace("_", "-")
        cleaned = cleaned.replace("deitsmall", "deit-small")
        if cleaned in MODEL_DISPLAY_TO_KEY:
            return MODEL_DISPLAY_TO_KEY[cleaned]
        if "deit" in cleaned or "vit" in cleaned or "transformer" in cleaned:
            return "deit_small"
        if "resnet" in cleaned or "cnn" in cleaned:
            return "resnet34"
        return None

    def _normalize_dataset_key(self, dataset: Any) -> str | None:
        if not isinstance(dataset, str):
            return None
        cleaned = dataset.lower().replace(" ", "").replace("_", "-")
        if cleaned in DATASET_DISPLAY_TO_KEY:
            return DATASET_DISPLAY_TO_KEY[cleaned]
        if "cifar10" in cleaned:
            return "cifar10"
        if "cifar100" in cleaned:
            return "cifar100"
        if "stl" in cleaned:
            return "stl10"
        if "flower" in cleaned:
            return "flowers102"
        if "pet" in cleaned or "oxford" in cleaned:
            return "oxford_pets"
        return None

    async def run_smoke_test(self, config: dict[str, Any]) -> dict[str, Any]:
        smoke_results = await self._run_experiment_suite(
            config=config,
            epochs=SMOKE_TEST_EPOCHS,
            smoke=True,
        )

        failures = [item for item in smoke_results["runs"] if item["status"] != "success"]
        if failures:
            raise RuntimeError(f"Smoke test failed with {len(failures)} failing run(s)")

        smoke_path = self.run_dir / "smoke_test_results.json"
        write_json(smoke_path, smoke_results)
        self._register_artifact("smoke_test_results", smoke_path)

        return smoke_results

    async def run_full_training(self, config: dict[str, Any]) -> dict[str, Any]:
        epochs = int(config["epochs"])
        epochs = max(FULL_TRAIN_MIN_EPOCHS, min(epochs, FULL_TRAIN_MAX_EPOCHS))

        full_results = await self._run_experiment_suite(
            config=config,
            epochs=epochs,
            smoke=False,
        )

        results_path = self.run_dir / "experiment_results.json"
        write_json(results_path, full_results)
        self._register_artifact("experiment_results", results_path)

        return full_results

    async def _run_experiment_suite(
        self,
        config: dict[str, Any],
        epochs: int,
        smoke: bool,
    ) -> dict[str, Any]:
        combinations: list[tuple[str, str]] = []
        for model_key in config["model_keys"]:
            for dataset_key in config["dataset_keys"]:
                combinations.append((model_key, dataset_key))

        if smoke:
            combinations = combinations[:1]

        runs: list[dict[str, Any]] = []
        for index, (model_key, dataset_key) in enumerate(combinations, start=1):
            self.logger.info(
                "Running experiment %d/%d: model=%s dataset=%s mode=%s epochs=%d",
                index,
                len(combinations),
                model_key,
                dataset_key,
                config["training_mode"],
                epochs,
            )

            try:
                run_result = self._run_single_experiment(
                    model_key=model_key,
                    dataset_key=dataset_key,
                    training_mode=config["training_mode"],
                    epochs=epochs,
                    batch_size=int(config["batch_size"]),
                    learning_rate=float(config["learning_rate"]),
                    seed=int(config.get("seed", 42)),
                )
                run_result["status"] = "success"
                runs.append(run_result)
            except Exception as exc:
                failed = {
                    "key": f"{model_key}_{dataset_key}",
                    "model": model_key,
                    "dataset": dataset_key,
                    "training_mode": config["training_mode"],
                    "epochs": epochs,
                    "status": "failed",
                    "error": str(exc),
                }
                runs.append(failed)
                self.logger.error("Experiment failed: %s", failed)
                if smoke:
                    raise

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        successful_runs = [item for item in runs if item["status"] == "success"]
        failed_runs = [item for item in runs if item["status"] != "success"]

        summary = {
            "mode": "smoke" if smoke else "full",
            "training_mode": config["training_mode"],
            "epochs": epochs,
            "total_runs": len(runs),
            "success_count": len(successful_runs),
            "failure_count": len(failed_runs),
            "mean_accuracy": self._safe_mean(
                [float(item.get("accuracy", 0.0)) for item in successful_runs]
            ),
            "runs": runs,
        }

        return summary

    def _run_single_experiment(
        self,
        model_key: str,
        dataset_key: str,
        training_mode: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> dict[str, Any]:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required for full research pipeline")

        device = torch.device("mps")
        set_seed(seed)

        num_classes = get_num_classes(dataset_key)
        start_time = time.time()

        if training_mode == "train_from_scratch":
            train_transform = get_transforms(pretrained=False, train=True)
            test_transform = get_transforms(pretrained=False, train=False)

            train_dataset = load_dataset(dataset_key, split="train", transform=train_transform)
            test_dataset = load_dataset(dataset_key, split="test", transform=test_transform)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            model = load_model_unfrozen(arch=model_key, pretrained=False, num_classes=num_classes)
            train_result = train_from_scratch(
                model,
                train_loader,
                test_loader,
                {
                    "device": str(device),
                    "arch": model_key,
                    "epochs": epochs,
                    "lr": learning_rate,
                    "early_stopping_patience": max(1, min(epochs, 5)),
                    "timeout_hours": 1 if epochs <= 1 else 4,
                },
            )

            accuracy = float(train_result.get("best_accuracy", 0.0))
            loss_curve = [float(item) for item in train_result.get("train_curve", [])]
            val_curve = [float(item) for item in train_result.get("val_curve", [])]
            metadata = {
                "converged": bool(train_result.get("converged", False)),
            }

        elif training_mode in {"linear_probe", "knn_evaluate"}:
            train_transform = get_transforms(pretrained=True, train=False)
            test_transform = get_transforms(pretrained=True, train=False)

            train_dataset = load_dataset(dataset_key, split="train", transform=train_transform)
            test_dataset = load_dataset(dataset_key, split="test", transform=test_transform)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            model = load_model(arch=model_key, pretrained=True, num_classes=num_classes)

            train_features, train_labels = extract_features(model, train_loader, device)
            test_features, test_labels = extract_features(model, test_loader, device)

            if training_mode == "knn_evaluate":
                accuracy = float(
                    knn_evaluate(
                        train_features=train_features,
                        train_labels=train_labels,
                        test_features=test_features,
                        test_labels=test_labels,
                        k=20,
                    )
                )
                loss_curve = []
                val_curve = []
                metadata = {"k": 20}
            else:
                lr_grid = sorted({learning_rate, learning_rate * 0.1, learning_rate * 10.0})
                linear_result = train_linear_probe(
                    features=train_features,
                    labels=train_labels,
                    num_classes=num_classes,
                    config={
                        "device": str(device),
                        "epochs": epochs,
                        "lr_grid": lr_grid,
                    },
                )

                probe_result = self._train_probe_on_train_and_evaluate(
                    train_features=train_features,
                    train_labels=train_labels,
                    test_features=test_features,
                    test_labels=test_labels,
                    num_classes=num_classes,
                    learning_rate=float(linear_result["best_lr"]),
                    device=device,
                    epochs=epochs,
                )

                accuracy = float(probe_result["accuracy"])
                loss_curve = [float(item) for item in linear_result.get("train_loss_curve", [])]
                val_curve = []
                metadata = {
                    "best_lr": float(linear_result["best_lr"]),
                    "probe_runtime_seconds": float(probe_result["runtime_seconds"]),
                }

        else:
            raise ValueError(f"Unsupported training mode: {training_mode}")

        runtime_seconds = time.time() - start_time

        return {
            "key": f"{model_key}_{dataset_key}",
            "model": model_key,
            "dataset": dataset_key,
            "training_mode": training_mode,
            "epochs": epochs,
            "accuracy": accuracy,
            "loss_curve": loss_curve,
            "val_curve": val_curve,
            "runtime_seconds": round(runtime_seconds, 2),
            "metadata": metadata,
        }

    def _train_probe_on_train_and_evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        num_classes: int,
        learning_rate: float,
        device: torch.device,
        epochs: int,
    ) -> dict[str, float]:
        import torch.nn as nn

        start_time = time.time()
        feature_dim = int(train_features.shape[1])
        classifier = nn.Linear(feature_dim, num_classes).to(device)
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        criterion = nn.CrossEntropyLoss()

        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        for _ in range(max(1, epochs)):
            classifier.train()
            for feats, labels in train_loader:
                feats = feats.to(device)
                labels = labels.to(device)

                logits = classifier(feats)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        classifier.eval()
        with torch.no_grad():
            test_feats = test_features.to(device)
            test_lbls = test_labels.to(device)
            predictions = classifier(test_feats).argmax(dim=1)
            accuracy = float((predictions == test_lbls).float().mean().item())

        return {
            "accuracy": accuracy,
            "runtime_seconds": round(time.time() - start_time, 2),
        }

    async def analyze_results(self, full_results: dict[str, Any]) -> dict[str, Any]:
        runs = [item for item in full_results.get("runs", []) if item.get("status") == "success"]
        if not runs:
            raise RuntimeError("No successful runs to analyze")

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        analysis = {
            "run_count": len(runs),
            "best_run": max(runs, key=lambda item: float(item.get("accuracy", 0.0))),
            "worst_run": min(runs, key=lambda item: float(item.get("accuracy", 0.0))),
            "mean_accuracy": self._safe_mean([float(item.get("accuracy", 0.0)) for item in runs]),
            "std_accuracy": self._safe_std([float(item.get("accuracy", 0.0)) for item in runs]),
        }

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            accuracy_labels = [item["key"] for item in runs]
            accuracies = [float(item.get("accuracy", 0.0)) for item in runs]

            fig1, ax1 = plt.subplots(figsize=(max(6, len(runs) * 1.8), 4.5))
            ax1.bar(accuracy_labels, accuracies, color="#2f6db3")
            ax1.set_title("Accuracy by Model-Dataset Configuration")
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel("Run")
            ax1.set_ylim(0.0, min(1.0, max(0.05, max(accuracies) + 0.05)))
            ax1.tick_params(axis="x", rotation=35)
            fig1.tight_layout()

            accuracy_path = self.figures_dir / "accuracy_bars.pdf"
            fig1.savefig(accuracy_path)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(7.0, 4.5))
            any_curve = False
            for run in runs:
                curve = self._extract_curve(run)
                if not curve:
                    continue
                any_curve = True
                ax2.plot(curve, label=run["key"])

            if any_curve:
                ax2.set_title("Loss / Training Curves")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Loss or Score")
                ax2.legend(fontsize=8)
            else:
                ax2.text(0.5, 0.5, "No training curves available", ha="center", va="center")
                ax2.set_axis_off()

            fig2.tight_layout()
            loss_path = self.figures_dir / "loss_curve.pdf"
            fig2.savefig(loss_path)
            plt.close(fig2)

            analysis["figure_paths"] = {
                "accuracy_bars": str(accuracy_path),
                "loss_curve": str(loss_path),
            }

            self._register_artifact("figures_dir", self.figures_dir)

        except Exception as exc:
            self.logger.warning("Figure generation skipped due to error: %s", exc)
            analysis["figure_generation_error"] = str(exc)

        updated_results = {
            **full_results,
            "analysis": analysis,
        }
        results_path = self.run_dir / "experiment_results.json"
        write_json(results_path, updated_results)
        self._register_artifact("experiment_results", results_path)

        return analysis

    def _extract_curve(self, run: dict[str, Any]) -> list[float]:
        candidates = [
            run.get("loss_curve", []),
            run.get("val_curve", []),
            run.get("metadata", {}).get("train_loss_curve", []),
        ]
        for candidate in candidates:
            if isinstance(candidate, list) and candidate:
                return [float(item) for item in candidate]
        return []

    async def generate_paper(
        self,
        hypothesis_text: str,
        config: dict[str, Any],
        full_results: dict[str, Any],
    ) -> str:
        if self.llm is None or self.vectordb is None:
            raise RuntimeError("Pipeline dependencies are not initialized")

        paper_input_results: dict[str, Any] = {}
        for run in full_results.get("runs", []):
            if run.get("status") != "success":
                continue
            paper_input_results[run["key"]] = {
                "accuracy": run.get("accuracy"),
                "runtime_seconds": run.get("runtime_seconds"),
                "loss": run.get("loss_curve", [])[-1] if run.get("loss_curve") else None,
            }

        generator = ExperimentPaperGenerator(groq_client=self.groq_llm, vector_db=self.vectordb)
        latex = await generator.generate_paper(
            hypothesis=hypothesis_text,
            config=config,
            results=paper_input_results,
            template_path="runs/templates/",
        )

        paper_path = self.run_dir / "paper.tex"
        paper_path.write_text(latex)
        self._register_artifact("paper_tex", paper_path)

        return latex

    async def review_generated_paper(
        self,
        hypothesis_text: str,
        config: dict[str, Any],
        full_results: dict[str, Any],
        paper_text: str,
    ) -> str:
        if self.llm is None or self.alex is None:
            raise RuntimeError("Pipeline dependencies are not initialized")

        conversation = ConversationManager(max_turns=MAX_REVIEW_TURNS)
        current_paper = paper_text
        restart_used = False

        serialized_config = json.dumps(config, indent=2)
        serialized_results = json.dumps(full_results, indent=2)

        while not conversation.is_converged():
            review = await self.alex.review_paper(
                paper_text=current_paper,
                hypothesis=hypothesis_text,
                experiment_design=serialized_config,
                results=serialized_results,
                conversation_history=conversation.state.turns,
            )
            conversation.add_turn("alex", review.verdict, review.__dict__)

            self.review_log["paper_review"].append(
                {
                    "timestamp": now_utc_iso(),
                    "verdict": review.verdict,
                    "strengths": review.strengths,
                    "weaknesses": review.weaknesses,
                    "suggestions": review.suggestions,
                    "scores": review.scores,
                    "artifact_preview": current_paper[:400],
                }
            )

            if review.verdict == "PASS":
                break

            if review.verdict == "RESTART_PIPELINE":
                self.logger.warning(
                    "Alex requested RESTART_PIPELINE for paper stage; keeping current draft"
                )
                break

            if review.verdict == "RESTART_STAGE":
                if restart_used:
                    self.logger.warning(
                        "Repeated RESTART_STAGE in paper review; stopping revisions"
                    )
                    break
                restart_used = True

            revised = await revise_paper(self.groq_llm, current_paper, review)
            current_paper = revised
            conversation.add_turn(
                "fleming",
                "Revised paper draft.",
                {"artifact": current_paper[:2000]},
            )

        paper_path = self.run_dir / "paper.tex"
        paper_path.write_text(current_paper)
        self._register_artifact("paper_tex", paper_path)

        write_json(self.review_log_path, self.review_log)
        self._register_artifact("review_log", self.review_log_path)

        return current_paper

    async def compile_pdf(self) -> Path:
        paper_path = self.run_dir / "paper.tex"
        if not paper_path.exists():
            raise RuntimeError("paper.tex not found; cannot compile PDF")

        template_style = Path("runs/templates/neurips_2024.sty")
        if template_style.exists():
            shutil.copy(template_style, self.run_dir / "neurips_2024.sty")

        compile_start = time.time()
        passes = 2

        for pass_index in range(1, passes + 1):
            elapsed = time.time() - compile_start
            if elapsed > PDF_MAX_DEBUG_SECONDS:
                raise RuntimeError("Exceeded LaTeX debug budget (30 minutes)")

            self.logger.info("pdflatex pass %d/%d", pass_index, passes)
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "paper.tex",
                ],
                cwd=self.run_dir,
                check=True,
                timeout=PDF_COMPILE_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )

        pdf_path = self.run_dir / "paper.pdf"
        if not pdf_path.exists():
            raise RuntimeError("pdflatex completed without creating paper.pdf")

        self._register_artifact("paper_pdf", pdf_path)
        return pdf_path

    async def run(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Run directory: %s", self.run_dir)
        self.logger.info("Dry-run mode: %s", self.dry_run)
        self.logger.info("Pipeline timeout: %d seconds", PIPELINE_TIMEOUT_SECONDS)

        self.groq_llm = GroqClient()
        self.claude_llm = ClaudeClient()
        self.vectordb = VectorDB()
        self.alex = Alex(self.claude_llm)

        try:
            await self._run_stage(
                "environment_validation", self.validate_environment, critical=True
            )

            if self.dry_run:
                self.logger.info("Dry-run finished successfully")
                self._save_partial_results(reason="dry-run completed")
                return 0

            generated = await self._run_stage(
                "hypothesis_generation", self.generate_hypotheses, critical=True
            )
            if not generated:
                raise RuntimeError("Hypothesis generation returned no candidates")

            # Stage 2.5: Feasibility check - validate hypothesis before expensive review
            selected_hypothesis = sorted(generated, key=self._rank_hypothesis, reverse=True)[0]
            await self._run_stage(
                "hypothesis_feasibility_check",
                lambda: self.check_hypothesis_feasibility(selected_hypothesis.hypothesis_text),
                critical=True,
            )

            reviewed_hypothesis = await self._run_stage(
                "hypothesis_review",
                lambda: self.review_hypothesis(generated),
                critical=True,
            )
            if not reviewed_hypothesis:
                raise RuntimeError("Hypothesis review did not return a final hypothesis")

            final_hypothesis_text = str(reviewed_hypothesis["hypothesis_text"])

            experiment_config = await self._run_stage(
                "experiment_translation",
                lambda: self.translate_experiment(final_hypothesis_text),
                critical=True,
            )
            if not experiment_config:
                raise RuntimeError("Experiment translation returned empty config")

            await self._run_stage(
                "smoke_test",
                lambda: self.run_smoke_test(experiment_config),
                critical=True,
            )

            full_results = await self._run_stage(
                "full_training",
                lambda: self.run_full_training(experiment_config),
                critical=True,
            )
            if not full_results:
                raise RuntimeError("Full training did not return results")

            await self._run_stage(
                "analysis_figures",
                lambda: self.analyze_results(full_results),
                critical=False,
            )

            paper_text = await self._run_stage(
                "paper_generation",
                lambda: self.generate_paper(final_hypothesis_text, experiment_config, full_results),
                critical=False,
            )

            if paper_text:
                reviewed_paper = await self._run_stage(
                    "paper_review",
                    lambda: self.review_generated_paper(
                        hypothesis_text=final_hypothesis_text,
                        config=experiment_config,
                        full_results=full_results,
                        paper_text=paper_text,
                    ),
                    critical=False,
                )

                if reviewed_paper:
                    paper_path = self.run_dir / "paper.tex"
                    paper_path.write_text(reviewed_paper)
                    self._register_artifact("paper_tex", paper_path)

                await self._run_stage("pdf_compilation", self.compile_pdf, critical=False)
            else:
                self.logger.warning("Skipping review/pdf: paper generation did not produce output")

            write_json(self.review_log_path, self.review_log)
            self._register_artifact("review_log", self.review_log_path)

            self._save_partial_results(reason="pipeline completed")
            self.logger.info("Full research pipeline completed")
            self.logger.info("Artifacts: %s", self.artifacts)
            return 0

        except Exception as exc:
            self.logger.error("Pipeline execution failed: %s", exc)
            self._save_partial_results(reason=f"pipeline failed: {exc}")
            return 1

        finally:
            write_json(self.review_log_path, self.review_log)
            write_json(self.stage_log_path, [asdict(item) for item in self.stage_records])

            if self.groq_llm is not None:
                try:
                    await self.groq_llm.close()
                except Exception as exc:
                    self.logger.warning("Failed to close GroqClient cleanly: %s", exc)

            if self.claude_llm is not None:
                try:
                    await self.claude_llm.close()
                except Exception as exc:
                    self.logger.warning("Failed to close ClaudeClient cleanly: %s", exc)

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 6)

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
        return round(math.sqrt(variance), 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Fleming-AI end-to-end research orchestration "
            "(hypothesis -> experiment -> analysis -> paper -> PDF)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment (MPS, API backends, VectorDB, pdflatex) without running training.",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "pipeline.log"
    configure_logging(log_path)

    pipeline = FullResearchPipeline(run_dir=run_dir, dry_run=args.dry_run)

    try:
        return await asyncio.wait_for(
            pipeline.run(),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logging.getLogger("run_full_research").error(
            "Pipeline timed out after %d seconds",
            PIPELINE_TIMEOUT_SECONDS,
        )
        pipeline._save_partial_results(
            reason=f"global timeout after {PIPELINE_TIMEOUT_SECONDS} seconds"
        )
        return 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
