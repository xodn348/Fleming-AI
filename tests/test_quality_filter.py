"""
Tests for Quality Filter and Pattern Extractor
"""

import pytest

from src.filters.pattern_extractor import PatternExtractor
from src.filters.quality import QualityAnalysis, QualityFilter


# Sample paper texts for testing
HIGH_QUALITY_PAPER = """
Title: A Novel Approach to Neural Network Optimization

Abstract:
We investigate how gradient descent can be improved for deep learning.
Our goal is to achieve faster convergence with better generalization.
We propose a novel approach that combines momentum with adaptive learning rates.

Introduction:
This paper investigates the fundamental question of how to optimize neural networks
more effectively. We aim to address the limitations of current optimization methods.
Our contribution includes a new algorithm that achieves state-of-the-art results.

Methodology:
We conduct extensive experiments with control groups and ablation studies.
The experimental design follows rigorous statistical principles.
Our method is reproducible and the code is available open source.

Results:
We achieved 95.2% accuracy on the benchmark dataset, with statistical significance
(p-value < 0.01, confidence interval [94.1%, 96.3%]).
This outperforms the baseline by 3.5% and represents a breakthrough in the field.

Discussion:
The practical implications of our work extend to real-world applications.
We discuss broader impact and ethical considerations of our approach.
Future work will explore additional applications.

Conclusion:
We have presented a novel approach with significant practical implications.
Limitations include the computational cost on very large datasets.
"""

LOW_QUALITY_PAPER = """
Title: Some Notes on Machine Learning

This paper talks about machine learning.
We did some experiments.
The results were good.
We think this is interesting.
Maybe someone will find this useful.
The end.
"""

THEORETICAL_PAPER = """
Title: A Theorem on Convergence Properties

Abstract:
We present a formal analysis of convergence properties.

Main Result:
Theorem 1: Under assumptions A1-A3, the algorithm converges.
Proof: By mathematical induction on n, we show that...
QED

Lemma 1: The sequence is bounded.
Corollary 1: The limit exists.

The derivation follows from first principles using formal methods.
"""

SURVEY_PAPER = """
Title: A Survey of Deep Learning Methods

Abstract:
This survey reviews recent advances in deep learning.

Introduction:
We present a systematic review and taxonomy of deep learning approaches.
The scope of this categorization includes methods from 2020-2024.

Related Work:
Our survey methodology follows established guidelines for systematic reviews.
"""


class TestPatternExtractor:
    """Tests for PatternExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create pattern extractor instance"""
        return PatternExtractor()

    def test_pattern_extractor_init(self, extractor):
        """Test pattern extractor initialization"""
        assert extractor.patterns is not None
        assert "research_question_types" in extractor.patterns
        assert "methodology_patterns" in extractor.patterns
        assert "evidence_patterns" in extractor.patterns

    def test_heuristic_extraction_high_quality(self, extractor):
        """Test heuristic pattern extraction on high quality paper"""
        patterns = extractor.extract_patterns_heuristic(HIGH_QUALITY_PAPER)

        assert "research_question_type" in patterns
        assert "methodology" in patterns
        assert "evidence_type" in patterns
        assert patterns["methodology"] == "experimental"
        assert patterns["evidence_type"] in ["quantitative", "comparative"]

    def test_heuristic_extraction_theoretical(self, extractor):
        """Test heuristic extraction on theoretical paper"""
        patterns = extractor.extract_patterns_heuristic(THEORETICAL_PAPER)

        assert patterns["methodology"] == "theoretical"
        assert patterns["evidence_type"] == "formal_proof"

    def test_heuristic_extraction_survey(self, extractor):
        """Test heuristic extraction on survey paper"""
        patterns = extractor.extract_patterns_heuristic(SURVEY_PAPER)

        assert patterns["methodology"] == "survey"

    def test_default_patterns_on_empty_text(self, extractor):
        """Test that empty text returns default patterns"""
        patterns = extractor.extract_patterns_heuristic("")

        assert patterns["research_question_type"] == "unknown"
        assert patterns["methodology"] == "unknown"
        assert patterns["evidence_type"] == "unknown"

    def test_validate_question_type(self, extractor):
        """Test question type validation"""
        assert extractor._validate_question_type("what") == "what"
        assert extractor._validate_question_type("HOW") == "how"
        assert extractor._validate_question_type("invalid") == "unknown"

    def test_validate_methodology(self, extractor):
        """Test methodology validation"""
        assert extractor._validate_methodology("experimental") == "experimental"
        assert extractor._validate_methodology("THEORETICAL") == "theoretical"
        assert extractor._validate_methodology("case study") == "case_study"
        assert extractor._validate_methodology("invalid") == "unknown"

    def test_validate_evidence_type(self, extractor):
        """Test evidence type validation"""
        assert extractor._validate_evidence_type("quantitative") == "quantitative"
        assert extractor._validate_evidence_type("formal proof") == "formal_proof"
        assert extractor._validate_evidence_type("invalid") == "unknown"


class TestQualityFilter:
    """Tests for QualityFilter class"""

    @pytest.fixture
    def quality_filter(self):
        """Create quality filter instance"""
        return QualityFilter()

    def test_quality_filter_init(self, quality_filter):
        """Test quality filter initialization"""
        assert quality_filter.patterns is not None
        assert quality_filter.pattern_extractor is not None
        assert "quality_criteria" in quality_filter.patterns

    def test_score_high_quality_paper(self, quality_filter):
        """Test scoring a high quality paper"""
        score = quality_filter.score(HIGH_QUALITY_PAPER)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.6, "High quality paper should score above 0.6"

    def test_score_low_quality_paper(self, quality_filter):
        """Test scoring a low quality paper"""
        score = quality_filter.score(LOW_QUALITY_PAPER)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.5, "Low quality paper should score below 0.5"

    def test_high_quality_beats_low_quality(self, quality_filter):
        """Test that high quality paper scores higher than low quality"""
        high_score = quality_filter.score(HIGH_QUALITY_PAPER)
        low_score = quality_filter.score(LOW_QUALITY_PAPER)

        assert high_score > low_score, "High quality should score higher"

    def test_analyze_heuristic_returns_analysis(self, quality_filter):
        """Test heuristic analysis returns QualityAnalysis"""
        analysis = quality_filter.analyze_heuristic(HIGH_QUALITY_PAPER)

        assert isinstance(analysis, QualityAnalysis)
        assert 0.0 <= analysis.overall_score <= 1.0
        assert 0.0 <= analysis.clarity_score <= 1.0
        assert 0.0 <= analysis.methodology_score <= 1.0
        assert 0.0 <= analysis.evidence_score <= 1.0
        assert 0.0 <= analysis.novelty_score <= 1.0
        assert 0.0 <= analysis.impact_score <= 1.0

    def test_analysis_to_dict(self, quality_filter):
        """Test QualityAnalysis to_dict method"""
        analysis = quality_filter.analyze_heuristic(HIGH_QUALITY_PAPER)
        analysis_dict = analysis.to_dict()

        assert isinstance(analysis_dict, dict)
        assert "overall_score" in analysis_dict
        assert "patterns" in analysis_dict
        assert "details" in analysis_dict

    def test_analysis_has_patterns(self, quality_filter):
        """Test that analysis includes extracted patterns"""
        analysis = quality_filter.analyze_heuristic(HIGH_QUALITY_PAPER)

        assert "research_question_type" in analysis.patterns
        assert "methodology" in analysis.patterns
        assert "evidence_type" in analysis.patterns

    def test_analysis_has_details(self, quality_filter):
        """Test that analysis includes detailed breakdown"""
        analysis = quality_filter.analyze_heuristic(HIGH_QUALITY_PAPER)

        assert "clarity" in analysis.details
        assert "methodology" in analysis.details
        assert "evidence" in analysis.details
        assert "novelty" in analysis.details
        assert "impact" in analysis.details

    def test_clarity_detection(self, quality_filter):
        """Test clarity score detection"""
        analysis = quality_filter.analyze_heuristic(HIGH_QUALITY_PAPER)

        # High quality paper has clear research question and objectives
        assert analysis.details["clarity"]["has_clear_question"] is True
        assert analysis.details["clarity"]["has_objectives"] is True
        assert analysis.clarity_score > 0.5

    def test_filter_papers(self, quality_filter):
        """Test filtering multiple papers"""
        papers = [HIGH_QUALITY_PAPER, LOW_QUALITY_PAPER, THEORETICAL_PAPER]
        filtered = quality_filter.filter_papers(papers, min_score=0.3)

        assert len(filtered) > 0
        # Results should be sorted by score descending
        scores = [f[2] for f in filtered]
        assert scores == sorted(scores, reverse=True)

    def test_filter_papers_threshold(self, quality_filter):
        """Test filtering with high threshold"""
        papers = [HIGH_QUALITY_PAPER, LOW_QUALITY_PAPER]
        filtered = quality_filter.filter_papers(papers, min_score=0.6)

        # At high threshold, only high quality should pass
        assert len(filtered) >= 1
        for idx, paper, score in filtered:
            assert score >= 0.6

    def test_empty_paper_handling(self, quality_filter):
        """Test handling of empty paper text"""
        score = quality_filter.score("")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_has_clear_research_question(self, quality_filter):
        """Test research question detection"""
        text_with_question = "We investigate how neural networks work."
        text_without = "Some random text here."

        assert quality_filter._has_clear_research_question(text_with_question) is True
        assert quality_filter._has_clear_research_question(text_without) is False

    def test_has_objectives(self, quality_filter):
        """Test objectives detection"""
        text_with_objectives = "Our goal is to improve accuracy."
        text_without = "Some random text here."

        assert quality_filter._has_objectives(text_with_objectives) is True
        assert quality_filter._has_objectives(text_without) is False


class TestQualityFilterAsync:
    """Async tests for QualityFilter (requires Ollama)"""

    @pytest.fixture
    def quality_filter(self):
        """Create quality filter instance"""
        return QualityFilter()

    @pytest.mark.asyncio
    async def test_async_analyze(self, quality_filter):
        """Test async LLM-based analysis"""
        analysis = await quality_filter.analyze(HIGH_QUALITY_PAPER)

        assert isinstance(analysis, QualityAnalysis)
        assert 0.0 <= analysis.overall_score <= 1.0
        assert analysis.patterns is not None

    @pytest.mark.asyncio
    async def test_async_score(self, quality_filter):
        """Test async scoring"""
        score = await quality_filter.score_async(HIGH_QUALITY_PAPER)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_async_pattern_extraction(self, quality_filter):
        """Test async pattern extraction via LLM"""
        patterns = await quality_filter.pattern_extractor.extract_patterns(HIGH_QUALITY_PAPER)

        assert "research_question_type" in patterns
        assert "methodology" in patterns
        assert "novelty_claim" in patterns
        assert "evidence_type" in patterns


class TestPatternExtractorAsync:
    """Async tests for PatternExtractor (requires Ollama)"""

    @pytest.fixture
    def extractor(self):
        """Create pattern extractor instance"""
        return PatternExtractor()

    @pytest.mark.asyncio
    async def test_llm_extraction_high_quality(self, extractor):
        """Test LLM-based pattern extraction"""
        patterns = await extractor.extract_patterns(HIGH_QUALITY_PAPER)

        assert patterns["research_question_type"] in ["what", "how", "why", "unknown"]
        assert patterns["methodology"] in [
            "experimental",
            "theoretical",
            "empirical",
            "survey",
            "case_study",
            "unknown",
        ]
        # Novelty claim should be extracted as string
        assert isinstance(patterns["novelty_claim"], str)

    @pytest.mark.asyncio
    async def test_llm_extraction_handles_errors(self, extractor):
        """Test that LLM extraction handles errors gracefully"""
        # Very short text might cause issues
        patterns = await extractor.extract_patterns("x")

        # Should return valid patterns even if extraction is poor
        assert "research_question_type" in patterns
        assert "methodology" in patterns
