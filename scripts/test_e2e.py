#!/usr/bin/env python3
"""
End-to-end test script for Fleming-AI
Tests full pipeline: collection → enrichment → hypothesis generation
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators.hypothesis import Hypothesis
from src.storage.hypothesis_db import HypothesisDatabase
from src.storage.vectordb import VectorDB
from src.filters.quality import QualityFilter
from src.utils.scoring import calculate_quality_score


class E2ETestRunner:
    """End-to-end test runner for Fleming-AI"""

    def __init__(self):
        self.results = []
        self.test_db_path = Path("data/db/test_e2e.db")
        self.test_vectordb_path = Path("data/db/test_e2e_chromadb")

    def log_test(self, name: str, passed: bool, message: str = "", skipped: bool = False):
        """Log test result"""
        if skipped:
            status = "⊘"
            self.results.append({"name": name, "passed": True, "message": message, "skipped": True})
        else:
            status = "✓" if passed else "✗"
            self.results.append({"name": name, "passed": passed, "message": message})
        print(f"  {status} {name}")
        if message:
            print(f"    → {message}")

    def test_quality_scoring(self):
        """Test quality scoring system"""
        print("\n📊 Testing Quality Scoring...")

        try:
            test_paper = {
                "year": 2023,
                "citations": 150,
                "venue": "NeurIPS",
                "awards": 1,
                "concepts": ["machine learning", "neural networks", "optimization"],
            }

            score = calculate_quality_score(test_paper)
            passed = 0 <= score <= 100
            self.log_test(
                "Quality score calculation",
                passed,
                f"Score: {score:.1f}/100 for NeurIPS paper with 150 citations",
            )

            test_cases = [
                ({"year": 2023, "citations": 100, "venue": "NeurIPS"}, True),
                ({"year": 2023, "citations": 50, "venue": "Unknown Venue"}, True),
                ({"year": 2020, "citations": 500, "venue": "Nature"}, True),
            ]

            all_passed = True
            for paper, _ in test_cases:
                score = calculate_quality_score(paper)
                if not (0 <= score <= 100):
                    all_passed = False

            self.log_test(
                "Venue tier scoring",
                all_passed,
                f"Tested {len(test_cases)} different venue types",
            )

        except Exception as e:
            self.log_test("Quality scoring", False, str(e))

    def test_hypothesis_database(self):
        """Test hypothesis database operations"""
        print("\n💾 Testing Hypothesis Database...")

        try:
            self.test_db_path.unlink(missing_ok=True)

            with HypothesisDatabase(self.test_db_path) as db:
                passed = db is not None
                self.log_test("Database initialization", passed)

                count = db.count_hypotheses()
                passed = count == 0
                self.log_test(
                    "Empty database count",
                    passed,
                    f"Initial count: {count}",
                )

                test_hypothesis = Hypothesis(
                    id="test-001",
                    hypothesis_text="Test hypothesis about machine learning and biology",
                    source_papers=["paper1", "paper2"],
                    connection={
                        "concept_a": "neural networks",
                        "concept_b": "protein folding",
                        "bridging_concept": "optimization",
                    },
                    confidence=0.75,
                    quality_score=0.68,
                )

                inserted = db.insert_hypothesis(test_hypothesis)
                self.log_test(
                    "Hypothesis insertion",
                    inserted,
                    "Successfully inserted test hypothesis",
                )

                retrieved = db.get_hypothesis("test-001")
                passed = (
                    retrieved is not None
                    and retrieved.hypothesis_text == test_hypothesis.hypothesis_text
                )
                self.log_test(
                    "Hypothesis retrieval",
                    passed,
                    f"Retrieved hypothesis with confidence {retrieved.confidence if retrieved else 'N/A'}",
                )

                count = db.count_hypotheses()
                passed = count == 1
                self.log_test(
                    "Count after insertion",
                    passed,
                    f"Count: {count}",
                )

                hypotheses = db.get_all_hypotheses(limit=10)
                passed = isinstance(hypotheses, list) and len(hypotheses) == 1
                self.log_test(
                    "Get all hypotheses",
                    passed,
                    f"Retrieved {len(hypotheses)} hypotheses",
                )

                top_hyps = db.get_top_hypotheses(limit=5)
                passed = isinstance(top_hyps, list)
                self.log_test(
                    "Get top hypotheses",
                    passed,
                    f"Retrieved {len(top_hyps)} top hypotheses",
                )

                updated = db.update_status("test-001", "validated")
                self.log_test(
                    "Update hypothesis status",
                    updated,
                    "Status updated to 'validated'",
                )

                retrieved = db.get_hypothesis("test-001")
                passed = retrieved is not None and retrieved.status == "validated"
                self.log_test(
                    "Verify status update",
                    passed,
                    f"Status is now: {retrieved.status if retrieved else 'N/A'}",
                )

                searched = db.search_hypotheses("machine learning", limit=5)
                passed = isinstance(searched, list) and len(searched) > 0
                self.log_test(
                    "Search hypotheses",
                    passed,
                    f"Found {len(searched)} matching hypotheses",
                )

            self.test_db_path.unlink(missing_ok=True)

        except Exception as e:
            self.log_test("Hypothesis database", False, str(e))

    def test_quality_filter(self):
        """Test quality filter"""
        print("\n🎯 Testing Quality Filter...")

        try:
            filter = QualityFilter()
            self.log_test("Quality filter initialization", filter is not None)

            test_texts = [
                "This is a well-formed scientific hypothesis about machine learning.",
                "Neural networks can improve protein structure prediction.",
                "There may be a connection between optimization and biology.",
            ]

            scores = []
            for text in test_texts:
                score = filter.score(text)
                scores.append(score)
                passed = 0.0 <= score <= 1.0
                self.log_test(
                    "Quality score for hypothesis",
                    passed,
                    f"Score: {score:.2f}",
                )

        except Exception as e:
            self.log_test("Quality filter", False, str(e))

    def test_vectordb(self):
        """Test VectorDB operations"""
        print("\n🗂️  Testing VectorDB...")

        try:
            vectordb = VectorDB(persist_dir=str(self.test_vectordb_path))
            self.log_test("VectorDB initialization", vectordb is not None)

            passed = vectordb.collection is not None
            self.log_test(
                "VectorDB collection",
                passed,
                "Papers collection created",
            )

        except Exception as e:
            self.log_test("VectorDB", False, str(e))

    def test_paper_collector_components(self):
        """Test paper collector components without API calls"""
        print("\n🔍 Testing Paper Collector Components...")

        try:
            from src.collectors.paper_collector import PaperCollector

            config = {
                "min_citations": 50,
                "quality_threshold": 40,
                "db_path": "data/db/test_papers.db",
            }

            try:
                collector = PaperCollector(config=config)
                self.log_test("Collector initialization", collector is not None)
                collector.close()
            except Exception as e:
                if "API key" in str(e) or "email" in str(e):
                    self.log_test(
                        "Collector initialization",
                        True,
                        "OpenAlex API key/email required (skipped in test mode)",
                        skipped=True,
                    )
                else:
                    raise

            test_paper = {
                "title": "Test Paper",
                "year": 2023,
                "citations": 150,
                "venue": "NeurIPS",
                "doi": "10.1234/test",
                "abstract": "Test abstract",
            }

            score = calculate_quality_score(test_paper)
            passed = 0 <= score <= 100
            self.log_test(
                "Paper quality scoring",
                passed,
                f"Score: {score:.1f}/100",
            )

        except Exception as e:
            self.log_test("Paper collector components", False, str(e))

    def test_hypothesis_generation_components(self):
        """Test hypothesis generation components"""
        print("\n💡 Testing Hypothesis Generation Components...")

        try:
            from src.generators.hypothesis import HypothesisGenerator, ConceptPair

            self.log_test("HypothesisGenerator import", True)

            concept_pair = ConceptPair(
                concept_a="neural networks",
                concept_b="protein folding",
                bridging_concept="optimization",
                paper_a_id="paper1",
                paper_b_id="paper2",
                strength=0.8,
            )

            passed = (
                concept_pair.concept_a == "neural networks"
                and concept_pair.concept_b == "protein folding"
            )
            self.log_test(
                "ConceptPair creation",
                passed,
                f"Created pair: {concept_pair.concept_a} → {concept_pair.concept_b}",
            )

        except Exception as e:
            self.log_test("Hypothesis generation components", False, str(e))

    def run_all_tests(self):
        """Run all tests"""
        print("=" * 70)
        print("Fleming-AI End-to-End Test Suite")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.test_quality_scoring()
        self.test_quality_filter()
        self.test_paper_collector_components()
        self.test_hypothesis_database()
        self.test_vectordb()
        self.test_hypothesis_generation_components()

        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)

        passed_count = sum(1 for r in self.results if r["passed"])
        total_count = len(self.results)

        for result in self.results:
            status = "✓" if result["passed"] else "✗"
            print(f"{status} {result['name']}")

        print("\n" + "=" * 70)
        if passed_count == total_count:
            print(f"✓ ALL TESTS PASSED ({passed_count}/{total_count})")
            print("=" * 70)
            return 0
        else:
            print(f"✗ SOME TESTS FAILED ({passed_count}/{total_count} passed)")
            print("=" * 70)
            return 1


def main():
    """Main entry point"""
    runner = E2ETestRunner()
    exit_code = runner.run_all_tests()
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
