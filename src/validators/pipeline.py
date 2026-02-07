"""
Validation Pipeline for Fleming-AI
Orchestrates hypothesis validation using classification and execution
"""

import asyncio
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.generators.hypothesis import Hypothesis
from src.storage.hypothesis_db import HypothesisDatabase
from src.validators.classifier import HypothesisClassifier
from src.validators.result import (
    CLASS_COMPUTATIONAL,
    CLASS_DATA_DRIVEN,
    CLASS_EXPERIMENTAL,
    CLASS_THEORETICAL,
    STATUS_INCONCLUSIVE,
    STATUS_NOT_TESTABLE,
    STATUS_REFUTED,
    STATUS_VERIFIED,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Maximum execution time for validation code (seconds)
MAX_EXECUTION_TIME = 30

# Maximum output size (bytes)
MAX_OUTPUT_SIZE = 10000


class ValidationPipeline:
    """
    Pipeline for validating hypotheses.

    Workflow:
    1. Classify hypothesis by validation type
    2. For computational/data_driven: generate and run validation code
    3. Analyze results and determine validation status
    4. Store results in database
    """

    def __init__(
        self,
        ollama_client: Any,
        hypothesis_db: HypothesisDatabase,
        classifier: Optional[HypothesisClassifier] = None,
        sandbox_enabled: bool = True,
    ):
        """
        Initialize ValidationPipeline.

        Args:
            ollama_client: OllamaClient for code generation
            hypothesis_db: Database for storing results
            classifier: Optional HypothesisClassifier (created if not provided)
            sandbox_enabled: Whether to run code in sandbox (subprocess)
        """
        self.ollama = ollama_client
        self.db = hypothesis_db
        self.classifier = classifier or HypothesisClassifier(
            ollama_client=ollama_client,
            use_llm=True,
        )
        self.sandbox_enabled = sandbox_enabled

    async def validate(self, hypothesis: Hypothesis) -> ValidationResult:
        """
        Validate a hypothesis through the full pipeline.

        Args:
            hypothesis: Hypothesis to validate

        Returns:
            ValidationResult with status and evidence
        """
        start_time = time.time()

        # Classify the hypothesis
        classification = await self.classifier.classify_async(hypothesis)

        # Initialize result
        result = ValidationResult(
            hypothesis_id=hypothesis.id,
            status=STATUS_INCONCLUSIVE,
            classification=classification,
        )
        result.add_log(f"Classified as: {classification}")

        try:
            # Run appropriate validation based on classification
            if classification == CLASS_COMPUTATIONAL:
                result = await self.run_computational_validation(hypothesis, result)
            elif classification == CLASS_DATA_DRIVEN:
                result = await self.run_data_driven_validation(hypothesis, result)
            elif classification == CLASS_EXPERIMENTAL:
                result = await self.run_experimental_validation(hypothesis, result)
            elif classification == CLASS_THEORETICAL:
                result = await self.run_theoretical_validation(hypothesis, result)
            else:
                result.status = STATUS_NOT_TESTABLE
                result.add_log(f"Unknown classification: {classification}")

        except Exception as e:
            result.error = str(e)
            result.add_log(f"Validation error: {e}")
            logger.error(f"Validation failed for {hypothesis.id}: {e}")

        # Record execution time
        result.execution_time_ms = int((time.time() - start_time) * 1000)
        result.validated_at = datetime.now()

        # Update hypothesis status in DB
        self._update_hypothesis_status(hypothesis.id, result.status)

        return result

    async def run_computational_validation(
        self,
        hypothesis: Hypothesis,
        result: ValidationResult,
    ) -> ValidationResult:
        """
        Run computational validation by generating and executing code.

        Args:
            hypothesis: Hypothesis to validate
            result: ValidationResult to update

        Returns:
            Updated ValidationResult
        """
        result.add_log("Starting computational validation")

        # Generate validation code
        code = await self._generate_validation_code(hypothesis)

        if not code:
            result.add_log("Failed to generate validation code")
            result.status = STATUS_INCONCLUSIVE
            return result

        result.code_executed = code
        result.add_log("Generated validation code")

        # Execute in sandbox
        if self.sandbox_enabled:
            output, error, success = await self._execute_code_sandbox(code)
        else:
            output, error, success = await self._execute_code_simulated(code)

        result.add_evidence("output", output)
        result.add_evidence("execution_error", error)
        result.add_evidence("execution_success", success)

        if error:
            result.add_log(f"Execution error: {error[:200]}")

        if output:
            result.add_log(f"Output: {output[:200]}")

        # Analyze execution results
        result = await self._analyze_execution_results(hypothesis, result, output, error)

        return result

    async def run_data_driven_validation(
        self,
        hypothesis: Hypothesis,
        result: ValidationResult,
    ) -> ValidationResult:
        """
        Run data-driven validation (simulated for security).

        In a real implementation, this would:
        1. Identify relevant public datasets
        2. Download/access the data
        3. Run statistical analysis
        4. Compare with hypothesis predictions

        Args:
            hypothesis: Hypothesis to validate
            result: ValidationResult to update

        Returns:
            Updated ValidationResult
        """
        result.add_log("Starting data-driven validation (simulated)")

        # Generate data analysis approach
        analysis_plan = await self._generate_data_analysis_plan(hypothesis)
        result.add_evidence("analysis_plan", analysis_plan)
        result.add_log("Generated data analysis plan")

        # Simulate data validation
        # In production, this would actually query databases/APIs
        simulated_result = await self._simulate_data_validation(hypothesis)

        result.add_evidence("simulated_analysis", simulated_result)
        result.status = simulated_result.get("status", STATUS_INCONCLUSIVE)
        result.add_log(f"Simulated validation result: {result.status}")

        return result

    async def run_experimental_validation(
        self,
        hypothesis: Hypothesis,
        result: ValidationResult,
    ) -> ValidationResult:
        """
        Handle experimental hypotheses (cannot be validated programmatically).

        Args:
            hypothesis: Hypothesis to validate
            result: ValidationResult to update

        Returns:
            Updated ValidationResult with not_testable status
        """
        result.add_log("Hypothesis requires physical experimentation")
        result.status = STATUS_NOT_TESTABLE

        # Generate experimental design suggestion
        experiment_design = await self._generate_experiment_design(hypothesis)
        result.add_evidence("suggested_experiment", experiment_design)
        result.add_log("Generated suggested experimental design")

        return result

    async def run_theoretical_validation(
        self,
        hypothesis: Hypothesis,
        result: ValidationResult,
    ) -> ValidationResult:
        """
        Run theoretical analysis on hypothesis.

        Args:
            hypothesis: Hypothesis to validate
            result: ValidationResult to update

        Returns:
            Updated ValidationResult
        """
        result.add_log("Starting theoretical analysis")

        # Generate theoretical analysis
        analysis = await self._generate_theoretical_analysis(hypothesis)
        result.add_evidence("theoretical_analysis", analysis)

        # Determine status based on analysis
        if analysis.get("is_logically_sound", False):
            result.status = STATUS_VERIFIED
            result.add_log("Hypothesis is theoretically sound")
        elif analysis.get("has_contradictions", False):
            result.status = STATUS_REFUTED
            result.add_log("Hypothesis has logical contradictions")
        else:
            result.status = STATUS_INCONCLUSIVE
            result.add_log("Theoretical analysis inconclusive")

        return result

    async def _generate_validation_code(self, hypothesis: Hypothesis) -> str:
        """Generate Python code to validate a hypothesis."""
        prompt = f"""Generate a short Python script to computationally test this hypothesis.

Hypothesis: {hypothesis.hypothesis_text}

Connection:
- Concept A: {hypothesis.connection.get("concept_a", "N/A")}
- Concept B: {hypothesis.connection.get("concept_b", "N/A")}
- Bridging: {hypothesis.connection.get("bridging_concept", "N/A")}

Requirements:
1. Use only Python standard library (no external packages)
2. Keep code under 50 lines
3. Print clear output: "VERIFIED", "REFUTED", or "INCONCLUSIVE"
4. Include brief reasoning in output
5. Handle errors gracefully

Return ONLY the Python code, no explanation:
```python
# Your code here
```"""

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000,
            )

            # Extract code from response
            code = self._extract_code_block(response)
            return code

        except Exception as e:
            logger.error(f"Failed to generate validation code: {e}")
            return ""

    def _extract_code_block(self, response: str) -> str:
        """Extract Python code from markdown code block."""
        response = response.strip()

        # Try to find code block
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            # Skip language identifier if present
            newline = response.find("\n", start)
            if newline > start:
                start = newline + 1
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Return as-is if no code block found
        return response

    async def _execute_code_sandbox(
        self,
        code: str,
    ) -> tuple[str, str, bool]:
        """
        Execute code in a sandboxed subprocess.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (stdout, stderr, success)
        """
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code)
                temp_path = f.name

            # Execute in subprocess with timeout
            proc = await asyncio.create_subprocess_exec(
                "python3",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=MAX_EXECUTION_TIME,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return "", "Execution timed out", False

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

            stdout_str = stdout.decode("utf-8", errors="replace")[:MAX_OUTPUT_SIZE]
            stderr_str = stderr.decode("utf-8", errors="replace")[:MAX_OUTPUT_SIZE]

            return stdout_str, stderr_str, proc.returncode == 0

        except Exception as e:
            return "", str(e), False

    async def _execute_code_simulated(
        self,
        code: str,  # noqa: ARG002
    ) -> tuple[str, str, bool]:
        """
        Simulate code execution (for testing/security).

        Args:
            code: Python code (not actually executed)

        Returns:
            Simulated output tuple
        """
        # Return simulated successful execution
        return (
            "INCONCLUSIVE\nSimulated execution - actual code not run for security.",
            "",
            True,
        )

    async def _analyze_execution_results(
        self,
        hypothesis: Hypothesis,
        result: ValidationResult,
        output: str,
        error: str,
    ) -> ValidationResult:
        """Analyze code execution results to determine validation status."""
        output_lower = output.lower()

        # Check for explicit status in output
        if "verified" in output_lower:
            result.status = STATUS_VERIFIED
            result.add_log("Code output indicates VERIFIED")
        elif "refuted" in output_lower:
            result.status = STATUS_REFUTED
            result.add_log("Code output indicates REFUTED")
        elif "inconclusive" in output_lower:
            result.status = STATUS_INCONCLUSIVE
            result.add_log("Code output indicates INCONCLUSIVE")
        elif error:
            result.status = STATUS_INCONCLUSIVE
            result.add_log("Execution had errors, marking as inconclusive")
        else:
            # Use LLM to analyze output
            result = await self._analyze_output_with_llm(hypothesis, result, output)

        return result

    async def _analyze_output_with_llm(
        self,
        hypothesis: Hypothesis,
        result: ValidationResult,
        output: str,
    ) -> ValidationResult:
        """Use LLM to analyze execution output."""
        prompt = f"""Analyze this validation output for the hypothesis.

Hypothesis: {hypothesis.hypothesis_text}

Execution Output:
{output[:1000]}

Based on the output, determine if the hypothesis is:
- "verified": Evidence supports the hypothesis
- "refuted": Evidence contradicts the hypothesis
- "inconclusive": Not enough evidence either way

Return JSON: {{"status": "<status>", "reasoning": "<brief explanation>"}}"""

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )

            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            analysis = json.loads(response)
            status = analysis.get("status", "inconclusive")

            if status in [STATUS_VERIFIED, STATUS_REFUTED, STATUS_INCONCLUSIVE]:
                result.status = status
                result.add_log(f"LLM analysis: {analysis.get('reasoning', '')}")

        except Exception as e:
            logger.warning(f"LLM output analysis failed: {e}")
            result.status = STATUS_INCONCLUSIVE

        return result

    async def _generate_data_analysis_plan(self, hypothesis: Hypothesis) -> dict[str, Any]:
        """Generate a plan for data-driven validation."""
        prompt = f"""Create a data analysis plan to validate this hypothesis.

Hypothesis: {hypothesis.hypothesis_text}

Provide:
1. Relevant public datasets that could be used
2. Statistical methods to apply
3. Expected outcomes for verification/refutation

Return JSON:
{{"datasets": ["..."], "methods": ["..."], "verification_criteria": "..."}}"""

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=500,
            )

            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            return json.loads(response)

        except Exception as e:
            logger.warning(f"Data analysis plan generation failed: {e}")
            return {"error": str(e)}

    async def _simulate_data_validation(self, hypothesis: Hypothesis) -> dict[str, Any]:
        """Simulate data-driven validation (placeholder)."""
        # In production, this would actually query public data
        return {
            "status": STATUS_INCONCLUSIVE,
            "note": "Data validation simulated - requires actual data access",
            "hypothesis_id": hypothesis.id,
        }

    async def _generate_experiment_design(self, hypothesis: Hypothesis) -> dict[str, Any]:
        """Generate suggested experimental design for physical experiments."""
        prompt = f"""Design an experiment to test this hypothesis.

Hypothesis: {hypothesis.hypothesis_text}

Connection:
- Concept A: {hypothesis.connection.get("concept_a", "N/A")}
- Concept B: {hypothesis.connection.get("concept_b", "N/A")}

Provide:
1. Experimental setup
2. Variables to control
3. Measurements to take
4. Expected outcomes

Return JSON with fields: setup, controls, measurements, expected_outcomes"""

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=500,
            )

            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            return json.loads(response)

        except Exception as e:
            logger.warning(f"Experiment design generation failed: {e}")
            return {"error": str(e)}

    async def _generate_theoretical_analysis(self, hypothesis: Hypothesis) -> dict[str, Any]:
        """Generate theoretical analysis of hypothesis."""
        prompt = f"""Perform a theoretical analysis of this hypothesis.

Hypothesis: {hypothesis.hypothesis_text}

Analyze:
1. Logical soundness of the hypothesis
2. Any contradictions with known principles
3. Theoretical plausibility
4. Required assumptions

Return JSON:
{{
    "is_logically_sound": true/false,
    "has_contradictions": true/false,
    "plausibility": "high/medium/low",
    "assumptions": ["..."],
    "analysis": "..."
}}"""

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=500,
            )

            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            return json.loads(response)

        except Exception as e:
            logger.warning(f"Theoretical analysis failed: {e}")
            return {
                "is_logically_sound": False,
                "has_contradictions": False,
                "error": str(e),
            }

    def _update_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        """Update hypothesis status in database."""
        try:
            # Map validation status to hypothesis status
            if status == STATUS_VERIFIED:
                db_status = "validated"
            elif status == STATUS_REFUTED:
                db_status = "rejected"
            else:
                db_status = "pending"

            self.db.update_status(hypothesis_id, db_status)

        except Exception as e:
            logger.error(f"Failed to update hypothesis status: {e}")

    async def validate_batch(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[ValidationResult]:
        """
        Validate multiple hypotheses.

        Args:
            hypotheses: List of hypotheses to validate

        Returns:
            List of ValidationResults
        """
        results = []
        for hypothesis in hypotheses:
            result = await self.validate(hypothesis)
            results.append(result)
        return results

    def get_validation_stats(
        self,
        results: list[ValidationResult],
    ) -> dict[str, int]:
        """
        Get statistics on validation results.

        Args:
            results: List of ValidationResults

        Returns:
            Dictionary with counts for each status
        """
        stats = {
            STATUS_VERIFIED: 0,
            STATUS_REFUTED: 0,
            STATUS_INCONCLUSIVE: 0,
            STATUS_NOT_TESTABLE: 0,
        }

        for result in results:
            if result.status in stats:
                stats[result.status] += 1

        return stats
