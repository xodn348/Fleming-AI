"""
Experiment Paper Generator for Fleming-AI
Generates NeurIPS-format LaTeX papers for actual ML experiments
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentPaperGenerator:
    """Generate NeurIPS-format academic papers from ML experiment results"""

    def __init__(self, groq_client: Any = None, vector_db: Any = None):
        """
        Initialize experiment paper generator

        Args:
            groq_client: GroqClient instance for LLM generation
            vector_db: VectorDB instance for related work citations
        """
        self.groq = groq_client
        self.db = vector_db

    async def generate_paper(
        self,
        hypothesis: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        template_path: str = "runs/templates/",
    ) -> str:
        """
        Generate complete LaTeX paper for an experiment

        Args:
            hypothesis: Research hypothesis being tested
            config: Experiment configuration (models, datasets, hyperparameters)
            results: Experiment results with metrics
            template_path: Path to NeurIPS template files

        Returns:
            Complete LaTeX document as string
        """
        logger.info(f"Generating paper for hypothesis: {hypothesis[:100]}...")

        title = await self._generate_title(hypothesis, results)

        logger.info("Generating abstract...")
        abstract = await self._generate_abstract(hypothesis, config, results)

        logger.info("Generating introduction...")
        introduction = await self._generate_introduction(hypothesis, config)

        logger.info("Generating related work...")
        related_work = await self._generate_related_work(hypothesis)

        logger.info("Generating methods...")
        methods = await self._generate_methods(hypothesis, config)

        logger.info("Generating experimental setup...")
        experimental_setup = await self._generate_experimental_setup(config)

        logger.info("Generating results...")
        results_section = await self._generate_results(hypothesis, config, results)

        logger.info("Generating discussion...")
        discussion = await self._generate_discussion(hypothesis, results)

        logger.info("Generating conclusion...")
        conclusion = await self._generate_conclusion(hypothesis, results)

        logger.info("Generating bibliography...")
        bibliography = await self._generate_bibliography(hypothesis)

        latex = self._assemble_latex(
            title=title,
            abstract=abstract,
            introduction=introduction,
            related_work=related_work,
            methods=methods,
            experimental_setup=experimental_setup,
            results=results_section,
            discussion=discussion,
            conclusion=conclusion,
            bibliography=bibliography,
        )

        logger.info("Paper generation complete")
        return latex

    async def _generate_title(self, hypothesis: str, results: Dict[str, Any]) -> str:
        """Generate paper title from hypothesis"""
        prompt = f"""Generate a concise, academic paper title for this research hypothesis:

Hypothesis: {hypothesis}

Key results: {self._format_results_summary(results)}

Requirements:
- 10-15 words maximum
- Formal academic style
- Highlight the main comparison or finding
- Follow NeurIPS title conventions
- Do NOT include "A Study of" or "An Analysis of" - be direct

Return ONLY the title, no quotes or additional text."""

        await asyncio.sleep(2)
        title = await self.groq.generate(prompt, temperature=0.3, max_tokens=100)
        return title.strip().strip('"')

    async def _generate_abstract(
        self, hypothesis: str, config: Dict[str, Any], results: Dict[str, Any]
    ) -> str:
        """Generate 150-200 word abstract"""
        models = config.get("models", [])
        datasets = config.get("datasets", [])
        results_text = self._format_results_summary(results)

        prompt = f"""Write an academic abstract (150-200 words) for a machine learning experiment paper.

Hypothesis: {hypothesis}

Models tested: {", ".join(models)}
Datasets: {", ".join(datasets)}
Key results: {results_text}

Structure the abstract to include:
1. Motivation (why this comparison matters)
2. Methods (experimental design, models, datasets)
3. Key quantitative results (include specific metrics)
4. Main conclusion (what the results show)

Write in formal academic style for NeurIPS. Be specific and quantitative.
Return ONLY the abstract text, no section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=400)

    async def _generate_introduction(self, hypothesis: str, config: Dict[str, Any]) -> str:
        """Generate introduction section"""
        models = config.get("models", [])
        datasets = config.get("datasets", [])

        prompt = f"""Write the Introduction section for an ML experiment paper testing this hypothesis:

Hypothesis: {hypothesis}

Models: {", ".join(models)}
Datasets: {", ".join(datasets)}

Structure (800-1000 words):
1. Background and motivation (why is this comparison important?)
2. Research question (what specific question are we answering?)
3. Experimental approach (brief overview of design)
4. Key contributions and findings preview
5. Paper organization (Section 2: Related Work, Section 3: Methods, etc.)

Write in formal NeurIPS style. Motivate the problem clearly.
Return ONLY the section content, no \\section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.4, max_tokens=1500)

    async def _generate_related_work(self, hypothesis: str) -> str:
        """Generate related work section using VectorDB citations"""
        MAX_SEARCH_TERMS = 3
        MAX_PAPERS_PER_TERM = 5
        MAX_TOTAL_PAPERS = 10

        related_papers = []
        if self.db:
            try:
                search_terms = await self._extract_search_terms(hypothesis)

                for term in search_terms[:MAX_SEARCH_TERMS]:
                    matches = self.db.search(term, k=MAX_PAPERS_PER_TERM)
                    related_papers.extend(matches)

                seen_ids = set()
                unique_papers = []
                for paper in related_papers:
                    paper_id = paper.get("metadata", {}).get("paper_id")
                    if paper_id and paper_id not in seen_ids:
                        seen_ids.add(paper_id)
                        unique_papers.append(paper)

                related_papers = unique_papers[:MAX_TOTAL_PAPERS]
                logger.info(f"Found {len(related_papers)} related papers from VectorDB")
            except Exception as e:
                logger.warning(f"VectorDB search failed: {e}")
                related_papers = []

        papers_text = self._format_papers_for_citation(related_papers)

        prompt = f"""Write the Related Work section for an ML experiment paper on this topic:

Hypothesis: {hypothesis}

Relevant papers from database:
{papers_text}

Structure (800-1000 words):
1. Overview of the research area
2. Discuss 5-8 most relevant papers, organized by theme
3. Identify gaps in current understanding
4. Position our contribution relative to prior work

Use academic citation style: Author et al. \\cite{{paper_key}}
Write in formal NeurIPS style.
Return ONLY the section content, no \\section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.4, max_tokens=1800)

    async def _generate_methods(self, hypothesis: str, config: Dict[str, Any]) -> str:
        """Generate methods section"""
        models = config.get("models", [])
        datasets = config.get("datasets", [])

        prompt = f"""Write the Methods section for an ML experiment.

Hypothesis: {hypothesis}
Models: {", ".join(models)}
Datasets: {", ".join(datasets)}
Configuration: {config}

Structure (1000-1200 words):
1. Experimental Design (factorial design, controlled variables)
2. Model Architectures (describe each model in detail)
3. Datasets (description, preprocessing, splits)
4. Evaluation Metrics (what metrics are used and why)
5. Statistical Analysis Plan (how results will be analyzed)

Be technical and detailed enough for reproducibility.
Write in formal NeurIPS style.
Return ONLY the section content, no \\section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=2000)

    async def _generate_experimental_setup(self, config: Dict[str, Any]) -> str:
        """Generate experimental setup section"""
        prompt = f"""Write the Experimental Setup section describing implementation details.

Configuration: {config}

Structure (600-800 words):
1. Hardware and Software (compute resources, frameworks, versions)
2. Hyperparameters (learning rates, batch sizes, epochs, optimizer settings)
3. Training Procedure (training loop, early stopping, checkpointing)
4. Evaluation Protocol (how models are tested, number of seeds)
5. Computational Cost (wall-clock time, GPU hours)

Be specific and quantitative. Include enough detail for exact reproduction.
Write in formal NeurIPS style.
Return ONLY the section content, no \\section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=1500)

    async def _generate_results(
        self, hypothesis: str, config: Dict[str, Any], results: Dict[str, Any]
    ) -> str:
        """Generate results section with actual metrics"""
        results_text = self._format_detailed_results(results)
        models = config.get("models", [])
        datasets = config.get("datasets", [])

        prompt = f"""Write the Results section presenting experimental findings.

Hypothesis: {hypothesis}
Models: {", ".join(models)}
Datasets: {", ".join(datasets)}

ACTUAL EXPERIMENTAL RESULTS:
{results_text}

Structure (1200-1500 words):
1. Overview of main findings
2. Per-model results (report actual metrics from results above)
3. Per-dataset results (break down by dataset)
4. Comparative analysis (which model performed better, by how much)
5. Statistical significance (reference p-values if available)
6. Reference tables and figures: Table~\\ref{{tab:main}}, Figure~\\ref{{fig:comparison}}

Use the ACTUAL numbers from the results. Be precise and objective.
Write in formal NeurIPS style.
Return ONLY the section content, no \\section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=2500)

    async def _generate_discussion(self, hypothesis: str, results: Dict[str, Any]) -> str:
        """Generate discussion section"""
        results_text = self._format_results_summary(results)

        prompt = f"""Write the Discussion section interpreting experimental results.

Hypothesis: {hypothesis}
Results summary: {results_text}

Structure (1000-1200 words):
1. Interpretation of findings (what do the results mean?)
2. Support or refutation of hypothesis (was it confirmed?)
3. Comparison with related work (how do results compare to literature?)
4. Implications for practice (what should practitioners do differently?)
5. Limitations (what are the constraints and caveats?)
6. Threats to validity (what could affect generalization?)
7. Future work (what follow-up experiments are needed?)

Be critical and balanced. Acknowledge limitations honestly.
Write in formal NeurIPS style.
Return ONLY the section content, no \\section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.4, max_tokens=1800)

    async def _generate_conclusion(self, hypothesis: str, results: Dict[str, Any]) -> str:
        """Generate conclusion section"""
        results_text = self._format_results_summary(results)

        prompt = f"""Write the Conclusion section summarizing the paper.

Hypothesis: {hypothesis}
Results summary: {results_text}

Structure (300-400 words):
1. Restate research question and approach
2. Summarize key findings (with specific numbers)
3. Main takeaway message
4. Broader implications for the field
5. Future research directions

Be concise and impactful. End on a forward-looking note.
Write in formal NeurIPS style.
Return ONLY the section content, no \\section header."""

        await asyncio.sleep(2)
        return await self.groq.generate(prompt, temperature=0.3, max_tokens=600)

    async def _generate_bibliography(self, hypothesis: str) -> str:
        """Generate BibTeX bibliography from VectorDB papers"""
        MAX_CITATIONS = 20
        MAX_SEARCH_TERMS = 3
        PAPERS_PER_TERM = 5

        entries = []

        if self.db:
            try:
                search_terms = await self._extract_search_terms(hypothesis)
                papers = []
                for term in search_terms[:MAX_SEARCH_TERMS]:
                    matches = self.db.search(term, k=PAPERS_PER_TERM)
                    papers.extend(matches)

                seen_ids = set()
                for i, paper in enumerate(papers[:MAX_CITATIONS]):
                    paper_id = paper.get("metadata", {}).get("paper_id")
                    if paper_id and paper_id not in seen_ids:
                        seen_ids.add(paper_id)
                        entry = self._create_bibtex_entry(paper, i)
                        if entry:
                            entries.append(entry)
            except Exception as e:
                logger.warning(f"Bibliography generation failed: {e}")

        if not entries:
            entries = self._get_placeholder_bibliography()

        return "\n\n".join(entries)

    async def _extract_search_terms(self, hypothesis: str) -> List[str]:
        """Extract key search terms from hypothesis using LLM"""
        prompt = f"""Extract 3-5 key technical terms or concepts from this hypothesis that would be good for searching an academic paper database:

Hypothesis: {hypothesis}

Return ONLY the search terms, one per line, no numbering or explanations."""

        result = await self.groq.generate(prompt, temperature=0.2, max_tokens=150)
        terms = [line.strip() for line in result.strip().split("\n") if line.strip()]
        return terms[:5]

    def _format_results_summary(self, results: Dict[str, Any]) -> str:
        """Format results as brief summary"""
        lines = []
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                lines.append(f"- {model_name}: {metric_str}")
            else:
                lines.append(f"- {model_name}: {metrics}")
        return "\n".join(lines) if lines else "Results available in experiment logs"

    def _format_detailed_results(self, results: Dict[str, Any]) -> str:
        """Format results with full detail"""
        lines = []
        for model_name, metrics in results.items():
            lines.append(f"\n{model_name}:")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"  {metric_name}: {value:.6f}")
                    else:
                        lines.append(f"  {metric_name}: {value}")
            else:
                lines.append(f"  Result: {metrics}")
        return "\n".join(lines) if lines else "No detailed results available"

    def _format_papers_for_citation(self, papers: List[Dict[str, Any]]) -> str:
        """Format VectorDB papers for LLM prompt"""
        MAX_PAPERS = 10
        TEXT_PREVIEW_LENGTH = 200

        lines = []
        for i, paper in enumerate(papers[:MAX_PAPERS]):
            metadata = paper.get("metadata", {})
            title = metadata.get("title", "Unknown Title")
            paper_id = metadata.get("paper_id", f"paper{i}")
            text = paper.get("text", "")[:TEXT_PREVIEW_LENGTH]
            lines.append(f"[{paper_id}] {title}\n  {text}...")
        return "\n\n".join(lines) if lines else "No related papers found"

    def _create_bibtex_entry(self, paper: Dict[str, Any], index: int) -> Optional[str]:
        """Create BibTeX entry from VectorDB paper"""
        metadata = paper.get("metadata", {})
        paper_id = metadata.get("paper_id", f"paper{index}")
        title = metadata.get("title", "Untitled")

        title = title.replace("{", "").replace("}", "").replace("\\", "")
        paper_id = paper_id.replace("/", "_").replace(":", "_")

        entry = f"""@article{{{paper_id},
  title = {{{title}}},
  year = {{2024}},
  note = {{Retrieved from database}}
}}"""
        return entry

    def _get_placeholder_bibliography(self) -> List[str]:
        """Get placeholder bibliography entries"""
        return [
            """@article{dosovitskiy2021vit,
  author = {Dosovitskiy, Alexey and others},
  title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  journal = {ICLR},
  year = {2021}
}""",
            """@article{he2016resnet,
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title = {Deep Residual Learning for Image Recognition},
  journal = {CVPR},
  year = {2016}
}""",
        ]

    def _assemble_latex(
        self,
        title: str,
        abstract: str,
        introduction: str,
        related_work: str,
        methods: str,
        experimental_setup: str,
        results: str,
        discussion: str,
        conclusion: str,
        bibliography: str,
    ) -> str:
        """Assemble complete LaTeX document"""
        latex = f"""\\documentclass{{article}}
\\usepackage{{neurips_2024}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{natbib}}
\\usepackage{{booktabs}}

\\title{{{title}}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
{introduction}

\\section{{Related Work}}
{related_work}

\\section{{Methods}}
{methods}

\\section{{Experimental Setup}}
{experimental_setup}

\\section{{Results}}
{results}

% Figure placeholders for experiment visualizations
\\begin{{figure}}[h]
  \\centering
  \\includegraphics[width=0.8\\textwidth]{{figures/loss_curve.pdf}}
  \\caption{{Training and validation loss curves for all models.}}
  \\label{{fig:loss}}
\\end{{figure}}

\\begin{{figure}}[h]
  \\centering
  \\includegraphics[width=0.8\\textwidth]{{figures/accuracy_comparison.pdf}}
  \\caption{{Accuracy comparison across models and datasets.}}
  \\label{{fig:comparison}}
\\end{{figure}}

% Table placeholder for main results
\\begin{{table}}[h]
  \\centering
  \\caption{{Main experimental results. Metrics reported are mean ± std over 3 seeds.}}
  \\label{{tab:main}}
  \\begin{{tabular}}{{lcccc}}
    \\toprule
    Model & Dataset & Accuracy & F1 Score & AUC \\\\
    \\midrule
    % Results to be filled from experiment logs
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\section{{Discussion}}
{discussion}

\\section{{Conclusion}}
{conclusion}

\\section*{{Acknowledgments}}
This research was conducted as part of the Fleming-AI autonomous research system. All experimental design, execution, and analysis were performed by the automated pipeline. Large language models assisted with paper writing and interpretation of results.

\\bibliographystyle{{plainnat}}

% Bibliography
\\begin{{thebibliography}}{{99}}

{self._format_bibliography_for_latex(bibliography)}

\\end{{thebibliography}}

\\end{{document}}
"""
        return latex

    def _format_bibliography_for_latex(self, bibtex: str) -> str:
        """Convert BibTeX to \\bibitem format for embedded bibliography"""
        items = []
        entries = bibtex.split("@article{")

        for entry in entries[1:]:
            if "}" in entry:
                key = entry.split(",")[0].strip()
                if "title = {" in entry:
                    title_start = entry.find("title = {") + 9
                    title_end = entry.find("}", title_start)
                    title = entry[title_start:title_end]
                    items.append(f"\\bibitem{{{key}}}\n{title}.")
                else:
                    items.append(f"\\bibitem{{{key}}}\nReference {key}.")

        return "\n\n".join(items) if items else "\\bibitem{placeholder}\nReferences to be added."
