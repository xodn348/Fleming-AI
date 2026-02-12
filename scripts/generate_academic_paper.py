#!/usr/bin/env python3
"""Generate academic paper from validated hypotheses"""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.llm.groq_client import GroqClient
from src.storage.hypothesis_db import HypothesisDatabase
from src.storage.database import PaperDatabase
from src.generators.academic_paper_generator import AcademicPaperGenerator
from src.generators.pdf_compiler import PDFCompiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def main():
    logger.info("=" * 60)
    logger.info("📄 Academic Paper Generation")
    logger.info("=" * 60)

    with HypothesisDatabase() as hyp_db:
        hypotheses = hyp_db.get_hypotheses_by_status("validated")
        logger.info(f"Loaded {len(hypotheses)} validated hypotheses")

    if len(hypotheses) < 5:
        logger.warning(f"Only {len(hypotheses)} hypotheses. Recommend at least 5.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return

    with PaperDatabase("data/db/papers.db") as paper_db:
        papers = paper_db.get_all_papers()
        logger.info(f"Loaded {len(papers)} source papers")

    async with GroqClient() as groq:
        generator = AcademicPaperGenerator(groq, hyp_db)

        logger.info("Generating abstract...")
        abstract = await generator.generate_abstract(hypotheses)

        logger.info("Generating introduction...")
        introduction = await generator.generate_introduction()

        logger.info("Generating methods...")
        methods = await generator.generate_methods()

        logger.info("Generating results...")
        results = await generator.generate_results(hypotheses)

        logger.info("Generating discussion...")
        discussion = await generator.generate_discussion(hypotheses)

        logger.info("Generating conclusion...")
        conclusion = await generator.generate_conclusion(hypotheses)

        logger.info("Generating bibliography...")
        bibliography = await generator.generate_bibliography(papers)

    template_path = Path("templates/academic_paper.tex")
    template = template_path.read_text()

    paper = template.replace(
        "{{title}}",
        "Novel Scientific Hypotheses Generated via Literature-Based Discovery",
    )
    paper = paper.replace("{{authors}}", "Fleming-AI System")
    paper = paper.replace("{{abstract}}", abstract)
    paper = paper.replace("{{introduction}}", introduction)
    paper = paper.replace("{{methods}}", methods)
    paper = paper.replace("{{results}}", results)
    paper = paper.replace("{{discussion}}", discussion)
    paper = paper.replace("{{conclusion}}", conclusion)
    paper = paper.replace("{{appendix}}", generator.generate_appendix(hypotheses))

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_file = output_dir / "fleming_ai_paper.tex"
    tex_file.write_text(paper)
    logger.info(f"✓ LaTeX file saved: {tex_file}")

    bib_file = output_dir / "references.bib"
    bib_file.write_text(bibliography)
    logger.info(f"✓ Bibliography saved: {bib_file}")

    logger.info("Compiling PDF...")
    compiler = PDFCompiler(output_dir)
    try:
        pdf_file = compiler.compile_latex_to_pdf(tex_file)

        logger.info("=" * 60)
        logger.info("✅ Paper generated successfully!")
        logger.info(f"📄 PDF: {pdf_file}")
        logger.info(f"📝 LaTeX: {tex_file}")
        logger.info("=" * 60)
    except RuntimeError as e:
        logger.error(f"PDF compilation failed: {e}")
        logger.info(f"LaTeX source saved at: {tex_file}")
        logger.info("You can compile manually with: pdflatex fleming_ai_paper.tex")


if __name__ == "__main__":
    asyncio.run(main())
