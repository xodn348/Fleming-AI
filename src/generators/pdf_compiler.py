"""PDF Compiler for LaTeX documents"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFCompiler:
    """Compile LaTeX documents to PDF using pdflatex and bibtex"""

    def __init__(self, output_dir: Path | str = "data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compile_latex_to_pdf(self, tex_file: Path) -> Path:
        """
        Compile LaTeX to PDF using pdflatex and bibtex

        Args:
            tex_file: Path to .tex file

        Returns:
            Path to generated PDF file

        Raises:
            RuntimeError: If pdflatex is not installed or compilation fails
        """
        if not shutil.which("pdflatex"):
            raise RuntimeError("pdflatex not found. Install TeX Live or MacTeX.")

        original_dir = Path.cwd()
        tex_file = Path(tex_file).resolve()
        os.chdir(tex_file.parent)

        try:
            logger.info(f"Compiling {tex_file.name} to PDF...")

            # Use timeout to prevent hanging on missing packages
            # pdflatex with -interaction=nonstopmode should complete in ~30 seconds
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file.name],
                capture_output=True,
                check=False,
                timeout=60,
            )

            subprocess.run(
                ["bibtex", tex_file.stem],
                capture_output=True,
                check=False,
                timeout=30,
            )

            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file.name],
                capture_output=True,
                check=False,
                timeout=60,
            )

            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file.name],
                capture_output=True,
                check=False,
                timeout=60,
            )

            pdf_file = tex_file.with_suffix(".pdf")

            if not pdf_file.exists():
                raise RuntimeError(f"PDF file was not created: {pdf_file}")

            for ext in [".aux", ".log", ".bbl", ".blg", ".out"]:
                aux_file = tex_file.with_suffix(ext)
                if aux_file.exists():
                    aux_file.unlink()

            logger.info(f"✓ PDF compiled: {pdf_file}")
            return pdf_file

        except subprocess.TimeoutExpired as e:
            logger.error(f"LaTeX compilation timed out (likely missing packages): {e}")
            logger.error("Try installing missing packages with: mpm --install <package-name>")
            raise RuntimeError(f"PDF compilation timed out: {e}")

        except subprocess.CalledProcessError as e:
            logger.error(f"LaTeX compilation failed: {e}")
            log_file = tex_file.with_suffix(".log")
            if log_file.exists():
                logger.error(f"Check log file: {log_file}")
            raise RuntimeError(f"PDF compilation failed: {e}")

        finally:
            os.chdir(original_dir)
