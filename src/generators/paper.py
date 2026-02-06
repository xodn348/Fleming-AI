"""Paper generator for Fleming-AI scientific discovery system."""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class PaperGenerator:
    """Generate research papers in Markdown and LaTeX formats."""

    def __init__(self, template_path: str | None = None):
        """Initialize paper generator.

        Args:
            template_path: Path to paper template. Defaults to templates/paper_template.md
        """
        if template_path is None:
            # Default to templates/paper_template.md relative to project root
            self.template_path = (
                Path(__file__).parent.parent.parent / "templates" / "paper_template.md"
            )
        else:
            self.template_path = Path(template_path)

        if not self.template_path.exists():
            msg = f"Template not found: {self.template_path}"
            raise FileNotFoundError(msg)

        self.template = self.template_path.read_text()

    def generate(self, hypothesis: dict[str, Any]) -> str:
        """Generate Markdown paper from hypothesis data.

        Args:
            hypothesis: Dictionary containing paper sections:
                - title: Paper title
                - abstract: Abstract text
                - introduction: Introduction section
                - related_work: Related work section
                - method: Method section
                - results: Results section
                - discussion: Discussion section
                - conclusion: Conclusion section
                - references: References section

        Returns:
            Generated paper in Markdown format with AI generation attribution
        """
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Replace template variables
        paper = self.template
        paper = paper.replace("{{title}}", hypothesis.get("title", "Untitled Research Paper"))
        paper = paper.replace("{{date}}", current_date)
        paper = paper.replace("{{abstract}}", hypothesis.get("abstract", ""))
        paper = paper.replace("{{introduction}}", hypothesis.get("introduction", ""))
        paper = paper.replace("{{related_work}}", hypothesis.get("related_work", ""))
        paper = paper.replace("{{method}}", hypothesis.get("method", ""))
        paper = paper.replace("{{results}}", hypothesis.get("results", ""))
        paper = paper.replace("{{discussion}}", hypothesis.get("discussion", ""))
        paper = paper.replace("{{conclusion}}", hypothesis.get("conclusion", ""))
        paper = paper.replace("{{references}}", hypothesis.get("references", ""))

        return paper

    def to_latex(self, markdown: str) -> str | None:
        """Convert Markdown paper to LaTeX using pandoc.

        Args:
            markdown: Markdown content to convert

        Returns:
            LaTeX content if pandoc is available, None otherwise
        """
        try:
            # Check if pandoc is available
            result = subprocess.run(
                ["pandoc", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return None

            # Convert markdown to LaTeX
            result = subprocess.run(
                ["pandoc", "-f", "markdown", "-t", "latex"],
                input=markdown,
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout

        except FileNotFoundError:
            # pandoc not installed
            return None
        except subprocess.CalledProcessError:
            # Conversion failed
            return None

    def save(self, content: str, path: str) -> None:
        """Save paper content to file.

        Args:
            content: Paper content to save
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    def generate_and_save(
        self,
        hypothesis: dict[str, Any],
        output_path: str,
        format: str = "markdown",  # noqa: A002
    ) -> str:
        """Generate paper and save to file.

        Args:
            hypothesis: Dictionary containing paper sections
            output_path: Output file path
            format: Output format ('markdown' or 'latex')

        Returns:
            Path to saved file

        Raises:
            ValueError: If format is 'latex' but pandoc is not available
        """
        # Generate markdown
        markdown = self.generate(hypothesis)

        # Convert to LaTeX if requested
        if format == "latex":
            latex = self.to_latex(markdown)
            if latex is None:
                msg = "LaTeX conversion requires pandoc to be installed"
                raise ValueError(msg)
            content = latex
        else:
            content = markdown

        # Save to file
        self.save(content, output_path)

        return output_path
