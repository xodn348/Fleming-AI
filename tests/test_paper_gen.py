"""Tests for paper generator."""

import subprocess
from pathlib import Path

import pytest

from src.generators.paper import PaperGenerator


@pytest.fixture
def sample_hypothesis():
    """Sample hypothesis data for testing."""
    return {
        "title": "Novel Approach to Quantum Computing",
        "abstract": "This paper presents a novel approach to quantum computing using AI-driven optimization.",
        "introduction": "Quantum computing has emerged as a promising field...",
        "related_work": "Previous work in this area includes...",
        "method": "Our method consists of three main steps...",
        "results": "We achieved 95% accuracy on benchmark datasets...",
        "discussion": "The results demonstrate the effectiveness of our approach...",
        "conclusion": "In conclusion, we have presented a novel method...",
        "references": "1. Smith et al. (2023). Quantum Computing Advances.\n2. Jones et al. (2024). AI Optimization.",
    }


@pytest.fixture
def paper_generator():
    """Create paper generator instance."""
    return PaperGenerator()


def test_paper_generator_init():
    """Test paper generator initialization."""
    generator = PaperGenerator()
    assert generator.template_path.exists()
    assert "{{title}}" in generator.template


def test_paper_generator_custom_template(tmp_path):
    """Test paper generator with custom template."""
    # Create custom template
    template_path = tmp_path / "custom_template.md"
    template_path.write_text("# {{title}}\n\n{{abstract}}")

    generator = PaperGenerator(str(template_path))
    assert generator.template_path == template_path


def test_paper_generator_missing_template():
    """Test paper generator with missing template."""
    with pytest.raises(FileNotFoundError):
        PaperGenerator("/nonexistent/template.md")


def test_generate_paper(paper_generator, sample_hypothesis):
    """Test paper generation."""
    paper = paper_generator.generate(sample_hypothesis)

    # Check that all sections are included
    assert sample_hypothesis["title"] in paper
    assert sample_hypothesis["abstract"] in paper
    assert sample_hypothesis["introduction"] in paper
    assert sample_hypothesis["related_work"] in paper
    assert sample_hypothesis["method"] in paper
    assert sample_hypothesis["results"] in paper
    assert sample_hypothesis["discussion"] in paper
    assert sample_hypothesis["conclusion"] in paper
    assert sample_hypothesis["references"] in paper

    # Check AI generation attribution
    assert "AI-Generated Research Paper" in paper
    assert "Fleming-AI" in paper
    assert "automatically generated" in paper


def test_generate_paper_missing_fields(paper_generator):
    """Test paper generation with missing fields."""
    hypothesis = {"title": "Test Paper"}
    paper = paper_generator.generate(hypothesis)

    # Should still generate paper with empty sections
    assert "Test Paper" in paper
    assert "AI-Generated Research Paper" in paper


def test_save_paper(paper_generator, sample_hypothesis, tmp_path):
    """Test saving paper to file."""
    output_path = tmp_path / "output" / "paper.md"
    paper = paper_generator.generate(sample_hypothesis)

    paper_generator.save(paper, str(output_path))

    assert output_path.exists()
    content = output_path.read_text()
    assert sample_hypothesis["title"] in content


def test_to_latex_without_pandoc(paper_generator, sample_hypothesis, monkeypatch):
    """Test LaTeX conversion when pandoc is not available."""

    def mock_run(*args, **kwargs):  # noqa: ARG001
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", mock_run)

    paper = paper_generator.generate(sample_hypothesis)
    latex = paper_generator.to_latex(paper)

    assert latex is None


def test_to_latex_with_pandoc(paper_generator, sample_hypothesis):
    """Test LaTeX conversion when pandoc is available."""
    # Check if pandoc is actually available
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            check=False,
        )
        pandoc_available = result.returncode == 0
    except FileNotFoundError:
        pandoc_available = False

    if not pandoc_available:
        pytest.skip("pandoc not installed")

    paper = paper_generator.generate(sample_hypothesis)
    latex = paper_generator.to_latex(paper)

    assert latex is not None
    assert isinstance(latex, str)
    # LaTeX should contain some LaTeX-specific commands
    assert "\\" in latex  # LaTeX commands start with backslash


def test_generate_and_save_markdown(paper_generator, sample_hypothesis, tmp_path):
    """Test generate and save in markdown format."""
    output_path = tmp_path / "paper.md"

    result_path = paper_generator.generate_and_save(
        sample_hypothesis,
        str(output_path),
        format="markdown",
    )

    assert result_path == str(output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert sample_hypothesis["title"] in content


def test_generate_and_save_latex_without_pandoc(
    paper_generator,
    sample_hypothesis,
    tmp_path,
    monkeypatch,
):
    """Test generate and save in LaTeX format without pandoc."""

    def mock_run(*args, **kwargs):  # noqa: ARG001
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", mock_run)

    output_path = tmp_path / "paper.tex"

    with pytest.raises(ValueError, match="pandoc"):
        paper_generator.generate_and_save(
            sample_hypothesis,
            str(output_path),
            format="latex",
        )


def test_generate_and_save_latex_with_pandoc(paper_generator, sample_hypothesis, tmp_path):
    """Test generate and save in LaTeX format with pandoc."""
    # Check if pandoc is actually available
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            check=False,
        )
        pandoc_available = result.returncode == 0
    except FileNotFoundError:
        pandoc_available = False

    if not pandoc_available:
        pytest.skip("pandoc not installed")

    output_path = tmp_path / "paper.tex"

    result_path = paper_generator.generate_and_save(
        sample_hypothesis,
        str(output_path),
        format="latex",
    )

    assert result_path == str(output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "\\" in content  # LaTeX commands


def test_paper_includes_date(paper_generator, sample_hypothesis):
    """Test that generated paper includes current date."""
    from datetime import datetime

    paper = paper_generator.generate(sample_hypothesis)
    current_date = datetime.now().strftime("%Y-%m-%d")

    assert current_date in paper
