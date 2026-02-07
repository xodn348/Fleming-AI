"""
PDF downloader for arXiv papers with rate limiting.
"""

import asyncio
from pathlib import Path
from typing import List

import httpx


class PDFDownloader:
    """Download PDF papers from arXiv with rate limiting."""

    RATE_LIMIT_SECONDS = 3  # arXiv rate limit: 3 seconds between requests
    ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"

    def __init__(self, output_dir: str | Path = "data/papers"):
        """Initialize PDF downloader.

        Args:
            output_dir: Directory to save downloaded PDFs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def download_paper(self, arxiv_id: str, output_dir: str | Path | None = None) -> str:
        """Download a single paper PDF from arXiv.

        Args:
            arxiv_id: arXiv ID (e.g., '1706.03762')
            output_dir: Optional custom output directory (defaults to self.output_dir)

        Returns:
            Path to downloaded PDF file

        Raises:
            httpx.HTTPError: If download fails
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = save_dir / f"{arxiv_id}.pdf"

        # Skip if already downloaded
        if pdf_path.exists():
            return str(pdf_path)

        url = self.ARXIV_PDF_URL.format(arxiv_id=arxiv_id)

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Save PDF
            pdf_path.write_bytes(response.content)

        return str(pdf_path)

    async def batch_download(
        self, arxiv_ids: List[str], output_dir: str | Path | None = None
    ) -> List[dict]:
        """Download multiple papers with rate limiting.

        Args:
            arxiv_ids: List of arXiv IDs
            output_dir: Optional custom output directory

        Returns:
            List of dicts with keys: arxiv_id, path, status, error
        """
        results = []

        for i, arxiv_id in enumerate(arxiv_ids):
            result = {
                "arxiv_id": arxiv_id,
                "path": None,
                "status": "pending",
                "error": None,
            }

            try:
                path = await self.download_paper(arxiv_id, output_dir)
                result["path"] = path
                result["status"] = "success"
            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)

            results.append(result)

            # Rate limit: wait 3 seconds between downloads (except for last one)
            if i < len(arxiv_ids) - 1:
                await asyncio.sleep(self.RATE_LIMIT_SECONDS)

        return results
