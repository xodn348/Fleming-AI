"""
Direct-to-Drive storage: Upload PDFs directly to Drive without local temp files
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional
import httpx

from src.storage.vectordb import VectorDB
from src.parsers.pdf_parser import PDFParser
from src.collectors.tamu_proxy import TAMUProxy

logger = logging.getLogger(__name__)


class DirectDriveStorage:
    """
    Stream PDFs directly to Google Drive without local storage
    """

    SCI_HUB_MIRRORS = [
        "https://sci-hub.se",
        "https://sci-hub.st",
        "https://sci-hub.ru",
    ]

    def __init__(
        self,
        drive_papers_path: str = "gdrive:Fleming-AI/papers",
        drive_db_path: str = "gdrive:Fleming-AI/db",
    ):
        self.drive_papers_path = drive_papers_path
        self.drive_db_path = drive_db_path

    async def _try_scihub(self, doi: str) -> Optional[bytes]:
        """
        Try downloading from Sci-Hub mirrors as fallback

        Args:
            doi: Digital Object Identifier

        Returns:
            PDF content bytes if successful, None otherwise
        """
        for mirror in self.SCI_HUB_MIRRORS:
            try:
                url = f"{mirror}/{doi}"
                logger.info(f"Trying Sci-Hub mirror: {mirror}")
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        content_type = response.headers.get("content-type", "")
                        if "application/pdf" in content_type:
                            logger.info(f"✓ Downloaded from Sci-Hub: {len(response.content)} bytes")
                            return response.content
            except Exception as e:
                logger.debug(f"Sci-Hub mirror {mirror} failed: {e}")
                continue
        return None

    async def download_and_upload_paper(
        self,
        paper: dict,
        vectordb: VectorDB,
        tamu_proxy: Optional[TAMUProxy] = None,
    ) -> bool:
        """
        Download PDF to Drive via rclone copyurl, parse locally, then embed

        Args:
            paper: Paper metadata
            vectordb: VectorDB for embeddings
            tamu_proxy: Optional TAMU proxy

        Returns:
            True if successful
        """
        arxiv_id = paper.get("arxiv_id")
        doi = paper.get("doi")
        paper_id = arxiv_id or doi or f"paper_{paper.get('id')}"
        title = paper.get("title", "unknown")

        if not (arxiv_id or doi):
            logger.warning(f"No download link for: {title[:60]}")
            return False

        if arxiv_id:
            arxiv_id_clean = arxiv_id.replace("arXiv:", "").strip()
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
            filename = f"{arxiv_id_clean}.pdf"
        else:
            doi_clean = doi.strip() if doi else ""
            pdf_url = f"https://doi.org/{doi_clean}"
            filename = f"{doi_clean.replace('/', '_')}.pdf"

        logger.info(f"Processing: {title[:60]}")

        temp_path = Path(f"/tmp/{filename}")

        try:
            pdf_content = None

            if doi and tamu_proxy and tamu_proxy.authenticated:
                logger.debug(f"Using TAMU proxy for {doi}")
                pdf_content = await tamu_proxy.download_with_proxy(doi)

            if not pdf_content:
                logger.debug(f"Downloading to temp: {pdf_url}")
                try:
                    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                        response = await client.get(pdf_url)
                        response.raise_for_status()
                        if "application/pdf" in response.headers.get("content-type", ""):
                            pdf_content = response.content
                except Exception as e:
                    logger.warning(f"Direct download failed: {e}, trying Sci-Hub")

            if not pdf_content and doi:
                logger.debug(f"Trying Sci-Hub fallback for {doi}")
                pdf_content = await self._try_scihub(doi)

            if pdf_content:
                temp_path.write_bytes(pdf_content)
            else:
                logger.warning(f"Failed to download PDF for {title[:60]}")
                return False

            logger.info(f"  Downloaded {temp_path.stat().st_size} bytes")

            parser = PDFParser()
            parsed_data = parser.parse(str(temp_path))

            full_text = parsed_data.get("full_text", "")
            if len(full_text.strip()) < 100:
                logger.warning(f"  Text too short, skipping")
                return False

            parsed_data["paper_id"] = paper_id
            num_chunks = vectordb.add_papers([parsed_data])
            logger.info(f"  Embedded {num_chunks} chunks")

            logger.info(f"  Uploading to Drive via rclone copyurl...")
            process = await asyncio.create_subprocess_exec(
                "rclone",
                "copyurl",
                pdf_url,
                f"{self.drive_papers_path}/{filename}",
                "--auto-filename",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.warning(f"  rclone copyurl failed: {stderr.decode()}, trying rcat")
                pdf_content = temp_path.read_bytes()
                rcat_process = await asyncio.create_subprocess_exec(
                    "rclone",
                    "rcat",
                    f"{self.drive_papers_path}/{filename}",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await rcat_process.communicate(input=pdf_content)
                if rcat_process.returncode != 0:
                    logger.error(f"  Both copyurl and rcat failed")
                    return False

            logger.info(f"  ✓ Uploaded to Drive")
            return True

        except Exception as e:
            logger.error(f"  Processing failed: {e}")
            return False

        finally:
            temp_path.unlink(missing_ok=True)

    def backup_db_to_drive(self, db_dir: Path) -> bool:
        """
        Backup database to Drive

        Args:
            db_dir: Local DB directory

        Returns:
            True if successful
        """
        try:
            logger.info("Backing up DB to Drive...")
            result = subprocess.run(
                ["rclone", "copy", str(db_dir), self.drive_db_path, "--verbose"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("✓ DB backed up to Drive")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"DB backup failed: {e.stderr}")
            return False
