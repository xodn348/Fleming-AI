"""
Lean storage manager: Drive-first, minimal local storage
Downloads → Embeds → Uploads to Drive → Deletes local
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import shutil

from src.storage.gdrive import sync_to_drive, check_rclone_available
from src.storage.vectordb import VectorDB
from src.parsers.pdf_parser import PDFParser
from src.collectors.tamu_proxy import TAMUProxy
import httpx

logger = logging.getLogger(__name__)


class LeanStorageManager:
    """
    Manages lean local storage with Drive-first strategy:
    1. Download PDF to temp location
    2. Parse and embed into VectorDB
    3. Upload PDF to Drive
    4. Delete local PDF
    5. Periodically backup DB to Drive
    """

    def __init__(
        self,
        papers_dir: Path,
        db_dir: Path,
        drive_papers_path: str = "gdrive:Fleming-AI/papers",
        drive_db_path: str = "gdrive:Fleming-AI/db",
    ):
        """
        Initialize lean storage manager

        Args:
            papers_dir: Local temp directory for PDFs (will be cleared)
            db_dir: Local database directory
            drive_papers_path: Remote Drive path for papers
            drive_db_path: Remote Drive path for DB backups
        """
        self.papers_dir = Path(papers_dir)
        self.db_dir = Path(db_dir)
        self.drive_papers_path = drive_papers_path
        self.drive_db_path = drive_db_path

        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        if not check_rclone_available():
            logger.warning("rclone not available - Drive sync disabled")
            self.drive_enabled = False
        else:
            self.drive_enabled = True
            logger.info("Drive sync enabled")

    async def download_paper_pdf(
        self, paper: dict, tamu_proxy: Optional[TAMUProxy] = None
    ) -> Optional[Path]:
        """
        Download paper PDF (same logic as sync_papers_to_gdrive.py)

        Args:
            paper: Paper metadata dict
            tamu_proxy: Optional TAMU proxy client

        Returns:
            Path to downloaded PDF or None if failed
        """
        arxiv_id = paper.get("arxiv_id")
        doi = paper.get("doi")
        title = paper.get("title", "untitled")

        if arxiv_id:
            arxiv_id_clean = arxiv_id.replace("arXiv:", "").strip()
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
            filename = f"{arxiv_id_clean}.pdf"
        elif doi:
            doi_clean = doi.strip()
            filename = f"{doi_clean.replace('/', '_')}.pdf"
            pdf_url = f"https://doi.org/{doi_clean}"
        else:
            logger.warning(f"No arXiv ID or DOI for paper: {title}")
            return None

        output_path = self.papers_dir / filename

        if output_path.exists():
            logger.debug(f"PDF already exists: {filename}")
            return output_path

        logger.info(f"Downloading PDF: {filename}")

        pdf_content = None

        # Try TAMU proxy for DOI papers
        if doi and tamu_proxy and tamu_proxy.authenticated:
            logger.debug(f"Attempting download via TAMU proxy for DOI: {doi}")
            pdf_content = await tamu_proxy.download_with_proxy(doi)

        # Fallback to direct download
        if pdf_content is None:
            logger.debug(f"Attempting direct download from {pdf_url}")
            try:
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    response = await client.get(pdf_url)
                    response.raise_for_status()

                    if "application/pdf" in response.headers.get("content-type", ""):
                        pdf_content = response.content
                    else:
                        logger.warning(
                            f"Response is not PDF: {response.headers.get('content-type')}"
                        )
                        return None

            except Exception as e:
                logger.error(f"Failed to download {pdf_url}: {e}")
                return None

        if pdf_content:
            output_path.write_bytes(pdf_content)
            logger.info(f"✓ Downloaded: {filename} ({len(pdf_content)} bytes)")
            return output_path

        return None

    async def parse_and_embed(self, pdf_path: Path, paper_id: str, vectordb: VectorDB) -> bool:
        """
        Parse PDF and embed into VectorDB

        Args:
            pdf_path: Path to PDF file
            paper_id: Unique paper identifier
            vectordb: VectorDB instance

        Returns:
            True if successful, False otherwise
        """
        try:
            parser = PDFParser()

            logger.info(f"Parsing PDF: {pdf_path.name}")
            parsed_data = parser.parse(str(pdf_path))

            full_text = parsed_data.get("full_text", "")
            if not full_text or len(full_text.strip()) < 100:
                logger.warning(f"Extracted text too short for {pdf_path.name}")
                return False

            logger.info(f"Embedding paper: {paper_id}")
            parsed_data["paper_id"] = paper_id
            num_chunks = vectordb.add_papers([parsed_data])
            logger.info(f"  Added {num_chunks} chunks to VectorDB")

            return True

        except Exception as e:
            logger.error(f"Failed to parse/embed {pdf_path.name}: {e}")
            return False

    def upload_pdf_to_drive(self, pdf_path: Path) -> bool:
        """
        Upload single PDF to Drive

        Args:
            pdf_path: Local PDF file path

        Returns:
            True if successful, False otherwise
        """
        if not self.drive_enabled:
            logger.warning("Drive sync disabled, skipping upload")
            return False

        try:
            result = sync_to_drive(
                str(pdf_path.parent),
                self.drive_papers_path,
            )
            return result is not None
        except Exception as e:
            logger.error(f"Failed to upload {pdf_path.name} to Drive: {e}")
            return False

    def batch_upload_to_drive(self) -> bool:
        """
        Batch upload all PDFs in papers directory to Drive

        Returns:
            True if successful, False otherwise
        """
        if not self.drive_enabled:
            logger.warning("Drive sync disabled, skipping batch upload")
            return False

        try:
            logger.info("Batch uploading papers to Drive...")
            result = sync_to_drive(
                str(self.papers_dir),
                self.drive_papers_path,
            )
            if result:
                logger.info("✓ Batch upload complete")
                return True
            return False
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            return False

    def upload_and_delete_pdf(self, pdf_path: Path) -> bool:
        """
        Upload PDF to Drive then delete local copy

        Args:
            pdf_path: Local PDF file path

        Returns:
            True if successful, False otherwise
        """
        if not self.drive_enabled:
            logger.warning("Drive sync disabled, keeping local PDF")
            return False

        try:
            if pdf_path.exists():
                logger.info(f"Uploading {pdf_path.name} to Drive...")
                upload_success = self.upload_pdf_to_drive(pdf_path)

                if upload_success:
                    pdf_path.unlink()
                    logger.info(f"✓ Uploaded and deleted local: {pdf_path.name}")
                    return True
                else:
                    logger.warning(f"Upload failed, keeping local: {pdf_path.name}")
                    return False
            return False
        except Exception as e:
            logger.error(f"Upload and delete failed: {e}")
            return False

    def delete_local_pdf(self, pdf_path: Path) -> bool:
        """
        Delete local PDF file

        Args:
            pdf_path: Local PDF file path

        Returns:
            True if successful, False otherwise
        """
        try:
            if pdf_path.exists():
                pdf_path.unlink()
                logger.info(f"✓ Deleted local PDF: {pdf_path.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete {pdf_path.name}: {e}")
            return False

    async def process_paper_lean(
        self,
        paper: dict,
        vectordb: VectorDB,
        tamu_proxy: Optional[TAMUProxy] = None,
    ) -> bool:
        """
        Full lean pipeline: Download → Embed → Upload → Delete

        Args:
            paper: Paper metadata dict
            vectordb: VectorDB instance
            tamu_proxy: Optional TAMU proxy client

        Returns:
            True if fully successful, False otherwise
        """
        # Download PDF
        pdf_path = await self.download_paper_pdf(paper, tamu_proxy)
        if not pdf_path:
            logger.warning(f"Failed to download paper: {paper.get('title', 'unknown')[:50]}")
            return False

        # Parse and embed
        paper_id = paper.get("arxiv_id") or paper.get("doi") or f"paper_{paper['id']}"
        embed_success = await self.parse_and_embed(pdf_path, paper_id, vectordb)

        if not embed_success:
            logger.warning(f"Failed to embed paper: {paper_id}")
            # Still try to upload PDF even if embedding failed
            pass

        # Upload to Drive
        if self.drive_enabled:
            upload_success = self.upload_pdf_to_drive(pdf_path)
            if not upload_success:
                logger.warning(f"Failed to upload to Drive: {pdf_path.name}")
                # Don't delete if upload failed
                return False

        # Delete local PDF
        self.delete_local_pdf(pdf_path)

        return embed_success

    def backup_db_to_drive(self) -> bool:
        """
        Backup entire DB directory to Drive

        Returns:
            True if successful, False otherwise
        """
        if not self.drive_enabled:
            logger.warning("Drive sync disabled, skipping DB backup")
            return False

        try:
            logger.info("Backing up DB to Drive...")
            result = sync_to_drive(
                str(self.db_dir),
                self.drive_db_path,
            )
            if result:
                logger.info("✓ DB backed up to Drive")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to backup DB to Drive: {e}")
            return False

    def cleanup_temp_pdfs(self) -> int:
        """
        Delete all PDFs in papers_dir (emergency cleanup)

        Returns:
            Number of files deleted
        """
        deleted = 0
        for pdf_file in self.papers_dir.glob("*.pdf"):
            try:
                pdf_file.unlink()
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {pdf_file.name}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} temp PDF files")

        return deleted
