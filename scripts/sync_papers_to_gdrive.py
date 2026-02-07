#!/usr/bin/env python3
"""
Sync papers to Google Drive and embed all papers from database
Downloads PDFs from arXiv/DOI, uploads to GDrive, and embeds locally
"""

import asyncio
import logging
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import PaperDatabase
from src.storage.gdrive import sync_to_drive, check_rclone_available
from src.storage.vectordb import VectorDB
from src.parsers.pdf_parser import PDFParser
from src.llm.ollama_client import OllamaClient
from src.collectors.tamu_proxy import TAMUProxy
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def download_paper_pdf(
    paper: dict[str, str], output_dir: Path, tamu_proxy: TAMUProxy | None = None
) -> Path | None:
    """
    Download paper PDF from arXiv or DOI with TAMU proxy fallback.

    Attempts download via TAMU proxy for subscription journals if credentials
    are available, then falls back to direct download for open access papers.

    Args:
        paper: Paper metadata dict
        output_dir: Directory to save PDF
        tamu_proxy: Optional TAMU proxy client for authenticated access

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

    output_path = output_dir / filename

    if output_path.exists():
        logger.info(f"PDF already exists: {filename}")
        return output_path

    logger.info(f"Downloading PDF: {filename}")

    pdf_content = None

    if doi and tamu_proxy and tamu_proxy.authenticated:
        logger.info(f"Attempting download via TAMU proxy for DOI: {doi}")
        pdf_content = await tamu_proxy.download_with_proxy(doi)

    if pdf_content is None:
        logger.info(f"Attempting direct download from {pdf_url}")
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                if "application/pdf" in response.headers.get("content-type", ""):
                    pdf_content = response.content
                else:
                    logger.warning(f"Response is not PDF: {response.headers.get('content-type')}")
                    return None

        except Exception as e:
            logger.error(f"Failed to download {pdf_url}: {e}")
            return None

    if pdf_content:
        output_path.write_bytes(pdf_content)
        logger.info(f"✓ Downloaded: {filename} ({len(pdf_content)} bytes)")
        return output_path

    return None


async def parse_and_embed_paper(
    pdf_path: Path,
    paper_id: str,
    vectordb: VectorDB,
    ollama: OllamaClient,
) -> bool:
    """
    Parse PDF and embed into VectorDB

    Args:
        pdf_path: Path to PDF file
        paper_id: Unique paper identifier
        vectordb: VectorDB instance
        ollama: Ollama client for embeddings

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
        logger.info(f"  Added {num_chunks} chunks")

        logger.info(f"✓ Embedded: {paper_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to parse/embed {pdf_path.name}: {e}")
        return False


async def main():
    """Main sync and embed workflow"""

    if not check_rclone_available():
        logger.error("rclone is not installed. Please install rclone first.")
        logger.error("Visit: https://rclone.org/install/")
        sys.exit(1)

    db_path = Path.home() / "Fleming-AI" / "data" / "db" / "papers.db"
    papers_dir = Path.home() / "Fleming-AI" / "data" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Fleming-AI: Sync Papers to GDrive & Embed All")
    logger.info("=" * 80)

    db = PaperDatabase(db_path)

    all_papers = db.get_all_papers()
    logger.info(f"Found {len(all_papers)} papers in database")

    vectordb = VectorDB()
    embedded_ids = set(vectordb.get_all_paper_ids())
    logger.info(f"Already embedded: {len(embedded_ids)} papers")

    papers_to_process = [
        p for p in all_papers if p["arxiv_id"] not in embedded_ids and p["doi"] not in embedded_ids
    ]
    logger.info(f"Papers to download & embed: {len(papers_to_process)}")

    if not papers_to_process:
        logger.info("All papers already embedded!")

        logger.info("Syncing papers to GDrive...")
        try:
            sync_to_drive(
                str(papers_dir),
                "gdrive:Fleming-AI/papers",
            )
            logger.info("✓ Synced to GDrive")
        except Exception as e:
            logger.error(f"GDrive sync failed: {e}")

        return

    tamu_proxy = TAMUProxy()
    if tamu_proxy.is_available():
        logger.info("TAMU credentials found, authenticating with EZProxy...")
        await tamu_proxy.authenticate()
    else:
        logger.info("TAMU credentials not configured, using direct downloads only")
        tamu_proxy = None

    async with OllamaClient() as ollama:
        downloaded_count = 0
        embedded_count = 0

        for i, paper in enumerate(papers_to_process, 1):
            logger.info(f"\n[{i}/{len(papers_to_process)}] Processing: {paper['title'][:60]}...")

            pdf_path = await download_paper_pdf(paper, papers_dir, tamu_proxy)

            if pdf_path:
                downloaded_count += 1

                paper_id = paper.get("arxiv_id") or paper.get("doi") or f"paper_{paper['id']}"

                success = await parse_and_embed_paper(pdf_path, paper_id, vectordb, ollama)
                if success:
                    embedded_count += 1

                if i % 10 == 0:
                    logger.info(
                        f"Progress: {i}/{len(papers_to_process)} - Downloaded: {downloaded_count}, Embedded: {embedded_count}"
                    )

                    logger.info("Syncing to GDrive...")
                    try:
                        sync_to_drive(
                            str(papers_dir),
                            "gdrive:Fleming-AI/papers",
                        )
                        logger.info("✓ Synced to GDrive")
                    except Exception as e:
                        logger.error(f"GDrive sync failed: {e}")

                await asyncio.sleep(2)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total papers in DB: {len(all_papers)}")
    logger.info(f"Already embedded: {len(embedded_ids)}")
    logger.info(f"Newly downloaded: {downloaded_count}")
    logger.info(f"Newly embedded: {embedded_count}")
    logger.info(f"Total embedded now: {len(embedded_ids) + embedded_count}")

    logger.info("\nFinal sync to GDrive...")
    try:
        sync_to_drive(
            str(papers_dir),
            "gdrive:Fleming-AI/papers",
        )
        logger.info("✓ All papers synced to GDrive: gdrive:Fleming-AI/papers")
    except Exception as e:
        logger.error(f"GDrive sync failed: {e}")

    if tamu_proxy:
        await tamu_proxy.close()

    db.close()
    logger.info("\n✓ Complete!")


if __name__ == "__main__":
    asyncio.run(main())
