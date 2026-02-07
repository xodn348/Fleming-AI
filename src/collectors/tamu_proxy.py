#!/usr/bin/env python3
"""
TAMU Library EZProxy authentication for accessing subscription-based papers.

Provides authenticated access to subscription journals (Nature, Science, etc.)
through TAMU's EZProxy service using NetID + password credentials.
"""

import logging
import os
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class TAMUProxy:
    """
    TAMU EZProxy client for authenticated DOI downloads.

    Transforms DOI URLs to proxy URLs and maintains authenticated sessions
    for accessing subscription-based journals.
    """

    # EZProxy base URL
    PROXY_BASE = "https://srv-proxy2.library.tamu.edu"
    LOGIN_URL = f"{PROXY_BASE}/login"

    # DOI proxy pattern
    DOI_PROXY_BASE = "https://doi-org.srv-proxy2.library.tamu.edu"

    def __init__(self, netid: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize TAMU proxy client.

        Args:
            netid: TAMU NetID (defaults to TAMU_NETID env var)
            password: TAMU password (defaults to TAMU_PASSWORD env var)
        """
        self.netid = netid or os.getenv("TAMU_NETID")
        self.password = password or os.getenv("TAMU_PASSWORD")
        self.session: Optional[httpx.AsyncClient] = None
        self.authenticated = False

    def is_available(self) -> bool:
        """Check if TAMU credentials are configured."""
        return bool(self.netid and self.password)

    async def authenticate(self) -> bool:
        """
        Authenticate with TAMU EZProxy.

        Returns:
            True if authentication successful, False otherwise
        """
        if not self.is_available():
            logger.debug("TAMU credentials not configured")
            return False

        if self.authenticated:
            logger.debug("Already authenticated with TAMU proxy")
            return True

        try:
            # Create session with cookies
            self.session = httpx.AsyncClient(timeout=30.0, follow_redirects=True, verify=True)

            # Prepare login form data
            login_data = {
                "user": self.netid,
                "pass": self.password,
                "url": "https://doi.org/",  # Target after login
            }

            logger.info(f"Authenticating with TAMU EZProxy as {self.netid}")

            # Attempt login
            response = await self.session.post(self.LOGIN_URL, data=login_data)

            # Check if login was successful (no 401/403 in response)
            if response.status_code >= 400:
                logger.warning(f"TAMU authentication failed with status {response.status_code}")
                self.authenticated = False
                return False

            self.authenticated = True
            logger.info("✓ Successfully authenticated with TAMU EZProxy")
            return True

        except Exception as e:
            logger.error(f"TAMU authentication error: {e}")
            self.authenticated = False
            return False

    def transform_doi_url(self, doi: str) -> str:
        """
        Transform DOI URL to TAMU proxy URL.

        Args:
            doi: DOI identifier (e.g., "10.1038/s41586-021-03819-2")

        Returns:
            Proxied DOI URL
        """
        doi_clean = doi.strip()
        if doi_clean.startswith("10."):
            return f"{self.DOI_PROXY_BASE}/{doi_clean}"
        return f"{self.DOI_PROXY_BASE}/{doi_clean}"

    async def download_with_proxy(self, doi: str, timeout: float = 60.0) -> Optional[bytes]:
        """
        Download paper PDF via TAMU proxy.

        Args:
            doi: DOI identifier
            timeout: Request timeout in seconds

        Returns:
            PDF content as bytes, or None if failed
        """
        if not self.authenticated:
            logger.warning("Not authenticated with TAMU proxy")
            return None

        if not self.session:
            logger.error("No active session")
            return None

        try:
            proxy_url = self.transform_doi_url(doi)
            logger.info(f"Downloading via TAMU proxy: {proxy_url}")

            response = await self.session.get(proxy_url, timeout=timeout, follow_redirects=True)

            # Check for successful response
            if response.status_code == 403:
                logger.warning(
                    f"Access denied via proxy (403): {doi}. "
                    "Paper may not be available through TAMU subscription."
                )
                return None

            response.raise_for_status()

            # Verify PDF content type
            content_type = response.headers.get("content-type", "")
            if "application/pdf" not in content_type:
                logger.warning(f"Response is not PDF (content-type: {content_type})")
                return None

            logger.info(f"✓ Downloaded via TAMU proxy: {doi} ({len(response.content)} bytes)")
            return response.content

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading {doi}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {doi} via proxy: {e}")
            return None

    async def close(self):
        """Close the authenticated session."""
        if self.session:
            await self.session.aclose()
            self.session = None
            self.authenticated = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
