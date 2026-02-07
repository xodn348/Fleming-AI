"""
VectorDB module for storing and retrieving paper embeddings using ChromaDB.
"""

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
import ollama


class VectorDB:
    """Vector database for paper embeddings using ChromaDB and Ollama."""

    def __init__(self, persist_dir: str = "data/db/chromadb"):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_dir: Directory to persist ChromaDB data
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Create or get collection for papers
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"description": "Research paper embeddings"},
        )

        self.embedding_model = "nomic-embed-text"

    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding using Ollama nomic-embed-text model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def _chunk_paper(self, paper_data: dict) -> list[dict]:
        """
        Split paper into chunks for embedding.

        Args:
            paper_data: Parsed paper data with sections

        Returns:
            List of chunks with metadata
        """
        chunks = []
        paper_id = paper_data.get("paper_id", "unknown")
        title = paper_data.get("title", "")

        # Add title + abstract as first chunk (truncate if too long)
        if paper_data.get("abstract"):
            abstract = paper_data["abstract"][:250]  # Limit abstract to 250 chars
            chunks.append(
                {
                    "text": f"{title}\n\n{abstract}",
                    "metadata": {
                        "paper_id": paper_id,
                        "section": "abstract",
                        "title": title,
                    },
                }
            )

        # Add each section as a chunk
        sections = ["introduction", "method", "results", "conclusion"]
        for section in sections:
            if paper_data.get(section):
                # Split long sections into smaller chunks (max 300 chars)
                section_text = paper_data[section]
                if len(section_text) > 300:
                    # Split by paragraphs or sentences
                    parts = section_text.split("\n\n")
                    current_chunk = ""
                    for part in parts:
                        if len(current_chunk) + len(part) < 300:
                            current_chunk += part + "\n\n"
                        else:
                            if current_chunk:
                                chunks.append(
                                    {
                                        "text": current_chunk.strip(),
                                        "metadata": {
                                            "paper_id": paper_id,
                                            "section": section,
                                            "title": title,
                                        },
                                    }
                                )
                            current_chunk = part + "\n\n"
                    if current_chunk:
                        chunks.append(
                            {
                                "text": current_chunk.strip(),
                                "metadata": {
                                    "paper_id": paper_id,
                                    "section": section,
                                    "title": title,
                                },
                            }
                        )
                else:
                    chunks.append(
                        {
                            "text": section_text,
                            "metadata": {
                                "paper_id": paper_id,
                                "section": section,
                                "title": title,
                            },
                        }
                    )

        return chunks

    def add_paper(self, paper_id: str, text: str, metadata: dict) -> None:
        """
        Add a single paper chunk to the vector database.

        Args:
            paper_id: Unique paper identifier
            text: Paper text content
            metadata: Additional metadata
        """
        embedding = self._generate_embedding(text)

        self.collection.add(
            ids=[f"{paper_id}_{metadata.get('section', 'unknown')}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def add_papers(self, papers: list[dict]) -> int:
        """
        Add multiple papers to the vector database.

        Args:
            papers: List of paper data dictionaries

        Returns:
            Number of chunks added
        """
        total_chunks = 0

        for paper_data in papers:
            chunks = self._chunk_paper(paper_data)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{chunk['metadata']['paper_id']}_{chunk['metadata']['section']}_{i}"
                embedding = self._generate_embedding(chunk["text"])

                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]],
                )
                total_chunks += 1

        return total_chunks

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for similar papers using semantic similarity.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of matching documents with metadata
        """
        query_embedding = self._generate_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        # Format results
        matches = []
        if results.get("ids") and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                matches.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": results["distances"][0][i]
                        if results.get("distances")
                        else None,
                    }
                )

        return matches

    def get_paper(self, paper_id: str) -> dict:
        """
        Get all chunks for a specific paper.

        Args:
            paper_id: Paper identifier

        Returns:
            Dictionary with paper chunks
        """
        results = self.collection.get(
            where={"paper_id": paper_id},
        )

        chunks = []
        if results and results.get("ids"):
            chunks = [
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                }
                for i in range(len(results["ids"]))
            ]

        return {
            "paper_id": paper_id,
            "chunks": chunks,
        }

    def get_all_paper_ids(self) -> list[str]:
        """
        Get all unique paper IDs in the database.

        Returns:
            List of unique paper IDs
        """
        results = self.collection.get()
        paper_ids = set()

        if results and results.get("metadatas"):
            metadatas = results.get("metadatas", [])
            if metadatas:
                for metadata in metadatas:
                    if metadata and "paper_id" in metadata:
                        paper_ids.add(metadata["paper_id"])

        return list(paper_ids)

    def count(self) -> int:
        """Get total number of chunks in the database."""
        return self.collection.count()

    def reset(self) -> None:
        """Reset the database (for testing)."""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"description": "Research paper embeddings"},
        )
