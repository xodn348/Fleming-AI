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

        # Add title + abstract as first chunk
        if paper_data.get("abstract"):
            chunks.append(
                {
                    "text": f"{title}\n\n{paper_data['abstract']}",
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
                # Split long sections into smaller chunks (max 1000 chars)
                section_text = paper_data[section]
                if len(section_text) > 1000:
                    # Split by paragraphs or sentences
                    parts = section_text.split("\n\n")
                    current_chunk = ""
                    for part in parts:
                        if len(current_chunk) + len(part) < 1000:
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
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                matches.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
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

        return {
            "paper_id": paper_id,
            "chunks": [
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
                for i in range(len(results["ids"]))
            ],
        }

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
