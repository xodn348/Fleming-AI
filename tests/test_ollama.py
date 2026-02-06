"""
Tests for Ollama Client
"""

import pytest
from src.llm.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_ollama_health_check():
    """Test Ollama server health check"""
    async with OllamaClient() as client:
        is_healthy = await client.health_check()
        assert is_healthy is True, "Ollama server should be healthy"


@pytest.mark.asyncio
async def test_ollama_list_models():
    """Test listing available models"""
    async with OllamaClient() as client:
        models = await client.list_models()
        assert isinstance(models, list), "Should return a list of models"
        assert len(models) > 0, "Should have at least one model"

        # Check if our required models are present
        model_names = [m.get("name", "") for m in models]
        assert any("qwen2.5:7b" in name for name in model_names), "qwen2.5:7b should be available"
        assert any("nomic-embed-text" in name for name in model_names), (
            "nomic-embed-text should be available"
        )


@pytest.mark.asyncio
async def test_ollama_generate():
    """Test text generation"""
    async with OllamaClient(model="qwen2.5:7b") as client:
        prompt = "What is 2+2? Answer with just the number."
        response = await client.generate(prompt=prompt, temperature=0.1, max_tokens=10)

        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert "4" in response, "Response should contain the answer 4"


@pytest.mark.asyncio
async def test_ollama_generate_with_system():
    """Test generation with system prompt"""
    async with OllamaClient(model="qwen2.5:7b") as client:
        system = "You are a helpful assistant that answers in one word."
        prompt = "What is the capital of France?"

        response = await client.generate(
            prompt=prompt, system=system, temperature=0.1, max_tokens=5
        )

        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert "Paris" in response or "paris" in response.lower(), "Response should mention Paris"


@pytest.mark.asyncio
async def test_ollama_chat():
    """Test chat completion"""
    async with OllamaClient(model="qwen2.5:7b") as client:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 5+3? Answer with just the number."},
        ]

        response = await client.chat(messages=messages, temperature=0.1, max_tokens=10)

        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert "8" in response, "Response should contain the answer 8"


@pytest.mark.asyncio
async def test_ollama_streaming_generate():
    """Test streaming text generation"""
    async with OllamaClient(model="qwen2.5:7b") as client:
        prompt = "Count from 1 to 3."

        chunks = []
        async for chunk in await client.generate(
            prompt=prompt, temperature=0.1, max_tokens=20, stream=True
        ):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive at least one chunk"
        full_response = "".join(chunks)
        assert len(full_response) > 0, "Full response should not be empty"


@pytest.mark.asyncio
async def test_ollama_streaming_chat():
    """Test streaming chat completion"""
    async with OllamaClient(model="qwen2.5:7b") as client:
        messages = [{"role": "user", "content": "Say hello."}]

        chunks = []
        async for chunk in await client.chat(
            messages=messages, temperature=0.1, max_tokens=20, stream=True
        ):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive at least one chunk"
        full_response = "".join(chunks)
        assert len(full_response) > 0, "Full response should not be empty"


@pytest.mark.asyncio
async def test_ollama_embeddings():
    """Test text embeddings"""
    async with OllamaClient() as client:
        text = "This is a test sentence for embeddings."

        embedding = await client.embed(text)

        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(x, (int, float)) for x in embedding), "All values should be numeric"


@pytest.mark.asyncio
async def test_ollama_context_manager():
    """Test async context manager"""
    client = OllamaClient()

    async with client:
        is_healthy = await client.health_check()
        assert is_healthy is True

    # Client should be closed after context exit
    # health_check returns False when client is closed (doesn't raise)
    is_healthy_after_close = await client.health_check()
    assert is_healthy_after_close is False, (
        "Health check should return False after client is closed"
    )


@pytest.mark.asyncio
async def test_ollama_custom_parameters():
    """Test generation with custom parameters"""
    async with OllamaClient(model="qwen2.5:7b") as client:
        response = await client.generate(
            prompt="Write one word.", temperature=0.1, max_tokens=5, top_p=0.9, top_k=40
        )

        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
