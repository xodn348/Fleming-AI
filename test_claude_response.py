#!/usr/bin/env python3
"""Test script to check actual Claude API response format"""

import asyncio
import json
import os
from src.llm.claude_client import ClaudeClient


async def test_claude_response():
    """Test Claude API and print raw response"""
    client = ClaudeClient()

    # Simple test message
    test_prompt = "Say 'hello' and nothing else."

    print("=" * 80)
    print("Testing Claude API Response Format")
    print("=" * 80)

    try:
        # Create conversation
        conv_id = await client._create_conversation()
        print(f"✓ Conversation created: {conv_id}")

        # Send message with detailed logging
        print(f"\nSending prompt: {test_prompt}")

        # Manually call the API to inspect raw response
        headers = client._get_headers()
        payload = {
            "prompt": test_prompt,
            "attachments": [],
        }

        import httpx

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                f"{client.base_url}/organizations/{client.org_id}/chat_conversations/{conv_id}/completion",
                headers=headers,
                json=payload,
                timeout=30.0,
            )

            print(f"\n✓ HTTP Status: {response.status_code}")
            print(f"✓ Response headers: {dict(response.headers)}")
            print(f"\n{'=' * 80}")
            print("RAW RESPONSE TEXT:")
            print(f"{'=' * 80}")
            print(response.text[:2000])  # First 2000 chars
            print(f"{'=' * 80}")

            # Try to parse
            print("\nParsing attempt:")
            full_response = ""
            line_count = 0
            for line in response.text.split("\n"):
                line_count += 1
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        print(f"  Line {line_count}: {list(data.keys())}")

                        # Check all possible completion fields
                        if "completion" in data:
                            full_response += data["completion"]
                            print(f"    → Found 'completion': {data['completion'][:50]}")
                        elif "message" in data:
                            print(f"    → Found 'message': {data.get('message', {})}")
                        elif "text" in data:
                            print(f"    → Found 'text': {data['text'][:50]}")
                        else:
                            print(f"    → No completion field. Keys: {list(data.keys())}")
                    except json.JSONDecodeError as e:
                        print(f"  Line {line_count}: JSON decode error: {e}")
                        print(f"    Content: {line[:100]}")

            print(f"\n{'=' * 80}")
            print(f"PARSED RESULT: {len(full_response)} chars")
            print(f"{'=' * 80}")
            if full_response:
                print(full_response)
            else:
                print("⚠️  EMPTY RESPONSE - This is the problem!")

        # Cleanup
        await client._delete_conversation(conv_id)
        print(f"\n✓ Conversation deleted: {conv_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_claude_response())
