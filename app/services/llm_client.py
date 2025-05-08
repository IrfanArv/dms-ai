import os
import json
import requests
import asyncio
from typing import AsyncGenerator, Dict, Any
import aiohttp

OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ai.ptedm.com")


async def generate_response(prompt: str, model: str = "llama3", temperature: float = 0.7):
    url = f"{OLLAMA_URL}/api/generate"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "temperature": temperature
                }
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if "text" in data:
                                yield data["text"]
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"Error: {str(e)}"


def generate_embedding(texts: list[str], model: str = "llama3"):
    url = f"{OLLAMA_URL}/api/embeddings"
    embeddings = []
    for text in texts:
        resp = requests.post(url, json={"model": model, "prompt": text})
        resp.raise_for_status()
        data = resp.json()
        embeddings.append(data.get("embedding", []))
    return embeddings
