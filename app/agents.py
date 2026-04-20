import os
import sys

from google import genai
from google.genai import types

# Ensure scripts can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.query_rag import build_bm25, generate_answer, load_index_and_chunks, retrieve

aclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def execute_with_retry_async(func, *args, max_retries=6, **kwargs):
    import asyncio
    import re
    from logger import get_logger
    logger = get_logger(__name__)

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)

            # Daily quota exhausted — retrying won't help, fail immediately
            if "PerDay" in error_str or "GenerateRequestsPerDay" in error_str:
                logger.error(f"Daily API quota exhausted: {e}")
                raise e

            match = re.search(r"Please retry in ([0-9\.]+)s", error_str)
            if match:
                delay = float(match.group(1)) + 1.0
            else:
                delay = min(60, 2 ** attempt)

            if attempt == max_retries - 1:
                logger.error(f"API async error: {e} — Final attempt failed.")
                raise e

            logger.warning(f"API async error: {e} — retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(delay)

async def agent_1_internal_researcher(query: str, index=None, chunks=None, bm25=None, k: int = 8):
    """Agent 1: Queries the existing RAG corpus for internal knowledge."""
    try:
        if index is None or chunks is None or bm25 is None:
            index, chunks = load_index_and_chunks()
            bm25 = build_bm25(chunks)

        retrieved_chunks = retrieve(query, index, chunks, bm25=bm25, k=k, rerank=True)
        # Using the synchronous generate_answer from our existing script for simplicity,
        # but we could rewrite it asynchronously if needed.
        answer = generate_answer(query, retrieved_chunks)
        return answer, retrieved_chunks
    except Exception as e:
        return f"Error in internal research: {e}", []

async def agent_2_external_fact_checker(query: str, internal_answer: str, mcp_client) -> str:
    """Agent 2: Uses the MCP server to verify/update the internal answer with live web data."""
    search_prompt = f"Original question: '{query}'\nInternal answer: '{internal_answer}'\nWhat search query should we use to fact-check this with live web data? Reply ONLY with the search query string."

    response = await execute_with_retry_async(
        aclient.aio.models.generate_content,
        model="gemini-2.5-flash",
        contents=search_prompt
    )
    search_query = response.text.strip() if response.text else query

    # Use FastMCP client tool
    try:
        print(f"      [External Fact-Checker] Executing DDGS Search for: '{search_query}'")
        search_results = await mcp_client.call_tool("search_web", {"query": search_query, "max_results": 3})
    except Exception as e:
        return f"External Fact-Check couldn't execute search: {e}"

    analyze_prompt = f"""
Original Query: {query}
Internal Answer: {internal_answer}

Web Search Results:
{search_results}

Fact-check the Internal Answer using the Web Search Results. Summarize any new, contradicting, or supporting information found on the live web.
"""
    response2 = await execute_with_retry_async(
        aclient.aio.models.generate_content,
        model="gemini-2.5-flash",
        contents=analyze_prompt
    )
    return response2.text or "No external data verified."

async def agent_3_synthesizer(query: str, internal_answer: str, external_verification: str) -> str:
    """Agent 3: Merges both internal and external outputs into a final synthesized response."""
    prompt = f"""
You are the Synthesizer Agent. Your job is to merge internal corpus knowledge with external web fact-checks.

User Query: {query}

[Internal Corpus Findings]
{internal_answer}

[External Live Web Findings]
{external_verification}

Synthesize these into a cohesive final response. 
Make sure to clearly delineate what our internal corpus says vs. what the live external search reveals (e.g. "According to our corpus, X. However, a live search indicates Y.").
"""
    response = await execute_with_retry_async(
        aclient.aio.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a precise and helpful synthesis agent."
        )
    )
    return response.text or "Synthesis failed."
