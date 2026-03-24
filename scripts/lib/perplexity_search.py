"""Direct Perplexity Sonar Pro web search for last30days skill.

Uses Perplexity's native API (OpenAI-compatible) with the sonar-pro model.
Same logic as openrouter_search.py but hits api.perplexity.ai directly.

API docs: https://docs.perplexity.ai/api-reference
"""

import re
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from . import http

ENDPOINT = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"

EXCLUDED_DOMAINS = {
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
}


def search_web(
    topic: str,
    from_date: str,
    to_date: str,
    api_key: str,
    depth: str = "default",
) -> List[Dict[str, Any]]:
    """Search the web via Perplexity Sonar Pro directly.

    Args:
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        api_key: Perplexity API key (pplx-...)
        depth: 'quick', 'default', or 'deep'

    Returns:
        List of result dicts with keys: url, title, snippet, source_domain, date, relevance
    """
    max_tokens = {"quick": 1024, "default": 2048, "deep": 4096}.get(depth, 2048)

    prompt = (
        f"Find recent blog posts, news articles, tutorials, and discussions "
        f"about {topic} published between {from_date} and {to_date}. "
        f"Exclude results from reddit.com, x.com, and twitter.com. "
        f"For each result, provide the title, URL, publication date, "
        f"and a brief summary of why it's relevant."
    )

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    sys.stderr.write(f"[Web] Searching Sonar Pro via Perplexity for: {topic}\n")
    sys.stderr.flush()

    response = http.post(
        ENDPOINT,
        json_data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        timeout=30,
    )

    return _normalize_results(response)


def _normalize_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Sonar Pro response to websearch item schema.

    Perplexity returns:
    - citations: [url, ...] -- flat list of cited URLs
    - choices[0].message.content -- synthesized text with [N] references
    """
    items = []

    # Try search_results first (if present)
    search_results = response.get("search_results", [])
    if isinstance(search_results, list) and search_results:
        items = _parse_search_results(search_results)

    # Fall back to citations
    if not items:
        citations = response.get("citations", [])
        content = _get_content(response)
        if isinstance(citations, list) and citations:
            items = _parse_citations(citations, content)

    sys.stderr.write(f"[Web] Sonar Pro (Perplexity): {len(items)} results\n")
    sys.stderr.flush()

    return items


def _parse_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse structured search_results array."""
    items = []
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        url = result.get("url", "")
        if not url:
            continue
        try:
            domain = urlparse(url).netloc.lower()
            if domain in EXCLUDED_DOMAINS:
                continue
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = ""
        title = str(result.get("title", "")).strip()
        if not title:
            continue
        date = result.get("date")
        items.append({
            "id": f"W{i+1}",
            "title": title[:200],
            "url": url,
            "source_domain": domain,
            "snippet": str(result.get("snippet", result.get("description", ""))).strip()[:500],
            "date": date,
            "date_confidence": "med" if date else "low",
            "relevance": 0.7,
            "why_relevant": "",
        })
    return items


def _parse_citations(citations: List[str], content: str) -> List[Dict[str, Any]]:
    """Parse flat citations array, enriching with content context."""
    items = []
    for i, url in enumerate(citations):
        if not isinstance(url, str) or not url:
            continue
        try:
            domain = urlparse(url).netloc.lower()
            if domain in EXCLUDED_DOMAINS:
                continue
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = ""
        title = _extract_title_for_citation(content, i + 1) or domain
        items.append({
            "id": f"W{i+1}",
            "title": title[:200],
            "url": url,
            "source_domain": domain,
            "snippet": "",
            "date": None,
            "date_confidence": "low",
            "relevance": 0.6,
            "why_relevant": "",
        })
    return items


def _get_content(response: Dict[str, Any]) -> str:
    """Extract text content from chat completion response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_title_for_citation(content: str, index: int) -> Optional[str]:
    """Try to extract a title near a citation reference [N] in the content."""
    if not content:
        return None
    pattern = rf'\[{index}\][)\s]*([^\[\n]{{5,80}})'
    match = re.search(pattern, content)
    if match:
        title = match.group(1).strip().rstrip('.')
        title = re.sub(r'[*_`]', '', title)
        return title if len(title) > 3 else None
    return None
