import json
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults
from config import CACHE_DIR

SEARCH_CACHE_DIR = CACHE_DIR / "search"
SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def tavily_search(query: str, max_results: int = 5, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Execute a Tavily web search with automatic local caching to reduce cost and latency."""
    cache_key = hashlib.md5(f"{query.strip().lower()}_{max_results}".encode("utf-8")).hexdigest()
    cache_file = SEARCH_CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({'query': query})

    normalised: List[Dict[str, Any]] = []
    for r in results or []:
        normalised.append({
            'title': r.get('title') or '',
            'url': r.get('url') or '',
            'snippet': r.get('content') or r.get('snippet') or '',
            'published_at': r.get('published_date') or r.get('published_at'),
            'source': r.get('source')
        })

    if use_cache:
        try:
            cache_file.write_text(json.dumps(normalised, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return normalised
