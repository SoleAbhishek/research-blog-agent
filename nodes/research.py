from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import SystemMessage, HumanMessage
from state import State
from models import EvidencePack, EvidenceItem
from prompts import RESEARCH_SYSTEM_PROMPT
from tools.search import tavily_search
from config import fast_llm

def research_node(state: State) -> dict:
    """Execute concurrent web searches and synthesize raw results into deduplicated EvidenceItems."""
    queries = (state.get("queries", []) or [])[:5]
    max_results = 5

    if not queries:
        return {"evidence": []}

    raw_results: List[Dict] = []

    # Parallel search execution
    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
        futures = {executor.submit(tavily_search, q, max_results=max_results): q for q in queries}
        for future in futures:
            try:
                res = future.result()
                if res:
                    raw_results.extend(res)
            except Exception as e:
                print(f"Warning: Tavily search failed for query '{futures[future]}': {e}")

    if not raw_results:
        return {"evidence": []}

    # Compact snippet size to avoid token inflation in research synthesizer
    compact_results = [
        {
            'title': r.get('title', ''),
            'url': r.get('url', ''),
            'snippet': (r.get('snippet', '') or '')[:300],
            'published_at': r.get('published_at'),
            'source': r.get('source')
        }
        for r in raw_results
    ]

    extractor = fast_llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=f"Raw results:\n{compact_results}"),
        ]
    )

    # Deduplicate by URL
    dedup: Dict[str, EvidenceItem] = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}
