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
    queries = state.get("queries", []) or []
    max_results = 6

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

    extractor = fast_llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    # Deduplicate by URL
    dedup: Dict[str, EvidenceItem] = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}
