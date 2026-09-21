import re
from typing import List
from models import Task, EvidenceItem

def filter_relevant_evidence(
    task: Task,
    evidence: List[EvidenceItem],
    top_k: int = 4
) -> List[EvidenceItem]:
    """Prune and rank evidence items so each worker receives only context relevant to its section."""
    if not evidence or not task.requires_research:
        return []

    # Build search context from task metadata
    task_context = f"{task.title} {task.goal} {' '.join(task.bullets)} {' '.join(task.tags)}".lower()
    stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'what', 'how', 'are', 'was', 'your', 'about'}
    task_keywords = {word for word in re.findall(r'\b[a-z0-9_-]{3,}\b', task_context) if word not in stop_words}

    scored_evidence = []
    for item in evidence:
        title_lower = (item.title or '').lower()
        snippet_lower = (item.snippet or '').lower()
        # Weight matches in the title higher than in snippet body
        score = sum(2 for kw in task_keywords if kw in title_lower)
        score += sum(1 for kw in task_keywords if kw in snippet_lower)
        scored_evidence.append((score, item))

    # Sort descending by relevance score
    scored_evidence.sort(key=lambda x: x[0], reverse=True)

    # Return top matches with a positive score; fallback to top_k general evidence if no direct keywords matched
    matched = [item for score, item in scored_evidence if score > 0]
    return matched[:top_k] if matched else evidence[:top_k]
