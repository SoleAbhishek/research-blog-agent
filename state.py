from typing import TypedDict, List, Annotated, Optional
import operator
from models import EvidenceItem, Plan, VisualPlan

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Plan
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_content: Optional[str]
    visual_plan: Optional[VisualPlan]
    final_blog: str
