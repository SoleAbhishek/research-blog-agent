from langchain_core.messages import SystemMessage, HumanMessage
from state import State
from models import Plan
from prompts import PLANNER_SYSTEM_PROMPT
from config import planner_llm
from tools.evidence import format_evidence_markdown

def create_plan(state: State) -> dict:
    """Architect the full blog outline, task specifications, and citation needs using the Tier 1 Reasoner."""
    evidence = state.get('evidence', [])
    mode = state.get('mode', 'closed_book')
    formatted_evidence = format_evidence_markdown(evidence, max_chars=250)

    response = planner_llm.with_structured_output(Plan).invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n"
                f"Research mode: {mode}\n\n"
                "Evidence gathered from web research:\n"
                f"{formatted_evidence}\n\n"
                "Use the evidence only for claims that require current or externally "
                "verifiable information. Do not invent facts or citations. "
                "Create the plan according to the required schema."
            )
        )
    ])

    return {'plan': response}
