from langchain_core.messages import SystemMessage, HumanMessage
from state import State
from models import RouterDecision
from prompts import ROUTER_SYSTEM_PROMPT
from config import fast_llm

def router(state: State) -> dict:
    """Classify the user's topic into closed_book, hybrid, or open_book research modes."""
    topic = state['topic']

    response = fast_llm.with_structured_output(RouterDecision).invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Topic: {topic}")
        ]
    )

    return {
        'needs_research': response.needs_research,
        'mode': response.mode,
        'queries': response.queries
    }

def route_next(state: State) -> str:
    """Conditional edge router: proceed to web research or jump directly to outline planning."""
    return "research" if state.get('needs_research', False) else "orchestrator"
