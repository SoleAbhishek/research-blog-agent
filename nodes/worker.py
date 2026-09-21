from typing import List
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from state import State
from prompts import WORKER_SYSTEM_PROMPT
from tools.evidence import filter_relevant_evidence, format_evidence_markdown
from config import writer_llm

def fanout(state: State) -> List[Send]:
    """Map each planned task to a parallel worker payload with pruned, section-relevant evidence."""
    all_evidence = state.get('evidence', [])
    mode = state.get('mode', 'closed_book')
    tasks = state['plan'].tasks

    payloads = []
    for idx, task in enumerate(tasks):
        # Only route pruned relevant evidence if research is needed
        task_evidence = (
            filter_relevant_evidence(task, all_evidence, top_k=4)
            if mode != 'closed_book' and task.requires_research
            else []
        )

        payloads.append(
            Send(
                'worker',
                {
                    'index': idx,
                    'task': task,
                    'topic': state['topic'],
                    'plan': state['plan'],
                    'mode': mode,
                    'evidence': task_evidence,
                }
            )
        )
    return payloads

def worker(payload: dict) -> dict:
    """Write an individual section using Tier 2 Creative Specialist model."""
    idx = payload.get('index', 0)
    formatted_evidence = format_evidence_markdown(payload.get('evidence', []), max_chars=250)
    response = writer_llm.invoke([
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Blog title: {payload['plan'].blog_title}\n"
            f"Topic: {payload['topic']}\n"
            f"Research mode: {payload.get('mode', 'closed_book')}\n"
            f"Audience: {payload['plan'].audience}\n"
            f"Tone: {payload['plan'].tone}\n\n"
            f"Section: {payload['task'].title}\n"
            f"Section type: {payload['task'].section_type}\n"
            f"Goal: {payload['task'].goal}\n"
            f"Points to cover: {'; '.join(payload['task'].bullets)}\n"
            f"Target word count: {payload['task'].target_words}\n"
            f"Requires research: {payload['task'].requires_research}\n"
            f"Requires citations: {payload['task'].requires_citations}\n\n"
            f"Available evidence:\n{formatted_evidence}\n\n"
            "Write the section now."
        ))
    ])

    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    res = str(content).strip()

    # Track index to ensure deterministic sorting at the reducer
    return {'sections': [(idx, res)]}
