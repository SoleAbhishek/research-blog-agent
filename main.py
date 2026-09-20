from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import TypedDict, List, Annotated, Literal
import operator
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

class Task(BaseModel):
    id: str
    title: str
    goal: str = Field(description="One sentence describing what the reader should be able to do/understand after this section")
    bullets: List[str] = Field(min_length=3, max_length=5, description="3-5 concrete, non-overlapping subpoints to cover in this section.")
    target_words: int = Field(description="Target word count for this section")
    section_type: Literal['intro', 'core', "example", 'checklist', 'common_mistakes', "conclusion"] = Field(description='The type of current section')
class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]
    audience: str = Field(description="Who is the blog for")
    tone: str = Field(description="Writing Tone(eg. Practical, Crisp)")


class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[list[str], operator.add]
    final_blog: str

llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

def create_plan(state: State):
    response = llm.with_structured_output(Plan).invoke([
        SystemMessage(content=(
            "You are an expert blog editor and information architect. Create a clear, "
            "accurate, practical outline for a high-quality blog post. Return only data "
            "that fits the requested schema.\n\n"
            "Planning requirements:\n"
            "- Choose a specific, useful blog title rather than repeating the topic.\n"
            "- Define the intended audience and use a consistent, appropriate writing tone.\n"
            "- Create 5-7 logically ordered sections with distinct purposes and no overlap.\n"
            "- Include exactly one introduction and one conclusion.\n"
            "- Use core sections for the main ideas, and use example, checklist, or "
            "common_mistakes sections only when they genuinely improve the article.\n"
            "- For every section, write a measurable goal, 3-5 concrete bullets, and a "
            "realistic target word count.\n"
            "- Make the outline useful to a writer: prefer specific concepts, examples, "
            "and reader outcomes over generic headings.\n"
            "- Order the sections so the article progresses naturally from context to "
            "understanding, application, and conclusion."
        )),
        HumanMessage(content=f'Topic: {state['topic']}')
    ])

    return {'plan': response}

def fanout(state: State):
    return [Send(
        'worker',
        {
            'task': task,
            'topic': state['topic'],
            'plan': state['plan']
        }
    )
    for task in state['plan'].tasks
    ]

def worker(payload: dict):
    response = llm.invoke([
        SystemMessage(content=(
            "You are a skilled blog writer. Write exactly one polished section of the "
            "planned blog post using the supplied section brief. Match the blog's audience "
            "and tone, explain ideas accurately, and prioritize clarity over filler.\n\n"
            "Writing requirements:\n"
            "- Return only the section in markdown; do not include commentary, planning "
            "notes, or a preamble.\n"
            "- Begin with one appropriate Markdown heading, unless this is the conclusion.\n"
            "- Address the section goal and cover every supplied bullet without repeating "
            "other sections.\n"
            "- Use short paragraphs, concrete examples, and lists only when they improve "
            "readability.\n"
            "- Stay close to the requested target word count.\n"
            "- Do not invent citations, statistics, quotations, or personal experiences."
        )),
        HumanMessage(content=(
            f'Blog Title: {payload['plan'].blog_title}\n'
            f'Topic: {payload['topic']}\n\n'
            f'Audience: {payload['plan'].audience}\n'
            f'Tone: {payload['plan'].tone}\n\n'
            f'Section: {payload['task'].title}\n'
            f'Section type: {payload['task'].section_type}\n'
            f'Goal: {payload['task'].goal}\n'
            f'Points to cover: {"; ".join(payload["task"].bullets)}\n'
            f'Target word count: {payload['task'].target_words}\n\n'
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

    return {'sections': [res]}

def reducer(state: State):
    """Combine generated sections, save the completed blog, and return it."""
    plan = state['plan']
    blog_title = plan.blog_title
    final_blog = f"# {blog_title}\n\n" + "\n\n".join(state['sections']) + "\n"

    filename = ''.join(
        char if char.isalnum() or char in (' ', '-', '_') else '_'
        for char in blog_title
    ).strip().replace(' ', '_') or 'final_blog'
    Path(f'{filename}.md').write_text(final_blog, encoding='utf-8')

    return {'final_blog': final_blog}

graph = StateGraph(State)
graph.add_node('orchestrator', create_plan)
graph.add_node('worker', worker)
graph.add_node('reducer', reducer)

graph.add_edge(START, 'orchestrator')
graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)

app = graph.compile()

app.invoke({'topic': 'Psycho-Cybernetics'})

