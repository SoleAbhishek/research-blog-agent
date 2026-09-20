from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import TypedDict, List, Annotated, Literal, Optional
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
    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]
    audience: str = Field(description="Who is the blog for")
    tone: str = Field(description="Writing Tone(eg. Practical, Crisp)")
    blog_kind: Literal['Explainer', 'Tutorial', "news_roundup", 'comparison', 'system_design'] = 'Explainer'
    constaints: List[str] = Field(default_factory=list)

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal['closed_book', 'hybrid', 'open_book']
    queries: List[str] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Plan
    sections: Annotated[List[tuple[int, str]], operator.add]
    final_blog: str

llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

ROUTER_SYSTEM_PROMPT = """
You are the research-routing agent for a blog-generation system. Your job is to decide whether the requested topic can be written from stable general knowledge or requires web research. Return only a response that fits the RouterDecision schema.

Use the topic and the intended article scope to classify the request into exactly one mode:

1. closed_book
    Use this for evergreen topics whose core claims are unlikely to change, such as
    established concepts, timeless principles, historical background, general explanations,
    and instructional material that does not depend on current tools, products, or events.
    Set needs_research to false and return an empty queries list.

2. hybrid
    Use this when the main explanation is evergreen but the article would benefit from a
    small amount of current evidence, examples, product details, standards, or recent
    developments. Set needs_research to true and provide 1-3 focused search queries.
    Queries should validate the time-sensitive parts without replacing the stable explanation.

3. open_book
    Use this when freshness is central to the article, including news, weekly or monthly
    roundups, latest developments, current rankings, current prices, recent research,
    changing APIs or products, regulations, market data, or comparisons that depend on
    present-day facts. Set needs_research to true and provide 3-5 focused search queries.

Decision rules:
- If correctness depends on what is current, do not choose closed_book.
- Prefer hybrid when only examples or supporting evidence need updating.
- Choose open_book when most of the article's claims must be verified against current sources.
- Do not request research merely because a topic is popular or broad.
- Avoid duplicate queries. Make each query target a distinct fact, source type, or subtopic.
- Include useful qualifiers such as a year, date range, official source, or publication type
  when the topic requires current information.
- If the request is ambiguous, choose the least research-intensive mode that still supports
  factual accuracy; choose hybrid when uncertainty could materially affect the article.

Output constraints:
- mode must be exactly one of: closed_book, hybrid, open_book.
- needs_research must be false only for closed_book, and true for hybrid or open_book.
- queries must be empty for closed_book and contain concise, actionable queries otherwise.

if needs_research=true:
- output 3-10 high-signal queries.
- Queries should be scoped and specific
- if asked lastest/last week/month, etc reflect that in the queries
"""

def router(state: State):
    topic = state['topic']

    response = llm.with_structured_output(RouterDecision).invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f'Topic:{topic}')
        ]
    )

    return {
        'needs_research':response.needs_research,
        'mode': response.mode,
        'queries': response.queries
    }

def route_next(state: State):
    return "research" if state['needs_research']==True else 'orchestrator'

def tavily_search(query: str, max_results: int = 5):
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({'query': query})

    normalised: List[dict] = []

    for r in results or []:
        normalised.append(
            {
                'title': r.get('title') or '',
                'url': r.get('url') or '',
                'snippet': r.get('content') or r.get('snippet') or '',
                'published_at': r.get('published_date') or r.get('published_at'),
                'source': r.get('source')
            }
        )
    return normalised

RESEARCH_SYSTEM_PRMPT = """
You are a research synthesizer. You receive raw web-search results and must transform
them into a concise, deduplicated list of EvidenceItem objects. Return only data that
matches the EvidenceItem schema.

For each useful source:
- Preserve the source's title and URL when available.
- Use the source's own snippet or extracted content as the evidence snippet.
- Preserve the publication date and source name only when they are explicitly provided.
- If a title is missing, create a short descriptive title based only on the available
    source content; do not add new claims.
- Exclude results without a usable URL or meaningful supporting content.

Deduplication and synthesis rules:
- Treat URLs as duplicates when they differ only by tracking parameters, fragments,
    protocol, or a trailing slash.
- Keep one EvidenceItem per canonical source URL.
- When duplicate results refer to the same source, merge their useful non-conflicting
    snippets and keep the most complete title, publication date, and source name.
- Do not merge different URLs merely because they discuss the same topic.
- Prefer authoritative, primary, and recent sources when the raw results provide that
    information.
- Do not invent URLs, dates, publishers, quotations, statistics, or factual claims.
- Do not treat search-result metadata such as relevance scores as evidence.

Output requirements:
- Return a JSON array of EvidenceItem objects, or the requested structured output wrapper
    containing that array, depending on the caller's schema.
- Include only fields supported by the EvidenceItem schema: title, url, published_at,
    snippet, and source.
- Return an empty list when no result contains reliable evidence.
"""

def research_node(state: State) -> dict:

    # take the first 10 queries from state
    queries = (state.get("queries", []) or [])
    max_results = 6

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM_PRMPT),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}

def create_plan(state: State):

    evidence = state.get('evidence', [])
    mode = state.get('mode', 'closed_book')
    response = llm.with_structured_output(Plan).invoke([
        SystemMessage(content=(
            "You are an expert blog editor, information architect, and research planner. "
            "Create a clear, accurate, practical outline for a high-quality blog post. "
            "Return only data that fits the requested schema.\n\n"
            "Research context:\n"
            "- The research mode is provided as closed_book, hybrid, or open_book.\n"
            "- In closed_book mode, rely on stable general knowledge and do not create "
            "sections that depend on current web facts.\n"
            "- In hybrid mode, use the supplied evidence only for current examples, "
            "supporting facts, or recent developments; keep the main explanation evergreen.\n"
            "- In open_book mode, make current evidence central to the relevant sections "
            "and mark those sections as requiring research and citations.\n"
            "- Treat supplied evidence as the only source of current factual claims. Do not "
            "invent facts, sources, dates, statistics, or citations.\n"
            "- When a section uses evidence, include a useful source URL or source title in "
            "its bullets or tags so the writer can cite it later.\n\n"
            "Planning requirements:\n"
            "- Choose a specific, useful blog title rather than repeating the topic.\n"
            "- Define the intended audience and use a consistent, appropriate writing tone.\n"
            "- Create 5-7 logically ordered sections with distinct purposes and no overlap.\n"
            "- Include exactly one introduction and one conclusion.\n"
            "- Use core sections for the main ideas, and use example, checklist, or "
            "common_mistakes sections only when they genuinely improve the article.\n"
            "- For every section, write a measurable goal, 3-5 concrete bullets, and a "
            "realistic target word count.\n"
            "- Set requires_research=true for sections that depend on supplied web evidence.\n"
            "- Set requires_citations=true whenever a section contains externally verifiable "
            "claims or uses supplied evidence.\n"
            "- Set requires_code=true only when the section genuinely needs code or a "
            "technical implementation example.\n"
            "- Make the outline useful to a writer: prefer specific concepts, examples, "
            "and reader outcomes over generic headings.\n"
            "- Order the sections so the article progresses naturally from context to "
            "understanding, application, and conclusion."
        )),
        HumanMessage(
    content=(
        f"Topic: {state['topic']}\n"
        f"Research mode: {mode}\n\n"
        "Evidence gathered from web research:\n"
        f"{evidence}\n\n"
        "Use the evidence only for claims that require current or externally "
        "verifiable information. Do not invent facts or citations. "
        "Create the plan according to the required schema."
    )
)
    ])

    return {'plan': response}

def fanout(state: State):
    return [Send(
        'worker',
        {
            'task': task,
            'topic': state['topic'],
            'plan': state['plan'],
            'mode': state.get('mode', 'closed_book'),
            'evidence': state.get('evidence', []),
        }
    )
    for task in state['plan'].tasks
    ]

def worker(payload: dict):
    response = llm.invoke([
        SystemMessage(content=(
            "You are a skilled blog writer. Write exactly one polished section of the "
            "planned blog post using the supplied section brief and evidence. Match the "
            "blog's audience and tone, explain ideas accurately, and prioritize clarity "
            "over filler.\n\n"
            "Writing requirements:\n"
            "- Return only the section in markdown; do not include commentary, planning "
            "notes, or a preamble.\n"
            "- Begin with one appropriate Markdown heading, unless this is the conclusion.\n"
            "- Address the section goal and cover every supplied bullet without repeating "
            "other sections.\n"
            "- Use short paragraphs, concrete examples, and lists only when they improve "
            "readability.\n"
            "- Stay close to the requested target word count.\n"
            "- Do not invent citations, statistics, quotations, personal experiences, or "
            "source details.\n"
            "- For closed_book sections, do not make claims that depend on current web "
            "information.\n"
            "- For hybrid or open_book sections, use only the supplied evidence for current "
            "or externally verifiable claims.\n"
            "- If citations are required, cite the supplied sources with inline Markdown "
            "links using their exact URLs. Do not cite a source that does not support the "
            "claim.\n"
            "- If the supplied evidence is insufficient, write cautiously instead of "
            "guessing."
        )),
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
            f"Available evidence:\n{payload.get('evidence', [])}\n\n"
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
graph.add_node('research', research_node)
graph.add_node('router', router)

graph.add_edge(START, 'router')
graph.add_conditional_edges('router', route_next, ['research', 'orchestrator'])
graph.add_edge('research','orchestrator')
graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)

app = graph.compile()

app.invoke({'topic': 'Newest breakthroughs in AI research and their implications for the future of technology'})

