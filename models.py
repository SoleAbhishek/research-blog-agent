from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class ImagePlacement(BaseModel):
    id: str = Field(description="Unique short id like 'hero', 'diagram_1', 'concept_2'")
    target_heading: str = Field(description="The section heading (or 'top' for hero cover image) where this image belongs")
    placement: Literal['top', 'after_heading', 'end_of_section'] = Field(
        default='after_heading',
        description="Position relative to target_heading: 'top' for below the main blog title, 'after_heading' for immediately after heading, or 'end_of_section'"
    )
    prompt: str = Field(description="Detailed visual prompt describing scene, visual metaphors, composition, colors, lighting, art style. Explicitly avoid text/words.")
    alt_text: str = Field(description="Descriptive alt text for accessibility")
    caption: str = Field(description="Insightful 1-sentence caption explaining what this image illustrates")

class VisualPlan(BaseModel):
    art_style: str = Field(description="Unified art direction across all images (e.g. 'Modern minimalist tech vector illustration, clean lines, subtle glowing accents')")
    images: List[ImagePlacement] = Field(min_length=1, max_length=2, description="1 to 2 high-impact images: 1 hero cover image plus at most 1 key conceptual diagram or illustration")

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
    constraints: List[str] = Field(default_factory=list)

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
