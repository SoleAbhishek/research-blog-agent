from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from state import State
from models import VisualPlan
from prompts import VISUAL_DIRECTOR_SYSTEM_PROMPT
from tools.images import generate_single_image, insert_image_into_markdown
from config import fast_llm, IMAGES_DIR

def assemble_sections(state: State) -> dict:
    """Sort worker sections by original task index and merge them into an initial markdown draft."""
    plan = state['plan']
    blog_title = plan.blog_title
    raw_sections = state.get('sections', [])

    if raw_sections and isinstance(raw_sections[0], (list, tuple)):
        sorted_sections = [text for _, text in sorted(raw_sections, key=lambda x: x[0])]
    else:
        sorted_sections = [str(s) for s in raw_sections]

    merged_content = f"# {blog_title}\n\n" + "\n\n".join(sorted_sections) + "\n"
    return {'merged_content': merged_content}

def plan_visuals(state: State) -> dict:
    """Evaluate the entire blog draft as an Art Director and generate a cohesive VisualPlan."""
    merged = state.get('merged_content', '')
    plan = state['plan']

    director_llm = fast_llm.with_structured_output(VisualPlan)
    visual_plan = director_llm.invoke([
        SystemMessage(content=VISUAL_DIRECTOR_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Blog Title: {plan.blog_title}\n"
            f"Topic: {state['topic']}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n\n"
            f"Complete Blog Draft:\n{merged}\n\n"
            "Create the visual plan for this blog post."
        ))
    ])
    return {'visual_plan': visual_plan}

def generate_and_insert_images(state: State) -> dict:
    """Generate planned images concurrently, inject markdown image tags, and save the final file."""
    plan = state['plan']
    blog_title = plan.blog_title
    merged_content = state.get('merged_content', '')
    visual_plan: Optional[VisualPlan] = state.get('visual_plan')

    filename = ''.join(
        char if char.isalnum() or char in (' ', '-', '_') else '_'
        for char in blog_title
    ).strip().replace(' ', '_') or 'final_blog'

    final_blog = merged_content

    if visual_plan and visual_plan.images:
        print(f"Generating {len(visual_plan.images)} images concurrently...")
        with ThreadPoolExecutor(max_workers=min(len(visual_plan.images), 3)) as executor:
            futures = [
                executor.submit(generate_single_image, img, visual_plan.art_style, filename, IMAGES_DIR)
                for img in visual_plan.images
            ]
            results = [f.result() for f in futures]

        for res in results:
            if res:
                # Use relative path for portable markdown viewing
                rel_path = f"images/{Path(res['path']).name}"
                final_blog = insert_image_into_markdown(
                    final_blog,
                    res['placement'],
                    rel_path
                )

    Path(f"{filename}.md").write_text(final_blog, encoding="utf-8")
    print(f"Final blog saved to {filename}.md with visual enhancements!")
    return {'final_blog': final_blog}

def create_reducer_subgraph():
    """Build and compile the reducer subgraph: assemble -> plan visuals -> generate & insert images."""
    subgraph = StateGraph(State)
    subgraph.add_node('assemble_sections', assemble_sections)
    subgraph.add_node('plan_visuals', plan_visuals)
    subgraph.add_node('generate_and_insert_images', generate_and_insert_images)

    subgraph.add_edge(START, 'assemble_sections')
    subgraph.add_edge('assemble_sections', 'plan_visuals')
    subgraph.add_edge('plan_visuals', 'generate_and_insert_images')
    subgraph.add_edge('generate_and_insert_images', END)

    return subgraph.compile()
