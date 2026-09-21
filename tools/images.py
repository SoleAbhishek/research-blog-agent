from pathlib import Path
from typing import Optional, Dict, Any
from google import genai
from models import ImagePlacement

def generate_single_image(
    placement: ImagePlacement,
    art_style: str,
    slug: str,
    output_dir: Path
) -> Optional[Dict[str, Any]]:
    """Generate an image using Google GenAI and save it to disk."""
    output_path = output_dir / f"{slug}_{placement.id}.png"
    try:
        client = genai.Client()
        full_prompt = (
            f"{placement.prompt}. Art style: {art_style}. "
            "High resolution, professional digital illustration, no text, no words, no letters, no watermark."
        )
        resp = client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=full_prompt,
        )
        for part in resp.candidates[0].content.parts:
            if getattr(part, 'inline_data', None):
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(part.inline_data.data)
                return {
                    'placement': placement,
                    'path': output_path.as_posix(),
                }
    except Exception as e:
        print(f"Warning: Failed to generate image '{placement.id}': {e}")
    return None

def insert_image_into_markdown(content: str, placement: ImagePlacement, img_path: str) -> str:
    """Inject markdown image tags and captions into the appropriate place in the markdown draft."""
    lines = content.splitlines()
    inserted = False
    new_lines = []
    img_markdown = f"![{placement.alt_text}]({img_path})\n*{placement.caption}*"

    # Place Hero image right after the title
    if placement.placement == 'top' or placement.target_heading.lower() in ('top', 'hero'):
        for line in lines:
            new_lines.append(line)
            if line.startswith('# ') and not inserted:
                new_lines.append('')
                new_lines.append(img_markdown)
                inserted = True
        if not inserted:
            new_lines.insert(0, img_markdown)
        return '\n'.join(new_lines)

    # Place contextual section images
    target_norm = placement.target_heading.lstrip('#').strip().lower()
    for line in lines:
        new_lines.append(line)
        line_norm = line.lstrip('#').strip().lower()
        if line.startswith('#') and (target_norm in line_norm or line_norm in target_norm) and not inserted:
            new_lines.append('')
            new_lines.append(img_markdown)
            inserted = True

    # Fallback to end of text if heading match was not found
    if not inserted:
        new_lines.append('')
        new_lines.append(img_markdown)

    return '\n'.join(new_lines)
