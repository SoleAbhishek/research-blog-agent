from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Project Directories
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
CACHE_DIR = BASE_DIR / ".cache"

IMAGES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# --- MODEL TIERING ---
# Tier 1: The Architect (High reasoning for planning and structural breakdown)
planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

# Tier 2: The Creative Specialist (Expressive, nuanced language modeling for section drafting)
writer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Tier 3: The Rapid Operator (Fast, deterministic extraction, routing, and visual directing)
fast_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
