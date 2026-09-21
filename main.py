import sys
from graph import app

DEFAULT_TOPIC = "Newest breakthroughs in AI research and their implications for the future of technology"

def run_agent(topic: str):
    """Run the research blog agent with live node progress streaming."""
    print("\n" + "=" * 70)
    print(f"🚀 Starting Research Blog Agent")
    print(f"📌 Topic: {topic}")
    print("=" * 70 + "\n")

    for event in app.stream({"topic": topic}, stream_mode="updates"):
        for node_name, state_update in event.items():
            if node_name == "router":
                mode = state_update.get("mode")
                queries = state_update.get("queries", [])
                print(f"🚦 [Router] Selected Mode: '{mode}' | Generated {len(queries)} search queries.")
            elif node_name == "research":
                evidence = state_update.get("evidence", [])
                print(f"🔍 [Research] Synthesized and deduplicated {len(evidence)} evidence sources.")
            elif node_name == "orchestrator":
                plan = state_update.get("plan")
                if plan:
                    print(f"📋 [Planner] Created outline: '{plan.blog_title}' with {len(plan.tasks)} sections.")
            elif node_name == "worker":
                sections = state_update.get("sections", [])
                if sections:
                    idx = sections[0][0] + 1
                    print(f"✍️  [Worker] Finished drafting section #{idx}.")
            elif node_name == "reducer":
                print(f"🎨 [Reducer] Blog assembled, visuals generated, and markdown saved!")

    print("\n" + "=" * 70)
    print("✅ Blog generation complete! Check the current directory for the generated .md file.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    run_agent(topic)
