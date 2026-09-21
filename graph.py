from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    router,
    route_next,
    research_node,
    create_plan,
    fanout,
    worker,
    create_reducer_subgraph,
)

def build_graph():
    """Assemble and compile the research blog agent StateGraph."""
    reducer_subgraph = create_reducer_subgraph()

    graph = StateGraph(State)
    graph.add_node('router', router)
    graph.add_node('research', research_node)
    graph.add_node('orchestrator', create_plan)
    graph.add_node('worker', worker)
    graph.add_node('reducer', reducer_subgraph)

    graph.add_edge(START, 'router')
    graph.add_conditional_edges('router', route_next, ['research', 'orchestrator'])
    graph.add_edge('research', 'orchestrator')
    graph.add_conditional_edges('orchestrator', fanout, ['worker'])
    graph.add_edge('worker', 'reducer')
    graph.add_edge('reducer', END)

    return graph.compile()

app = build_graph()
