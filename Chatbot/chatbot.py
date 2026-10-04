from aup_config import aup_setup
aup_setup()


# Import necessary libraries

from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import display
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# start by creating our State class that is used to build the StateGraph object.
# The StateGraph defines the structure of our chatbot as a set of nodes and edges.
# We add nodes to represent the LLM and functions/tools our chatbot can use, and edges to specify the transitions between nodes.

class State(TypedDict):
    """
    Messages have the type 'list'.
    The `add_messages` function defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Use model_provider='openai' so you can use a different LLM serving framework if needed.

llm = init_chat_model(
    model='qwen3.5:9b',
    model_provider='openai',
    base_url='http://localhost:11434/v1/',
    api_key='abc-123'
)

# Now, we can add a node to the graph. Nodes represent units of work.
# Let us add the LLM call into a simple node
# Note how the chatbot function takes the State as input and returns a dictionary containing an updated messages list.
# The .add_node takes at least two arguments
# Unique node name
# Object that will be called when the node is used.

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)