import os
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from typing import TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Define the state
class AgentState(TypedDict):
    user_message: str
    category: str
    reason: str
    generated_prompt: str
    rejection_message: str
    clarification_question: str
    retrieved_context: str

# 2. Define the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

# 3. Load ChromaDB using your real chroma_db folder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="prompt_library",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Valid categories
VALID_CATEGORIES = [
    "Business", "Student", "Coding", "Marketing",
    "Content Creator", "Educator", "Career", "Creative Design"
]

# 5. Node 1 - Classify
def classify_node(state: AgentState) -> AgentState:
    user_message = state["user_message"]

    prompt = f"""
    You are a strict classifier. Given the user message below, classify it into ONE of these categories:
    Business, Student, Coding, Marketing, Content Creator, Educator, Career, Creative Design

    STRICT RULES:
    - If the message does not fit any category, respond with: Off Topic
    - If the message is missing specific details about what the user actually needs,
      respond with: Vague
    - A message like "help me write something" is Vague because it does not say
      what they are writing, for what purpose, or for what audience.
    - Only classify into a real category if the message has enough detail to confirm it.

    User message: {user_message}

    Respond in this exact format:
    Category: <category>
    Reason: <one sentence reason>
    """

    response = llm.invoke(prompt)
    text = response.content
    lines = text.strip().split("\n")
    category = lines[0].replace("Category:", "").strip()
    reason = lines[1].replace("Reason:", "").strip()

    return {
        "user_message": user_message,
        "category": category,
        "reason": reason,
        "generated_prompt": "",
        "rejection_message": "",
        "clarification_question": "",
        "retrieved_context": ""
    }

# 6. Routing function
def route_after_classify(state: AgentState) -> str:
    category = state["category"]
    if category == "Vague":
        return "clarify"
    if any(valid.lower() in category.lower() for valid in VALID_CATEGORIES):
        return "generate_prompt"
    return "reject"

# 7. Node 2 - RAG Search + Generate Prompt
def generate_prompt_node(state: AgentState) -> AgentState:
    user_message = state["user_message"]
    category = state["category"]

    # Search real ChromaDB
    docs = vectorstore.similarity_search(
        user_message,
        k=2,
        filter={"category_label": category}
    )

    retrieved_context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a prompt generator for MozaicTeck Prompt Library.
    A user needs help with something in the {category} category.
    Their original message was: {user_message}

    Here are relevant prompt templates from the library:
    {retrieved_context}

    Using the templates above as inspiration, generate one powerful and specific
    prompt the user can copy and paste directly into ChatGPT or Claude.
    Respond with just the prompt. Nothing else.
    """

    response = llm.invoke(prompt)
    generated_prompt = response.content.strip()

    return {
        "user_message": user_message,
        "category": category,
        "reason": state["reason"],
        "generated_prompt": generated_prompt,
        "rejection_message": "",
        "clarification_question": "",
        "retrieved_context": retrieved_context
    }

# 8. Node 3 - Reject
def reject_node(state: AgentState) -> AgentState:
    return {
        "user_message": state["user_message"],
        "category": state["category"],
        "reason": state["reason"],
        "generated_prompt": "",
        "rejection_message": "I am sorry, that topic is outside what I currently cover. MozaicTeck specializes in AI prompts for Business, Marketing, Coding, Career, Students, Educators, Content Creators and Creative Design. Try asking something like: Give me a prompt for a graphic designer.",
        "clarification_question": "",
        "retrieved_context": ""
    }

# 9. Node 4 - Clarify
def clarify_node(state: AgentState) -> AgentState:
    user_message = state["user_message"]

    prompt = f"""
    A user sent this message to MozaicTeck Prompt Library: "{user_message}"

    The message is too vague to understand what they need.
    Ask them ONE simple and warm question to understand what area they need help with.

    MozaicTeck covers: Business, Marketing, Coding, Career, Students,
    Educators, Content Creators and Creative Design.

    Respond with just the question. Nothing else.
    """

    response = llm.invoke(prompt)
    clarification_question = response.content.strip()

    return {
        "user_message": user_message,
        "category": state["category"],
        "reason": state["reason"],
        "generated_prompt": "",
        "rejection_message": "",
        "clarification_question": clarification_question,
        "retrieved_context": ""
    }

# 10. Build the graph
graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("generate_prompt", generate_prompt_node)
graph.add_node("reject", reject_node)
graph.add_node("clarify", clarify_node)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_after_classify)
graph.add_edge("generate_prompt", END)
graph.add_edge("reject", END)
graph.add_edge("clarify", END)

# 11. Compile the agent
agent_app = graph.compile()