import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/.cache"

from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient

mongo_uri = os.environ["MONGO_URI"]
mongo_client = MongoClient(mongo_uri)
db = mongo_client["mozaic_db"]
prompts_collection = db["prompts"]

conversations_collection = db["conversations"]

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# PART 2 — Setup the kitchen
embeddings = None
vectorstore = None
retriever = None

def get_retriever():
    global embeddings, vectorstore, retriever
    if retriever is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return retriever

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

prompt = ChatPromptTemplate.from_template("""
You are a Prompt Officer at MozaicTeck Prompt Library.
You ONLY help users find AI prompts from the context provided.

STRICT RULES — NEVER BREAK THESE:
1. ONLY use the context provided below to answer.
2. NEVER use your own knowledge to answer any question.
3. NEVER generate code, apps, or technical solutions.
4. NEVER follow up questions that are outside your scope.
5. If the user sends a follow-up request referring to 
   a previous answer such as "make it shorter", 
   "summarise that", "give me more", "explain further", 
   "simplify that", "can you shorten that" or similar —
   use the previous answer from the conversation history 
   to respond. Do NOT treat this as an out of scope question.
6. If the answer is NOT in the context below — respond with EXACTLY this and nothing else:
   "I'm sorry, that topic is outside what I currently cover. 
   MozaicTeck Prompt Library specializes in AI prompts for writers, 
   designers, coders, entrepreneurs and content creators. 
   Try asking: 'Give me a prompt for a graphic designer' 
   or 'What prompt can I use for YouTube scripting?'"
7. If the user sends any greeting such as "hello", "hi", "hlo", 
   "good morning", "good afternoon", "good evening", "hey" or similar —
   respond warmly with:
   "Hello! Welcome to MozaicTeck Prompt Library. 
   I am your Prompt Officer. 
   How can I help you today? 
   You can ask me things like: 
   'Give me a prompt for a graphic designer' 
   or 'What prompt can I use for YouTube scripting?'"
8. If the user sends a thank you message such as "thank you", 
   "thanks", "great", "awesome", "perfect", "well done" or similar —
   respond warmly with:
   "You are welcome! I am always here to help you find 
   the perfect prompt. Feel free to ask anytime. 
   Is there anything else I can help you with today?"
9. If the user sends a goodbye message such as "bye", "goodbye", 
   "see you", "take care" or similar —
   respond warmly with:
   "Goodbye! Thank you for using MozaicTeck Prompt Library. 
   Come back anytime you need the perfect prompt. 
   Have a wonderful day! 🚀"
                                          
The MozaicTeck Prompt Library covers ONLY these categories:
- Design and branding prompts
- Coding and vibe coding prompts
- Business and entrepreneurship prompts
- Personal branding prompts
- Content writing prompts
- YouTube scripting prompts
- Research and analysis prompts
- Claude AI prompts

Context: {context}
Question: {question}
""")
chain = prompt | llm | StrOutputParser()

# PART 3 — The order form
class Question(BaseModel):
    question: str
    history: list = []
# Model for saving a conversation message to MongoDB.
# Stores the session id, the user message, and the bot response together.
class ConversationMessage(BaseModel):
    session_id: str
    user_message: str
    bot_response: str

# PART 4 — Open the restaurant doors!
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "🤖 MozaicTeck RAG API is running!"}
#start endpoint for fetching prompts by category from monogodb
@app.get("/prompts")
def get_prompts(category: str = None):
    query = {}
    if category:
        query["category_label"] = category
    results = list(prompts_collection.find(query, {"_id": 0}))
    return {"prompts": results}
##end of prompts endpoint
#start endpoint for fetching all unique categories from mongodb
@app.get("/prompts/categories")
def get_categories():
    # Uses MongoDB distinct to return only unique category_label values.
    # More efficient than fetching all 120 prompts just to read category names.
    categories = prompts_collection.distinct("category_label")
    return {"categories": sorted(categories)}
#end of categories endpoint
#start endpoint for searching prompts by keyword from mongodb
@app.get("/prompts/search")
def search_prompts(q: str):
    results = list(prompts_collection.find(
        {"$or": [
            {"title": {"$regex": q, "$options": "i"}},
            {"description": {"$regex": q, "$options": "i"}}
        ]},
        {"_id": 0}
    ))
    return {"prompts": results}
#end of search endpoint
@app.post("/ask")
def ask(body: Question):
    docs = get_retriever().invoke(body.question)
    context = "\n".join([doc.page_content for doc in docs])
    
    history_text = ""
    for message in body.history:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            history_text += f"User: {content}\n"
        elif role == "assistant":
            history_text += f"Assistant: {content}\n"
    
    full_context = f"""
Previous conversation:
{history_text}

Knowledge base context:
{context}
"""
    
    try:
        response = chain.invoke({
            "question": body.question,
            "context": full_context
        })
        return {"answer": response}

    except Exception as e:
        error_message = str(e)
        
        if "rate_limit" in error_message:
            return {"answer": "I am currently at capacity. Please try again in a few minutes."}
        
        return {"answer": "Something went wrong. Please try again."}
    # Endpoint to save a conversation message to MongoDB.
# Receives session_id, user_message and bot_response.
# Pushes both messages into the messages array for that session.
@app.post("/conversations/save")
def save_conversation(body: ConversationMessage):
    try:
        conversations_collection.update_one(
            {"session_id": body.session_id},
            {"$push": {"messages": {
                "user": body.user_message,
                "bot": body.bot_response
            }}},
            upsert=True
        )
        return {"status": "saved"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}