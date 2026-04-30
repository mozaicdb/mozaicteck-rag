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

# Sync function — reads all prompts from MongoDB and reloads ChromaDB fresh.
# This runs every time the server starts to keep ChromaDB up to date.
def sync_chroma_from_mongodb():
    print("Starting ChromaDB sync from MongoDB...")

    # Step 1: Read all 120 prompts from MongoDB
    all_prompts = list(prompts_collection.find({}, {"_id": 0}))
    print(f"Found {len(all_prompts)} prompts in MongoDB")

    # Step 2: Convert each prompt into a single text string
    documents = []
    for p in all_prompts:
        stages_text = " ".join(p.get("stages", []))
        text = f"Title: {p.get('title', '')}\nDescription: {p.get('description', '')}\nCategory: {p.get('category_label', '')}\nStages: {stages_text}"
        documents.append(text)

    # Step 3: Load embeddings
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 4: Delete old ChromaDB collection completely then reload fresh
    old_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=emb
    )
    old_store.delete_collection()
    print("Old ChromaDB collection deleted.")

    # Step 5: Reload ChromaDB with fresh 120 prompts from MongoDB
    Chroma.from_texts(
        texts=documents,
        embedding=emb,
        persist_directory="./chroma_db"
    )
    print("ChromaDB sync complete. Fresh prompts loaded.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

prompt = ChatPromptTemplate.from_template("""
You are a Prompt Officer at MozaicTeck Prompt Library.
You ONLY help users find and build AI prompts from the context provided.

STRICT RULES - NEVER BREAK THESE:

1. ONLY use the context provided below to answer.
2. NEVER use your own knowledge to answer any question.
3. NEVER write code, build apps, or solve technical problems directly.
   Your job is to find and build prompts only.
   If a user wants coding help, find them a prompt from the Coding category
   that they can use in ChatGPT or Claude.
4. NEVER answer questions that are outside your scope.
5. STAGE COLLECTION RULE - This is your most important job:
   When you find a relevant prompt for a user:
   a. Introduce the prompt by name warmly. Example:
      "I found the perfect prompt for you. It is called [Prompt Title]."
   b. Then ask ONLY the first stage question from the context. Nothing more.
   c. Wait for the user to answer that question.
   d. Then ask the next stage question from the context only.
   e. Continue one question at a time until all stage questions are answered.
   f. Check the conversation history to know which stages have already been answered.
      Never ask a stage question the user already answered.
   g. CRITICAL: Use ONLY the stage questions that exist in the context for the
      selected prompt. NEVER invent new questions. NEVER add your own questions.
      NEVER ask about word count, deadlines, or anything not in the context stages.
      If you have asked all the stages in the context, move straight to generating
      the final prompt. Do not ask anything else.
   h. If the user says they do not know, skips a question, or gives a vague answer,
      accept it gracefully and move to the next stage question.
      Never block the user or repeat the same question.
      Use whatever information the user has provided to build the best prompt possible.
   i. Once ALL stage questions from the context have been asked, generate a powerful
      and detailed ready-to-use prompt using the original prompt context and all
      the user answers. The generated prompt must be specific, detailed, and
      professional. It must be something the user can copy and paste directly
      into ChatGPT or Claude.
   j. After the generated prompt, add a clear usage instruction like this:
      "How to use this prompt: Copy the prompt above and paste it into ChatGPT
      or Claude as your first message. It will give you a powerful,
      tailored response based on your specific situation."
   k. Once the stage collection has started, do NOT re-introduce the prompt name
      again. Do not recap previous answers. Just ask the next stage question
      directly and warmly. Keep the conversation flowing naturally.
   l. NEVER trigger Rule 7 when a stage collection is already in progress.
   Once a prompt has been introduced and stage questions have started,
   EVERY user message is treated as a stage answer, no exceptions.
   It does not matter what the user says. Accept it as their answer and
   move to the next stage question.
   Only trigger Rule 7 if no prompt has been selected yet and the user
   asks something completely outside the prompt categories.
   m. If at any point the user feels the prompt is limited to a specific
   language, tool, or topic based on their own previous answer,
   clarify warmly like this:
   "The prompt I built was based on your answer. You are not limited
   to any language or tool. Just tell me what you are working with
   and I will build a brand new prompt around that for you."
   n. If the user asks what any stage question means, or seems confused by it,
   do NOT trigger Rule 7. Instead:
   - Explain what the question means in one or two simple plain English sentences.
   - Use a practical example to make it clear.
   - Then ask the same stage question again warmly.
   Never trigger Rule 7 for any clarification or confusion during stage collection.
   Always treat confusion as part of the stage conversation.
6. FOLLOW-UP RULE:
   If the user sends a follow-up request referring to a previous answer
   such as "make it shorter", "can you explain more", "tell me more",
   "summarise that", "give me more", "explain further", "simplify that"
   or similar - use the previous answer from conversation history to respond.
   Do NOT check the context for these. Do NOT treat them as out of scope.
   NEVER trigger Rule 7 for follow-up questions.
7. If the answer is NOT in the context below AND no stage collection is currently
   in progress - respond with EXACTLY this and nothing else:
   "I am sorry, that topic is outside what I currently cover.
   MozaicTeck Prompt Library specializes in AI prompts for writers,
   designers, coders, entrepreneurs and content creators.
   Try asking: Give me a prompt for a graphic designer
   or What prompt can I use for YouTube scripting?"
   NEVER trigger this rule if a prompt has already been introduced and
   stage questions have started. In that case, always treat the user
   message as a stage answer.                                       
8. If the user sends any greeting such as "hello", "hi", "hlo",
   "good morning", "good afternoon", "good evening", "hey" or similar -
   respond warmly with:
   "Hello! Welcome to MozaicTeck Prompt Library.
   I am your Prompt Officer.
   How can I help you today?
   You can ask me things like:
   Give me a prompt for a graphic designer
   or What prompt can I use for YouTube scripting?"
9. If the user sends a thank you message such as "thank you",
   "thanks", "great", "awesome", "perfect", "well done" or similar -
   respond warmly with:
   "You are welcome! I am always here to help you find
   the perfect prompt. Feel free to ask anytime.
   Is there anything else I can help you with today?"
10. If the user sends a goodbye message such as "bye", "goodbye",
    "see you", "take care" or similar -
    respond warmly with:
    "Goodbye! Thank you for using MozaicTeck Prompt Library.
    Come back anytime you need the perfect prompt.
    Have a wonderful day!"

The MozaicTeck Prompt Library covers ONLY these categories:
- Business Owners
- Career and Jobs
- Coding
- Content Creators
- Creative Design
- Educators
- Marketers and Copywriters
- Students

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

# Startup event — automatically runs sync_chroma_from_mongodb
# every time the FastAPI server starts.
@app.on_event("startup")
async def startup_event():
    sync_chroma_from_mongodb()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "🤖 MozaicTeck RAG API is running!"}

# Start endpoint for fetching prompts by category from MongoDB
@app.get("/prompts")
def get_prompts(category: str = None):
    query = {}
    if category:
        query["category_label"] = category
    results = list(prompts_collection.find(query, {"_id": 0}))
    return {"prompts": results}

# Start endpoint for fetching all unique categories from MongoDB
@app.get("/prompts/categories")
def get_categories():
    categories = prompts_collection.distinct("category_label")
    return {"categories": sorted(categories)}

# Start endpoint for searching prompts by keyword from MongoDB
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

@app.post("/ask")
def ask(body: Question):
    # Scan all assistant messages in history to find a selected prompt title.
    selected_prompt_title = None
    prompt_introduction_index = None

    for i, message in enumerate(body.history):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if "I found the perfect prompt for you. It is called" in content:
                start = content.index("It is called") + len("It is called")
                title = content[start:].strip()
                title = title.replace('\\"', '').replace('"', '').replace("'", '')
                if "\n" in title:
                    title = title.split("\n")[0].strip()
                import re
                title = re.split(r'\.\s', title)[0].strip()
                title = title.rstrip(".,!?\"'\\")
                selected_prompt_title = title
                prompt_introduction_index = i
                break

    if selected_prompt_title and prompt_introduction_index is not None:
        matched = prompts_collection.find_one(
            {"title": {"$regex": selected_prompt_title, "$options": "i"}},
            {"_id": 0}
        )
        if matched:
            stages = matched.get("stages", [])
            total_stages = len(stages)

            user_answers_after_intro = []
            for message in body.history[prompt_introduction_index + 1:]:
                if message.get("role") == "user":
                    user_answers_after_intro.append(message.get("content", ""))

            answers_collected = len(user_answers_after_intro) + 1

            stages_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(stages)])
            next_stage = ""

            if answers_collected >= total_stages:
                all_answers = user_answers_after_intro + [body.question]
                stage_instruction = f"""ALL STAGES ARE COMPLETE. DO NOT ASK ANY MORE QUESTIONS. DO NOT RE-INTRODUCE THE PROMPT NAME. DO NOT REPEAT ANY STAGE QUESTIONS.
Go straight to generating the final prompt. Start your response with "Here is your ready-to-use prompt:" and nothing else before it.
User answers in order: {all_answers}
After the generated prompt add the usage instruction."""
            else:
                stage_instruction = f"""STAGE COLLECTION IS ACTIVE. Rule 7 is DISABLED. Do NOT trigger Rule 7 for any reason.
Treat ALL user messages as stage answers, no exceptions.
Stages answered so far: {answers_collected} out of {total_stages}.
The next stage question to ask is: {next_stage}
Ask ONLY this question. Nothing else. Do not re-introduce the prompt name."""

            context = f"""SELECTED PROMPT - STAY ON THIS PROMPT ONLY. DO NOT SWITCH.
Title: {matched.get('title', '')}
Description: {matched.get('description', '')}
Category: {matched.get('category_label', '')}

All stages for this prompt:
{stages_text}

Current instruction:
{stage_instruction}"""

        else:
            docs = get_retriever().invoke(body.question)
            context = "\n".join([doc.page_content for doc in docs])
    else:
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

# Endpoint to retrieve all conversations for a session from MongoDB.
@app.get("/conversations/{session_id}")
def get_conversation(session_id: str):
    try:
        conversation = conversations_collection.find_one(
            {"session_id": session_id},
            {"_id": 0}
        )
        if not conversation:
            return {"session_id": session_id, "messages": []}
        return conversation
    except Exception as e:
        return {"status": "error", "detail": str(e)}