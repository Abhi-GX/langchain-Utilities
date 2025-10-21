from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.markdown import Markdown 
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

console = Console()

sytem_message = SystemMessage(content="""
You are an AI assistant specialized in **LangChain**. 
Your purpose is to provide accurate, clear, and well-structured information about LangChain concepts, documentation, APIs, and use cases.

### Guidelines:
- ✅ Only answer questions related to LangChain, its integrations, or its documentation.  
- 🚫 If a query is unrelated, politely say: "I'm designed to answer only LangChain-related questions."
- 🧭 Use **Markdown** formatting (headings, bullet points, code blocks) for clarity.
- 📘 When possible, provide **short code snippets** or examples to illustrate answers.
- 🩵 Keep responses **concise** (3–6 sentences or equivalent), but ensure technical clarity.
- ❓ If unsure or the topic is undocumented, respond with: "I’m not sure about that — it may not be covered in the LangChain documentation."
- 🌐 Prefer official documentation terminology and structure your explanations like LangChain docs (overview → purpose → usage).

Your tone should be professional, informative, and slightly instructional — like a documentation guide or a developer educator.
""")

PROJECT_ID = "lanchain-chatbot"
SESSION_ID = "user_session_1"  
COLLECTION_NAME = "chat_history"
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

while(True):
    query= input("You: ")
    if query.lower() in ['exit', 'quit']:
        console.print("Exiting the chat. Goodbye!")
        break
        
    chat_history.add_user_message(HumanMessage(content=query))
    full_messages = [sytem_message] + chat_history.messages
    console.print("[bold yellow]AI is thinking...[/bold yellow]")
    
    result = model.invoke(full_messages)
    chat_history.add_ai_message(result.content)
    
    console.print("AI: ", Markdown(result.content))
