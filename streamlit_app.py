import streamlit as st
import os
import sys
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessageChunk

# Import from our task_maistro module
import task_maistro
import configuration

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="task_mAIstro | AI Task Checker",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; }
    .sidebar .stButton button { width: 100%; }
    .todo-card { background-color: #f0f2f6; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
    .streaming-cursor { animation: blink 1s step-end infinite; }
    @keyframes blink { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

st.title("✅ task_mAIstro")
st.caption("AI-powered task management assistant with persistent long-term memory (Profile • ToDos • Instructions) backed by PostgreSQL — now with streaming responses!")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    user_id = st.text_input(
        "User ID", 
        value=st.session_state.get("user_id", "default-user"),
        help="Change this to switch between different users/profiles."
    )
    st.session_state.user_id = user_id
    
    todo_category = st.text_input(
        "Todo Category", 
        value=st.session_state.get("todo_category", "general"),
        help="e.g. work, personal, fitness, projects"
    )
    st.session_state.todo_category = todo_category
    
    task_maistro_role = st.text_area(
        "System Role / Personality",
        value=st.session_state.get("task_maistro_role", configuration.Configuration.task_maistro_role),
        height=80,
        help="Customize how the AI behaves"
    )
    st.session_state.task_maistro_role = task_maistro_role
    
    st.divider()
    
    st.subheader("🔐 Secrets & Database")
    
    secrets = {}
    if hasattr(st, "secrets"):
        try:
            secrets = dict(st.secrets)
        except:
            pass
    
    default_postgres = os.getenv("POSTGRES_URI") or secrets.get("POSTGRES_URI", "")
    default_openai = os.getenv("OPENAI_API_KEY") or secrets.get("OPENAI_API_KEY", "")
    
    postgres_uri = st.text_input(
        "PostgreSQL Connection String",
        value=default_postgres,
        type="password",
        placeholder="postgresql://user:password@host:5432/dbname?sslmode=require",
        help="Use Supabase, Neon, Railway, or any Postgres. Required for persistence."
    )
    
    openai_api_key = st.text_input(
        "OpenAI / OpenRouter API Key",
        value=default_openai,
        type="password",
        help="Required for the model"
    )
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENROUTER_API_KEY"] = openai_api_key
    if postgres_uri:
        os.environ["POSTGRES_URI"] = postgres_uri
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Apply & Reconnect DB", use_container_width=True, type="primary"):
            st.cache_resource.clear()
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()
    with col2:
        if st.button("🧹 Clear Chat UI", use_container_width=True):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    st.caption("💡 Tip: Add `POSTGRES_URI` and `OPENAI_API_KEY` to `.streamlit/secrets.toml` for Streamlit Cloud.")
    
    st.divider()
    
    st.subheader("🧠 Current Long-Term Memory")
    
    if st.button("🔄 Refresh Memories from DB", use_container_width=True):
        st.rerun()
    
    memory_placeholder = st.empty()

# ============================================
# PERSISTENCE (using ConnectionPool + create_graph)
# ============================================

@st.cache_resource(show_spinner="Connecting to PostgreSQL and setting up tables...")
def get_persisted_graph_and_store(postgres_uri: str):
    """Create Postgres checkpointer + store using ConnectionPool (recommended for Streamlit)."""
    if not postgres_uri:
        st.error("❌ PostgreSQL URI is required. Please provide it in the sidebar.")
        st.stop()
    
    try:
        pool = ConnectionPool(
            conninfo=postgres_uri,
            max_size=20,
            kwargs={"autocommit": True}
        )
        
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool)
        
        # Create required tables (idempotent)
        checkpointer.setup()
        store.setup()
        
        # Compile graph with the provided persistence
        graph = task_maistro.create_graph(checkpointer=checkpointer, store=store)
        
        return graph, checkpointer, store
    except Exception as e:
        st.error(f"Failed to initialize Postgres persistence: {e}")
        st.info("Make sure `psycopg_pool` and `langgraph-checkpoint-postgres` are installed, and your Postgres connection string is correct.")
        st.stop()

if not postgres_uri:
    st.info("👈 Please enter your **PostgreSQL connection string** in the sidebar to continue. Free options: Supabase, Neon.tech")
    st.stop()

graph, checkpointer, store = get_persisted_graph_and_store(postgres_uri)

# Thread ID
thread_id = f"{user_id}__{todo_category}"

config = {
    "configurable": {
        "thread_id": thread_id,
        "user_id": user_id,
        "todo_category": todo_category,
        "task_maistro_role": task_maistro_role,
    }
}

# ============================================
# DISPLAY MEMORIES
# ============================================

def display_memories(store, user_id, todo_category):
    with memory_placeholder.container():
        # Profile
        profile_ns = ("profile", todo_category, user_id)
        profiles = store.search(profile_ns)
        profile = profiles[0].value if profiles else None
        
        with st.expander("👤 User Profile", expanded=bool(profile)):
            if profile:
                st.json(profile, expanded=False)
            else:
                st.caption("No profile information yet. Mention details in chat.")
        
        # ToDos
        todo_ns = ("todo", todo_category, user_id)
        todos = store.search(todo_ns)
        
        with st.expander(f"📋 To-Do List ({len(todos)} items)", expanded=True):
            if todos:
                for item in todos:
                    todo = item.value
                    status = todo.get("status", "not started")
                    emoji_map = {
                        "not started": "⬜",
                        "in progress": "🔄",
                        "done": "✅",
                        "archived": "📦"
                    }
                    emoji = emoji_map.get(status, "⬜")
                    
                    st.markdown(f"""
                    <div class="todo-card">
                        <strong>{emoji} {todo.get('task', 'Untitled task')}</strong><br>
                        <small>⏱️ Est. {todo.get('time_to_complete', '?')} min 
                        | 📅 {todo.get('deadline') or 'No deadline'}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if todo.get("solutions"):
                        with st.popover("💡 Suggested solutions"):
                            for sol in todo["solutions"]:
                                st.write(f"• {sol}")
            else:
                st.caption("No tasks yet. Ask the AI to add some!")
        
        # Instructions
        instr_ns = ("instructions", todo_category, user_id)
        instr_item = store.get(instr_ns, "user_instructions")
        instructions = instr_item.value.get("memory", "") if instr_item else ""
        
        with st.expander("⚙️ Custom Update Instructions", expanded=False):
            if instructions:
                st.text(instructions)
            else:
                st.caption("No custom instructions yet.")
        
        # Clear button
        if st.button("🗑️ Clear ALL memories for this user + category", type="secondary", use_container_width=True):
            try:
                for ns in [profile_ns, todo_ns, instr_ns]:
                    items = store.search(ns)
                    for it in items:
                        store.delete(ns, it.key)
                st.success("Memories cleared from PostgreSQL!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear memories: {e}")

display_memories(store, user_id, todo_category)

# ============================================
# CHAT INTERFACE WITH TOKEN STREAMING
# ============================================

st.subheader(f"💬 Conversation  •  Thread: `{thread_id}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    role = "assistant" if msg["role"] in ("assistant", "ai") else "user"
    with st.chat_message(role):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Tell me about your tasks or preferences... e.g. 'Add buy groceries by Friday' or 'My name is Alex and I work as a designer in Manila'"):
    
    # Add user message to UI and state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    # Stream the assistant response token by token
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("task_mAIstro is thinking..."):
                # Use stream_mode="messages" to get token chunks from the LLM
                for event in graph.stream(inputs, config=config, stream_mode="messages"):
                    chunk, metadata = event
                    
                    # AIMessageChunk or similar has .content
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, list):
                            content = "".join(str(c) for c in content if isinstance(c, str))
                        
                        full_response += str(content)
                        # Live update with cursor
                        message_placeholder.markdown(full_response + "▌")
            
            # Final render without cursor
            message_placeholder.markdown(full_response)
            
            # Save to session history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Refresh memories sidebar after possible updates
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during streaming: {e}")
            st.exception(e)
            # Clean up the last user message if something failed
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

# ============================================
# FOOTER
# ============================================
st.divider()

with st.expander("🚀 Deployment & Notes"):
    st.markdown("""
    **Streaming is now enabled!** The assistant response appears token-by-token for a much better UX.
    
    **Requirements** (already in `requirements.txt`):
    - `langgraph`, `langgraph-checkpoint-postgres`, `psycopg[binary]`, `psycopg_pool`
    - `streamlit>=1.35`, `langchain-openai`, etc.
    
    **For Streamlit Cloud:**
    1. Point the app to `streamlit_app.py`
    2. Add secrets:
       ```toml
       OPENAI_API_KEY = "sk-..."
       POSTGRES_URI = "postgresql://..."

Use a managed Postgres (Supabase / Neon free tier recommended).

The graph now uses ConnectionPool + create_graph from task_maistro.py.
""")
st.caption("Built with LangGraph + PostgreSQL + Streamlit • Token streaming enabled • Persistent memory across sessions")