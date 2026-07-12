import streamlit as st
import os
import sys
from datetime import datetime
from langchain_core.messages import HumanMessage

# Import from our updated task_maistro module (provides builder and create_graph)
import task_maistro
import configuration

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

st.set_page_config(
    page_title="task_mAIstro | AI Task Checker",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for nicer look
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; }
    .sidebar .stButton button { width: 100%; }
    .todo-card { background-color: #f0f2f6; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("✅ task_mAIstro")
st.caption("AI-powered task management assistant with persistent long-term memory (Profile • ToDos • Instructions) backed by PostgreSQL")

# ============================================
# SIDEBAR - Configuration & Memories
# ============================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # User & Category
    user_id = st.text_input(
        "User ID", 
        value=st.session_state.get("user_id", "default-user"),
        help="Change this to switch between different users/profiles. Memories are isolated per user+category."
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
    
    # Persistence & API Keys
    st.subheader("🔐 Secrets & Database")
    
    # Load from Streamlit secrets or env (for Cloud deploy)
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
        placeholder="postgresql://user:password@host:5432/dbname?sslmode=disable",
        help="Use Supabase, Neon, Railway, or any Postgres. Required for persistence across sessions."
    )
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=default_openai,
        type="password",
        help="Required for GPT-4o calls"
    )
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if postgres_uri:
        os.environ["POSTGRES_URI"] = postgres_uri
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Apply & Reconnect DB", use_container_width=True, type="primary"):
            # Clear caches so new connection is used
            st.cache_resource.clear()
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()
    with col2:
        if st.button("🧹 Clear Chat UI", use_container_width=True):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    st.caption("💡 Tip: Add these to `.streamlit/secrets.toml` for Streamlit Cloud deploy.")
    
    st.divider()
    
    # Current Memories Viewer
    st.subheader("🧠 Current Long-Term Memory")
    
    if st.button("🔄 Refresh Memories from DB", use_container_width=True):
        st.rerun()
    
    # We will populate this after getting store
    memory_placeholder = st.empty()

# ============================================
# MAIN - Get Graph & Store (cached) - UPDATED WITH ConnectionPool
# ============================================

@st.cache_resource(show_spinner="Connecting to PostgreSQL and setting up tables...")
def get_persisted_graph_and_store(postgres_uri: str):
    """Create Postgres checkpointer + store using ConnectionPool (recommended for Streamlit)."""
    if not postgres_uri:
        st.error("❌ PostgreSQL URI is required. Please provide it in the sidebar.")
        st.stop()
    
    try:
        # Create a connection pool (best practice for long-running apps like Streamlit)
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
        
        # Compile graph with persistence
        graph = task_maistro.builder.compile(checkpointer=checkpointer, store=store)
        
        return graph, checkpointer, store
    except Exception as e:
        st.error(f"Failed to initialize Postgres persistence: {e}")
        st.info("Make sure `psycopg_pool` is installed (`pip install psycopg_pool`) and your Postgres connection string is correct. Also verify your database is running and accessible.")
        st.stop()

if not postgres_uri:
    st.info("👈 Please enter your **PostgreSQL connection string** in the sidebar to continue. Free options: Supabase, Neon.tech")
    st.stop()

graph, checkpointer, store = get_persisted_graph_and_store(postgres_uri)

# Thread ID for conversation history (short-term memory via checkpointer)
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
# HELPER: Display & Manage Memories
# ============================================

def display_memories(store, user_id, todo_category):
    """Fetch and nicely display current long-term memories."""
    with memory_placeholder.container():
        # Profile
        profile_ns = ("profile", todo_category, user_id)
        profiles = store.search(profile_ns)
        profile = profiles[0].value if profiles else None
        
        with st.expander("👤 User Profile", expanded=bool(profile)):
            if profile:
                st.json(profile, expanded=False)
            else:
                st.caption("No profile information yet. Mention details in chat (name, location, job, interests...)")
        
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
                st.caption("No custom instructions yet. The AI will learn your preferences from feedback.")
        
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

# Show memories in sidebar
display_memories(store, user_id, todo_category)

# ============================================
# CHAT INTERFACE
# ============================================

st.subheader(f"💬 Conversation  •  Thread: `{thread_id}`")

# Session state for chat UI (short-term display; full history persisted in Postgres via checkpointer)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    role = "assistant" if msg["role"] in ("assistant", "ai") else "user"
    with st.chat_message(role):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Tell me about your tasks or preferences...  e.g. 'Add buy groceries by Friday' or 'My name is Alex and I work as a designer in Manila'"):
    
    # Add to UI history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare input for graph (new human message only; checkpointer loads previous state)
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    try:
        with st.spinner("task_mAIstro is thinking and updating memories..."):
            result = graph.invoke(inputs, config=config)
        
        # Extract assistant's final response
        last_msg = result["messages"][-1]
        assistant_reply = getattr(last_msg, "content", str(last_msg))
        
        # Add to UI
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        
        # Refresh memories sidebar after possible updates
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during graph invocation: {e}")
        st.exception(e)
        # Remove the user message from history on failure? optional
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()

# ============================================
# FOOTER & DEPLOY INSTRUCTIONS
# ============================================
st.divider()

with st.expander("🚀 How to Deploy on Streamlit Community Cloud"):
    st.markdown("""
    1. **Fork or copy** this repo to your GitHub.
    2. Create a new Streamlit app at [share.streamlit.io](https://share.streamlit.io) pointing to `streamlit_app.py`.
    3. In the app settings, add **Secrets** (or use `.streamlit/secrets.toml`):
       ```toml
       OPENAI_API_KEY = "sk-..."
       POSTGRES_URI = "postgresql://user:pass@db.supabase.co:5432/postgres?sslmode=require"
       (Recommended) Use a managed Postgres:
Supabase (free tier)
Neon (free tier)
Railway / Render / Fly.io Postgres

Requirements are in requirements.txt — Streamlit Cloud will install them automatically.

Don't forget to add psycopg_pool to your requirements.txt!
""")