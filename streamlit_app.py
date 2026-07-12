import streamlit as st
import os
from datetime import datetime
from langchain_core.messages import HumanMessage

import task_maistro
import configuration

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv

load_dotenv()

# ============================================
# PAGE CONFIG + GROK-LIKE STYLING
# ============================================
st.set_page_config(
    page_title="task_mAIstro",
    page_icon="✅",
    layout="centered",   # More focused chat feel like Grok
    initial_sidebar_state="expanded"
)

# Grok-inspired clean styling
st.markdown("""
<style>
    /* Overall app */
    .stApp {
        background-color: #0f0f0f;
        color: #e5e5e5;
    }
    
    /* Message bubbles - Grok style */
    .stChatMessage {
        padding: 1rem 1.25rem;
        border-radius: 18px;
        margin-bottom: 12px;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #1f6feb;
        color: white;
        border-bottom-right-radius: 4px;
        margin-left: auto;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #1f2937;
        color: #e5e5e5;
        border-bottom-left-radius: 4px;
    }
    
    /* Input box - nice rounded Grok style */
    .stChatInputContainer {
        background-color: #1f2937;
        border-radius: 24px;
        border: 1px solid #374151;
        padding: 4px 12px;
    }
    
    .stChatInput textarea {
        background-color: transparent !important;
        color: #161616 !important;
        border: none !important;
        font-size: 15px;
    }
    
    /* Title */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    /* Streaming cursor */
    .streaming-cursor {
        display: inline-block;
        width: 8px;
        height: 18px;
        background-color: #3b82f6;
        margin-left: 2px;
        animation: blink 0.8s step-end infinite;
        vertical-align: middle;
    }
    
    @keyframes blink {
        50% { opacity: 0; }
    }
    
    /* Memory cards */
    .todo-card {
        background-color: #1f2937;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER (Grok-like)
# ============================================
st.markdown('<p class="main-title">✅ task_mAIstro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your persistent AI task assistant • Powered by LangGraph + PostgreSQL</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR (Settings + Memories)
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    user_id = st.text_input("User ID", value=st.session_state.get("user_id", "default-user"))
    st.session_state.user_id = user_id
    
    todo_category = st.text_input("Category", value=st.session_state.get("todo_category", "general"))
    st.session_state.todo_category = todo_category
    
    task_maistro_role = st.text_area(
        "Personality / Role",
        value=st.session_state.get("task_maistro_role", configuration.Configuration.task_maistro_role),
        height=70
    )
    st.session_state.task_maistro_role = task_maistro_role
    
    st.divider()
    
    st.subheader("🔐 Database")
    postgres_uri = st.text_input(
        "PostgreSQL URI",
        value=os.getenv("POSTGRES_URI", ""),
        type="password",
        placeholder="postgresql://user:pass@host:5432/db?sslmode=require"
    )
    
    if postgres_uri:
        os.environ["POSTGRES_URI"] = postgres_uri
    
    if st.button("🔄 Reconnect Database", use_container_width=True):
        st.cache_resource.clear()
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()
    
    st.caption("💡 Add to `.streamlit/secrets.toml` for Cloud deploys")
    
    st.divider()
    
    # Memories section in sidebar
    st.subheader("🧠 Long-term Memory")
    
    if st.button("Refresh Memories", use_container_width=True):
        st.rerun()
    
    memory_placeholder = st.empty()

# ============================================
# PERSISTENCE
# ============================================
@st.cache_resource(show_spinner="Connecting to PostgreSQL...")
def get_graph_and_store(postgres_uri: str):
    if not postgres_uri:
        st.error("Please enter a PostgreSQL connection string in the sidebar.")
        st.stop()
    
    try:
        pool = ConnectionPool(
            conninfo=postgres_uri,
            max_size=20,
            # Remove this line completely:
            # kwargs={"autocommit": True}
        )
        
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool)
        
        checkpointer.setup()
        store.setup()
        
        graph = task_maistro.create_graph(checkpointer=checkpointer, store=store)
        return graph, store
        
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

graph, store = get_graph_and_store(postgres_uri)

# Config for this thread
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
# MEMORY DISPLAY (Sidebar)
# ============================================
def show_memories():
    with memory_placeholder.container():
        # Profile
        profile_ns = ("profile", todo_category, user_id)
        prof = store.search(profile_ns)
        with st.expander("👤 Profile", expanded=bool(prof)):
            if prof:
                st.json(prof[0].value)
            else:
                st.caption("No profile yet — tell me about yourself!")
        
        # Todos
        todo_ns = ("todo", todo_category, user_id)
        todos = store.search(todo_ns)
        with st.expander(f"📋 To-Dos ({len(todos)})", expanded=True):
            if todos:
                for t in todos:
                    item = t.value
                    emoji = {"not started": "⬜", "in progress": "🔄", "done": "✅", "archived": "📦"}.get(item.get("status"), "⬜")
                    st.markdown(f"""
                    <div class="todo-card">
                        <strong>{emoji} {item.get('task')}</strong><br>
                        <small>⏱ {item.get('time_to_complete', '?')} min • {item.get('deadline') or 'No deadline'}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No tasks yet. Ask me to add some!")
        
        # Instructions
        instr_ns = ("instructions", todo_category, user_id)
        instr = store.get(instr_ns, "user_instructions")
        with st.expander("⚙️ Instructions", expanded=False):
            if instr:
                st.text(instr.value.get("memory", ""))
            else:
                st.caption("No custom instructions yet.")

show_memories()

# ============================================
# CHAT AREA (Grok-style)
# ============================================
st.subheader(f"💬 {todo_category.capitalize()} • Thread: {thread_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

# ============================================
# CHAT INPUT + STREAMING (Main Feature)
# ============================================
if prompt := st.chat_input("What would you like to do today? Add tasks, update preferences, or just chat..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    # Assistant streaming response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Thinking..."):
                for event in graph.stream(inputs, config=config, stream_mode="messages"):
                    chunk, _ = event
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, list):
                            content = "".join(str(c) for c in content if isinstance(c, str))
                        full_response += str(content)
                        placeholder.markdown(full_response + '<span class="streaming-cursor"></span>', unsafe_allow_html=True)
            
            # Final message without cursor
            placeholder.markdown(full_response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Refresh memories after possible updates
            st.rerun()
            
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("task_mAIstro • Persistent memory with PostgreSQL • Token streaming enabled • Built with LangGraph + Streamlit")