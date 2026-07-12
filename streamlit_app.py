import streamlit as st
import os
import warnings
from langchain_core.messages import HumanMessage

import task_maistro
import configuration

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv

load_dotenv()

# Suppress the harmless ConnectionPool shutdown warning
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*cannot join current thread.*"
)

# ============================================
# PAGE CONFIG + CLEAN UI
# ============================================
st.set_page_config(
    page_title="task_mAIstro",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; color: #e5e5e5; }
    .stChatMessage {
        padding: 1rem 1.25rem; border-radius: 18px; margin-bottom: 12px;
        max-width: 85%; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #1f6feb; color: white; border-bottom-right-radius: 4px; margin-left: auto;
    }
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #1f2937; color: #e5e5e5; border-bottom-left-radius: 4px;
    }
    .stChatInputContainer {
        background-color: #1f2937; border-radius: 24px; border: 1px solid #374151; padding: 4px 12px;
    }
    .main-title { font-size: 2.1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.1rem; }
    .subtitle { color: #9ca3af; font-size: 0.9rem; margin-bottom: 1.2rem; }
    .streaming-cursor {
        display: inline-block; width: 7px; height: 17px; background-color: #3b82f6;
        margin-left: 3px; animation: blink 0.75s step-end infinite; vertical-align: middle;
    }
    @keyframes blink { 50% { opacity: 0; } }
    .todo-card {
        background-color: #1f2937; padding: 0.8rem 1rem; border-radius: 12px;
        margin-bottom: 0.5rem; border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<p class="main-title">✅ task_mAIstro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Persistent AI Task Assistant • Long-term memory powered by PostgreSQL</p>', unsafe_allow_html=True)

# ============================================
# LOAD DATABASE SECURELY (NO INPUT BOX)
# ============================================
postgres_uri = os.getenv("POSTGRES_URI") or st.secrets.get("POSTGRES_URI", "")

if not postgres_uri:
    st.error("❌ PostgreSQL URI not found. Please set it in your environment or `.streamlit/secrets.toml`.")
    st.stop()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    user_id = st.text_input("User ID", value=st.session_state.get("user_id", "default-user"))
    st.session_state.user_id = user_id
    
    todo_category = st.text_input("Category", value=st.session_state.get("todo_category", "general"))
    st.session_state.todo_category = todo_category
    
    task_maistro_role = st.text_area(
        "AI Personality",
        value=st.session_state.get("task_maistro_role", configuration.Configuration.task_maistro_role),
        height=65
    )
    st.session_state.task_maistro_role = task_maistro_role
    
    st.caption("🔒 Database connected securely via environment variable")
    
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.cache_resource.clear()
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.subheader("🧠 Long-term Memory")
    
    if st.button("🔄 Load / Refresh Memories", use_container_width=True):
        with st.spinner("Loading from database..."):
            show_memories()
    else:
        st.caption("Click above to load your profile and todos")

# ============================================
# DATABASE CONNECTION (Best settings)
# ============================================
@st.cache_resource(show_spinner="Connecting to PostgreSQL...")
def get_graph_and_store(postgres_uri: str):
    try:
        pool = ConnectionPool(
            conninfo=postgres_uri,
            max_size=15,
            min_size=1,
            max_lifetime=300,
            timeout=30,
            kwargs={
                "prepare_threshold": 0,
                "autocommit": True,
            }
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

# Config
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
# MEMORY DISPLAY
# ============================================
def show_memories():
    # Profile
    profile_ns = ("profile", todo_category, user_id)
    prof = store.search(profile_ns)
    with st.expander("👤 Profile", expanded=bool(prof)):
        if prof:
            st.json(prof[0].value)
        else:
            st.caption("No profile yet.")

    # To-Dos
    todo_ns = ("todo", todo_category, user_id)
    todos = store.search(todo_ns)
    with st.expander(f"📋 To-Do List ({len(todos)})", expanded=True):
        if todos:
            for t in todos:
                item = t.value
                emoji = {"not started": "⬜", "in progress": "🔄", "done": "✅", "archived": "📦"}.get(item.get("status"), "⬜")
                st.markdown(f"""
                <div class="todo-card">
                    <strong>{emoji} {item.get('task', 'Untitled')}</strong><br>
                    <small>⏱ {item.get('time_to_complete', '?')} min</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No tasks yet. Ask the AI to add some!")

    # Instructions
    instr_ns = ("instructions", todo_category, user_id)
    instr = store.get(instr_ns, "user_instructions")
    with st.expander("⚙️ Custom Instructions", expanded=False):
        if instr:
            st.text(instr.value.get("memory", ""))
        else:
            st.caption("No custom instructions set.")

# ============================================
# CHAT + STREAMING
# ============================================
st.subheader(f"💬 Chat • {todo_category}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What would you like me to do?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    inputs = {"messages": [HumanMessage(content=prompt)]}

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

            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

st.divider()
st.caption("task_mAIstro • Persistent memory with PostgreSQL • Token streaming")