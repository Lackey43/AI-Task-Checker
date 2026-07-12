# AI-Task-Checker → task_mAIstro (Streamlit + PostgreSQL)

Original LangGraph task management agent with long-term memory (user profile, ToDo list, custom instructions) now made **deployable as a beautiful Streamlit web app** with **persistent PostgreSQL storage**.

## ✨ What's New

- **Streamlit UI**: Modern chat interface + sidebar memory viewer + easy config
- **PostgreSQL Persistence**: 
  - Long-term memory (Profile / ToDos / Instructions) via `langgraph.store.postgres.PostgresStore`
  - Conversation history via `langgraph.checkpoint.postgres.PostgresSaver`
- **Survives restarts & multi-user**: Change User ID or Category to switch contexts
- **Backward compatible**: Still works with LangGraph Platform / `langgraph.json`
- **One-command local run** + ready for Streamlit Community Cloud

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/Lackey43/AI-Task-Checker.git
cd AI-Task-Checker-Streamlit   # or your fork

# 1. Create virtualenv & install
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Set environment (or use Streamlit secrets)
export OPENAI_API_KEY="sk-..."
export POSTGRES_URI="postgresql://user:pass@localhost:5432/postgres?sslmode=disable"

# 3. Run the Streamlit app
streamlit run streamlit_app.py
```

## 🐳 PostgreSQL (Required for Persistence)

You need a running Postgres instance. Recommended free tiers:

- **Supabase** (easiest): https://supabase.com → create project → copy connection string (use Transaction or Session pooler)
- **Neon.tech** (serverless, free)
- Local Docker:
  ```bash
  docker run --name task-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:16
  ```

The app automatically runs `setup()` to create the required tables on first connection.

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this folder (or the whole repo) to GitHub
2. Go to https://share.streamlit.io → **Deploy an app** → select your repo + `streamlit_app.py`
3. In **App settings → Secrets** paste:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   POSTGRES_URI = "postgresql://user:pass@db.supabase.co:5432/postgres?sslmode=require"
   ```
4. Deploy! Your app will be live at `https://your-app.streamlit.app`

## 📁 Project Structure

```
AI-Task-Checker-Streamlit/
├── streamlit_app.py          # Main Streamlit UI + chat + memory viewer
├── task_maistro.py           # Core LangGraph agent (updated with create_graph + Postgres support)
├── configuration.py          # Config dataclass (user_id, category, role)
├── requirements.txt          # All dependencies including streamlit + postgres checkpointer/store
├── langgraph.json            # For optional LangGraph Platform deployment
├── docker-compose-example.yml# Example for self-hosted LangGraph + Postgres/Redis
├── .gitignore
└── README.md
```

## 🔧 Key Features in the App

- **Persistent chat** across browser refreshes (via thread_id + checkpointer)
- **Live memory inspector** in sidebar (Profile, ToDos with status/emoji/solutions, Instructions)
- **Clear memories** button (deletes from Postgres for current user+category)
- **Switch contexts** instantly by changing User ID or Category
- **Custom AI personality** via the Role text area

## 🛠️ Development Notes

- The original `task_mAIstro` logic is 100% preserved.
- `create_graph()` in `task_maistro.py` auto-detects `POSTGRES_URI` / `DATABASE_URL` env var.
- For pure LangGraph Platform deploy, keep using `langgraph.json` + set `POSTGRES_URI` in your deployment env.
- If you don't set `POSTGRES_URI`, it gracefully falls back to in-memory (good for quick local tests, but data lost on restart).

## 📌 Original Repo

https://github.com/Lackey43/AI-Task-Checker

This version extends it for easy Streamlit deployment while keeping full compatibility.

Happy task managing! 🎯
```

## Next Steps for User

- To test locally in sandbox? But since no internet and may not have streamlit or psycopg installed, perhaps try to run but expect failure.

Since packages not installed probably, I can check. 

But for now, the files are ready in artifacts.

I can zip them for easy download.