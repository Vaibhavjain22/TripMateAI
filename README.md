# 🌍 TripMateAI — Multi-Agent Travel Planner

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LangGraph-1.2.2-green)](https://github.com/langchain-ai/langgraph)
[![API Gateway](https://img.shields.io/badge/FastAPI-0.136-teal)](https://fastapi.tiangolo.com/)
[![Database](https://img.shields.io/badge/PostgreSQL-Supabase-blueviolet)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TripMateAI** is an intelligent, autonomous multi-agent travel orchestration system powered by **LangGraph**, **FastAPI**, **PostgreSQL Checkpointing**, **OpenAI**, and the **Model Context Protocol (MCP)**. It transforms plain-language travel prompts into detailed, actionable travel dossiers complete with live flight options, hotel suggestions, day-by-day itineraries, and estimated budgets.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Directory Structure](#-project-directory-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
- [Installation & Quickstart](#-installation--quickstart)
- [How It Works](#-how-it-works)
- [API Reference](#-api-reference)
- [State Persistence & Checkpointing](#-state-persistence--checkpointing)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## 🚀 Overview

Planning a multi-day trip requires consulting multiple disparate sources: flight schedules, hotel availability, local attractions, and itinerary logistics. 

**TripMateAI** automates this end-to-end through specialized collaborating agents:
1. **Flight Agent**: Parses origin, destination, and routes to query live flight statuses and airline logistics.
2. **Hotel Agent**: Identifies top accommodations, ratings, and locations matching user preferences.
3. **Itinerary Agent**: Synthesizes flight and hotel data into realistic, well-paced day-by-day plans (morning, afternoon, evening).
4. **Final Aggregator Agent**: Validates output quality, computes estimated budgets, and formats a polished Markdown travel dossier.
5. **State Checkpointer**: Stores conversational checkpoints in PostgreSQL (Supabase) to enable seamless multi-turn sessions using unique `thread_id`s.

---

## 🏗️ System Architecture

```text
                                 +--------------------------------+
                                 |         User (Browser)         |
                                 +--------------------------------+
                                                  |
                                                  | HTTP POST /api/travel
                                                  v
                                 +--------------------------------+
                                 |        FastAPI Gateway         |
                                 |            (app.py)            |
                                 +--------------------------------+
                                                  |
                                                  v
                                 +--------------------------------+
                                 |      LangGraph StateGraph      |
                                 |          (backend.py)          |
                                 +--------------------------------+
                                                  |
         +-----------------------+----------------+-----------------------+
         |                       |                                        |
         v                       v                                        v
+------------------+    +------------------+                    +------------------+
|   Flight Agent   |    |   Hotel Agent    |                    | Itinerary Agent  |
|  (AviationStack) |    |  (Tavily Search) |                    |  (OpenAI LLM)    |
+------------------+    +------------------+                    +------------------+
         |                       |                                        |
         +-----------------------+----------------+-----------------------+
                                                  |
                                                  v
                                 +--------------------------------+
                                 |       Final Concierge Agent    |
                                 | (Aggregates Dossier & Budget)  |
                                 +--------------------------------+
                                                  |
                                                  v
                                 +--------------------------------+
                                 |    PostgreSQL Checkpointer     |
                                 |     (Supabase PostgresSaver)   |
                                 +--------------------------------+
```

---

## ✨ Key Features

- **Autonomous Agentic Workflow**: Built with `LangGraph` using state machines (`StateGraph`) with typed states and functional agent nodes.
- **Live Flight Intelligence**: Natural language route parsing into IATA codes (`airportsdata`, `pycountry`) with real-time status queries via AviationStack.
- **Deep Web Search**: Real-time accommodation and destination intelligence powered by **Tavily AI Search**.
- **Model Context Protocol (MCP)**: Native integration with remote MCP tool servers via `langchain-mcp-adapters` (`MultiServerMCPClient`).
- **Resilient Chat History & Persistence**: Production-grade `PostgresSaver` backed by Supabase PostgreSQL. Conversations survive server restarts and browser reloads.
- **Glassmorphic Responsive Web UI**: Modern dark-mode interface with live Markdown parsing (`marked.js`), 1-click clipboard copy, and client-side PDF export (`html2pdf.js`).

---

## 📂 Project Directory Structure

```text
TripMateAI/
├── .env                  # Environment secrets (ignored in Git)
├── .gitignore            # Git exclusion rules (venv, .env, cache)
├── LICENSE               # MIT License
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies with pinned versions
├── app.py                # FastAPI web server and routing gateway
├── backend.py            # LangGraph multi-agent workflow & PostgreSQL setup
├── demo_mcp_client.py    # Remote MCP Client adapter demo (Tavily Streamable HTTP)
├── static/
│   ├── style.css         # Glassmorphic dark UI design system
│   └── script.js         # Frontend logic, fetch handlers, and PDF export
├── templates/
│   └── index.html        # Interactive single-page travel planner UI
└── tools/
    ├── __init__.py
    ├── flight_tool.py    # AviationStack API wrapper + IATA & country parser
    └── tavily_tool.py    # Tavily Search API client
```

---

## 🛠️ Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Agentic Framework** | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain) |
| **LLM Provider** | [OpenAI](https://openai.com/) (`gpt-4o-mini`) |
| **Backend & Web Server** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) |
| **Tool Protocols (MCP)** | [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), `langchain-mcp-adapters` |
| **Database & Memory** | [PostgreSQL](https://www.postgresql.org/), [Supabase](https://supabase.com/), `langgraph-checkpoint-postgres`, `psycopg 3` |
| **Search & Real-time Data** | [Tavily AI Search](https://tavily.com/), [AviationStack API](https://aviationstack.com/) |
| **Geodata & Airports** | `airportsdata`, `pycountry` |
| **Frontend** | HTML5, Modern CSS (Glassmorphism), JavaScript (ES6+), Marked.js, html2pdf.js |

---

## ⚙️ Prerequisites & Environment Setup

### 1. Requirements
- **Python 3.10+** (Tested on Python 3.12)
- **PostgreSQL Database** (Cloud via [Supabase](https://supabase.com/) / [Neon](https://neon.tech/) or Local)
- **API Keys**:
  - [OpenAI API Key](https://platform.openai.com/)
  - [Tavily Search API Key](https://tavily.com/)
  - [AviationStack API Key](https://aviationstack.com/)
  - [LangSmith API Key](https://smith.langchain.com/) *(Optional for observability)*

### 2. Environment Configuration (`.env`)
Create a `.env` file in the root directory and configure your keys:

```env
# LLM Configuration
OPENAI_API_KEY="your-openai-api-key"

# Live Tools API Keys
TAVILY_API_KEY="your-tavily-api-key"
AVIATIONSTACK_API_KEY="your-aviationstack-api-key"

# PostgreSQL Database (Supabase Pooler connection string)
DATABASE_URL="postgresql://postgres.[PROJECT_ID]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require"

# Travel Defaults
DEFAULT_ORIGIN_IATA="DEL"

# Observability (Optional)
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT="TripMateAI"
```

---

## 📥 Installation & Quickstart

### Step 1: Clone the Repository
```bash
git clone https://github.com/Vaibhavjain22/TripMateAI.git
cd TripMateAI
```

### Step 2: Create and Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
Start the FastAPI server:
```bash
python app.py
```
*Or run directly with Uvicorn:*
```bash
uvicorn app:app --host 127.0.0.1 --port 5050 --reload
```

### Step 5: Open the Web UI
Navigate to **`http://127.0.0.1:5050`** in your browser to start planning trips!

---

## 🔍 How It Works

1. **User Query Input**: The user inputs a destination query (e.g. *"Plan a 7-day trip to Japan from Delhi under 2 lakhs"*).
2. **FastAPI Handler**: Receives the request and passes `user_query` and optional `thread_id` to `run_travel_agent()`.
3. **Flight Agent Execution**:
   - Parses locations (e.g., "Delhi" -> `DEL`, "Japan" -> `NRT`).
   - Queries AviationStack for real scheduled flight routes.
4. **Hotel Agent Execution**:
   - Constructs contextual search prompts for Tavily Search to identify top-rated accommodations and price-per-night estimates.
5. **Itinerary Agent Execution**:
   - Synthesizes flight and hotel candidates into day-by-day morning, afternoon, and evening activities.
6. **Final Agent Synthesis**:
   - Formats the complete dossier with an executive summary, budget breakdown, and pre-trip checklist.
7. **State Checkpointing**:
   - State is committed to PostgreSQL under the specified `thread_id`.
   - The user can ask follow-up questions in the same session without re-running flight searches!

---

## 🔌 Remote MCP Client Demo

TripMateAI includes a standalone client demonstration (`demo_mcp_client.py`) showing how agents interface with **Remote Model Context Protocol (MCP)** tool servers over **Streamable HTTP**:

```python
# demo_mcp_client.py
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "tavily": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    }
})

# Dynamically loads remote tools into standard LangChain tools
tools = await client.get_tools()
```

To run the MCP tool inspection demo:
```bash
python demo_mcp_client.py
```

---

## 📡 API Reference

### `GET /`
Renders the single-page TripMate AI frontend.

### `POST /api/travel`
Executes the multi-agent travel planning graph.

**Request Body:**
```json
{
  "message": "Plan a 5-day trip to Dubai including flights and hotels.",
  "thread_id": "user_session_abc123" // Optional; generated automatically if omitted
}
```

**Response Body:**
```json
{
  "success": true,
  "thread_id": "user_session_abc123",
  "answer": "# Trip Summary\n...",
  "flight_results": "Live flights from DEL to DXB...",
  "hotel_results": "[...]",
  "itinerary": "...",
  "llm_calls": 3
}
```

### `GET /health`
Returns system status.
```json
{
  "status": "ok",
  "message": "AI Travel Planner API is running"
}
```

---

## 💾 State Persistence & Checkpointing

LangGraph uses `langgraph-checkpoint-postgres` to store conversational memory directly in PostgreSQL.

When `checkpointer.setup()` runs, the following tables are automatically managed:
- **`checkpoints`**: Stores thread metadata, node execution steps, and state snapshots.
- **`checkpoint_blobs`**: Stores serialized message payloads and intermediate tool outputs.
- **`checkpoint_writes`**: Stores pending graph state transitions.

Because state is persisted by `thread_id`, the client browser saves the ID in `localStorage` to preserve session context seamlessly.

---

## 🗺️ Future Roadmap

- [ ] **Dual Human-in-the-Loop (HITL)**:
  - Gate 1: User approves/selects preferred flight & hotel candidate options before full itinerary generation.
  - Gate 2: Final review checkpoint asking *"Are you satisfied with this itinerary?"* with feedback re-routing.
- [ ] **Real-Time Streaming**: Implement Server-Sent Events (SSE) `/api/v1/plan/stream` to stream live agent steps and token generation.
- [ ] **Containerization**: Provide a full `docker-compose.yml` orchestrating FastAPI, local PostgreSQL, and a standalone FastMCP tool container.
- [ ] **Input & Output Guardrails**: Add schema validation and PII detection filters.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Crafted with ❤️ by [Vaibhav Jain](https://github.com/Vaibhavjain22)*
