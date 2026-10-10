# 🌍 TripMateAI — Production Multi-Agent Travel Planner

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Agent Framework](https://img.shields.io/badge/LangGraph-1.2.2-green)](https://github.com/langchain-ai/langgraph)
[![API Gateway](https://img.shields.io/badge/FastAPI-0.136-teal)](https://fastapi.tiangolo.com/)
[![Database Persistence](https://img.shields.io/badge/PostgreSQL-Supabase-blueviolet)](https://supabase.com/)
[![Protocol](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-orange)](https://modelcontextprotocol.io/)
[![Observability](https://img.shields.io/badge/LangSmith-Enabled-orange.svg)](https://smith.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TripMateAI** is an autonomous, production-grade multi-agent travel orchestration engine built with **LangGraph**, **FastAPI**, **PostgreSQL Checkpointing (Supabase)**, **OpenAI**, and the **Model Context Protocol (MCP)**. 

Transforming natural language travel prompts into comprehensive travel dossiers, TripMateAI dynamically coordinates specialist agents for live flights, accommodations, real-time weather forecasts, budget analysis, and day-by-day itineraries — complete with deterministic safety guardrails and native **Human-in-the-Loop (HITL)** approval before finalization.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Multi-Agent Orchestration & Workflow](#-multi-agent-orchestration--workflow)
- [Key Features](#-key-features)
- [Model Context Protocol (MCP) Integration](#-model-context-protocol-mcp-integration)
- [Human-in-the-Loop (HITL) Workflow](#-human-in-the-loop-hitl-workflow)
- [Project Directory Structure](#-project-directory-structure)
- [Tech Stack](#-tech-stack)
- [Environment Setup & Configuration](#-environment-setup--configuration)
- [Installation & Quickstart](#-installation--quickstart)
- [API Reference](#-api-reference)
- [State Persistence & Checkpointing](#-state-persistence--checkpointing)
- [Observability with LangSmith](#-observability-with-langsmith)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## 🚀 Overview

Planning travel involves consulting fragmented sources: airline schedules, hotel reviews, meteorological conditions, budget constraints, and activity scheduling.

**TripMateAI** replaces manual planning with a collaborative multi-agent architecture:
1. **Input Guardrail**: Deterministically screens user prompts to block out-of-domain, malicious, or abusive requests before downstream LLM execution.
2. **Supervisor Agent**: Parses travel constraints (`origin`, `destination`, `budget`, `duration`) and dynamically selects which specialist agents are required.
3. **Flight Agent**: Interacts with the **AviationStack MCP Server** (using live departure boards and operating carrier tracking) to extract realistic route options.
4. **Hotel Agent**: Performs deep web search via **Tavily MCP** to recommend accommodations tailored to the destination and budget.
5. **Weather Agent**: Queries a custom **FastMCP Weather Server** for real-time conditions and 5-day forecasts to provide weather-aware packing and activity advice.
6. **Budget Analyst**: Evaluates expense feasibility across flights, hotels, meals, and local transit.
7. **Itinerary Agent**: Fuses specialist findings into a realistic draft itinerary.
8. **Human-in-the-Loop (HITL) Gate**: Pauses execution via LangGraph's `interrupt()` to present the draft plan to the user for approval or iterative feedback.
9. **Final Agent**: Synthesizes the finalized, approved dossier into an executive 7-part travel plan.

---

## 🏗️ System Architecture

```text
                               +------------------------------------+
                               |           User Browser             |
                               +------------------------------------+
                                           |            ^
                    POST /api/travel       |            | Final Plan / Draft Review
                    POST /api/travel/approve v            |
                               +------------------------------------+
                               |          FastAPI Gateway           |
                               |             (app.py)               |
                               +------------------------------------+
                                                 |
                                                 v
                               +------------------------------------+
                               |        LangGraph StateGraph        |
                               |            (backend.py)            |
                               +------------------------------------+
                                                 |
                                                 v
                               +------------------------------------+
                               |     Input Safety Guardrail         |
                               +------------------------------------+
                                        /                  \
                       [Blocked]       /                    \ [Allowed]
                                      v                      v
                       +--------------------+      +--------------------+
                       | Guardrail Blocked  |      |  Supervisor Agent  |
                       +--------------------+      +--------------------+
                                 |                           |
                                 v              Dynamic Route Selection
                                END                          |
                     +-------------------+-------------------+-------------------+
                     |                   |                   |                   |
                     v                   v                   v                   v
            +-----------------+ +-----------------+ +-----------------+ +-----------------+
            |  Flight Agent   | |   Hotel Agent   | |  Weather Agent  | |  Budget Agent   |
            | (AviationStack) | |  (Tavily MCP)   | | (FastMCP Server)| | (LLM Reasoning) |
            +-----------------+ +-----------------+ +-----------------+ +-----------------+
                     |                   |                   |                   |
                     +-------------------+-------------------+-------------------+
                                                 |
                                                 v
                               +------------------------------------+
                               |         Itinerary Agent            |
                               |     (Draft Synthesis Agent)        |
                               +------------------------------------+
                                                 |
                                                 v
                               +------------------------------------+
                               |     Human-in-the-Loop (HITL)       |
                               |       interrupt() Pause Gate       |
                               +------------------------------------+
                                        /                  \
                        [Approved]     /                    \ [Revision Requested]
                                      v                      v
                       +--------------------+      +--------------------+
                       |    Final Agent     |      |  Refined Feedback  |
                       | (7-Part Dossier)   |      |  Applied to Final  |
                       +--------------------+      +--------------------+
                                        \                  /
                                         v                v
                               +------------------------------------+
                               |      PostgreSQL Checkpointer       |
                               |    (Supabase PostgresSaver)        |
                               +------------------------------------+
                                                 |
                                                 v
                                                END
```

---

## 🤖 Multi-Agent Orchestration & Workflow

| Agent / Node | Primary Responsibility | Data Source / Tool Protocol |
| :--- | :--- | :--- |
| **`guardrail_blocked`** | Early exit for non-travel queries | System prompt classification |
| **`supervisor`** | Constraint extraction & dynamic agent dispatch | `gpt-4o-mini` with strict JSON schema |
| **`flight_agent`** | Departure schedules, airlines, and duration estimates | **AviationStack MCP** (`stdio`) |
| **`hotel_agent`** | Accommodations, neighborhoods, and nightly rates | **Tavily AI Search MCP** (`streamable_http`) |
| **`weather_agent`** | Live weather conditions, climate overview, and forecast | **Custom Weather FastMCP Server** (`stdio`) |
| **`budget_agent`** | Viability checks, per-category cost breakdown | `gpt-4o-mini` financial analysis |
| **`itinerary_agent`** | Assembles integrated morning/afternoon/evening schedule | State-fused synthesis node |
| **`human_approval`** | Pauses graph execution for user validation / critique | LangGraph `interrupt()` mechanism |
| **`final_agent`** | Generates polished 7-part travel dossier | LLM synthesis with user approval context |

---

## ✨ Key Features

- **Dynamic Supervisor Routing**: Only executes agents relevant to the user request. A flight-only query bypasses hotel and weather scrapers; a comprehensive query orchestrates the full specialist panel.
- **Model Context Protocol (MCP)**: Native integration with Anthropic's open standard for tool servers, supporting both **Streamable HTTP** (remote Tavily) and local **`stdio` subprocesses** (AviationStack and FastMCP OpenWeather).
- **Human-in-the-Loop Review**: Allows the user to inspect draft itineraries before finalizing, request adjustments (e.g. *"find cheaper hotels"* or *"more outdoor activities"*), or approve with one click.
- **Production State Checkpointing**: Powered by `langgraph-checkpoint-postgres` over Supabase PostgreSQL connection pooling (`PostgresSaver`). Conversations survive restarts and thread states resume seamlessly.
- **Resilient Tool Design**: Tailored to AviationStack's free-tier `/v1/flights` capabilities, ensuring live airport departures and airline tracking operate without paid-tier catalog errors.
- **Full Observability**: Integrated with **LangSmith** for trace visualization, token consumption auditing, and latency inspection across every agent node.
- **Interactive UI**: Dark-mode glassmorphic interface with active agent chips, marked.js Markdown rendering, 1-click clipboard copying, and client-side PDF export (`html2pdf.js`).

---

## 🔌 Model Context Protocol (MCP) Integration

TripMateAI uses `langchain-mcp-adapters` with a multi-server MCP architecture:

```python
# mcp_client.py
client = MultiServerMCPClient({
    # 1. Remote Streamable HTTP MCP
    "tavily": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    },
    # 2. Local stdio MCP (AviationStack)
    "aviationstack": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["aviationstack-mcp"],
        "env": {"AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY}
    },
    # 3. Custom FastMCP Weather Server (Local stdio)
    "weather": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(WEATHER_SERVER_PATH)],
        "env": {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY}
    }
})
```

### Custom Weather FastMCP Server (`custom_weather_mcp_server.py`)
Built using FastMCP to expose clean tool schemas to LLMs:
- **`get_current_weather(city: str)`**: Temperature, humidity, wind speed, and meteorological conditions.
- **`get_forecast(city: str)`**: 5-day daily forecast breakdowns.

---

## ⏸️ Human-in-the-Loop (HITL) Workflow

TripMateAI implements true state-level execution pausing:

1. **State Interruption**: When `itinerary_agent` finishes, `human_approval_agent` triggers:
   ```python
   review = interrupt({
       "question": "Do you approve this itinerary?",
       "draft_itinerary": state.get("itinerary", ""),
       "approval_request": state.get("approval_request", "")
   })
   ```
2. **Checkpoint Saved**: LangGraph saves the paused execution state in PostgreSQL and returns `requires_approval: true` to FastAPI.
3. **User Action**: The frontend renders the **Human-in-the-Loop Review** banner:
   - **Approve & Finalize**: Calls `POST /api/travel/approve` with `approved: true`.
   - **Request Revision**: Allows the user to type revision feedback (e.g. *"add vegetarian food recommendations on Day 3"*) and submits with `approved: false`.
4. **Deterministic Resumption**:
   ```python
   travel_graph.invoke(
       Command(resume={"approved": approved, "feedback": feedback}),
       config={"configurable": {"thread_id": thread_id}}
   )
   ```
5. **Polished Dossier**: `final_agent` receives the review instruction, applies the user's critique, and outputs the final plan.

---

## 📂 Project Directory Structure

```text
TripMateAI/
├── .env                              # API keys and connection secrets (git-ignored)
├── .gitignore                        # Git exclusion rules
├── LICENSE                           # MIT License
├── README.md                         # Comprehensive documentation
├── requirements.txt                  # Pinned Python package dependencies
├── app.py                            # FastAPI application server and REST endpoints
├── backend.py                        # LangGraph DAG definition, supervisor & agent nodes
├── custom_weather_mcp_server.py      # Standalone FastMCP Weather server (OpenWeatherMap)
├── mcp_client.py                     # MultiServerMCPClient orchestrator & tool wrappers
├── test.py                           # Quick-validation script for testing graph runs
├── static/
│   ├── style.css                     # Glassmorphic dark design system & print styles
│   └── script.js                     # Frontend API bindings, HITL handling, and PDF export
└── templates/
    └── index.html                    # Single-page interface with dynamic agent badges
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) (v1.2.2) | State machines, cyclic graphs, and interrupt gates |
| **Core Framework** | [LangChain](https://github.com/langchain-ai/langchain) | LLM messaging, prompts, and tool abstractions |
| **Language Model** | [OpenAI](https://openai.com/) (`gpt-4o-mini`) | Supervisor classification and specialist agent generation |
| **Tool Protocol** | [Model Context Protocol](https://modelcontextprotocol.io/) | Standardized tool client (`MultiServerMCPClient`) |
| **Custom MCP** | [FastMCP](https://github.com/jlowin/fastmcp) | Local weather server exposing OpenWeatherMap endpoints |
| **Web Server** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) | Async REST API gateway with Jinja2 template rendering |
| **Persistence** | [PostgreSQL](https://www.postgresql.org/), [Supabase](https://supabase.com/) | Durable thread checkpointer (`PostgresSaver`, `psycopg 3`) |
| **Live Tool APIs** | [Tavily Search](https://tavily.com/), [AviationStack](https://aviationstack.com/) | Real-time web intelligence and flight departure tracking |
| **Observability** | [LangSmith](https://smith.langchain.com/) | End-to-end trace auditing, token costs, and node latencies |
| **Frontend UI** | Vanilla HTML5, CSS3, JavaScript | Modern glassmorphism, Marked.js, and html2pdf.js |

---

## ⚙️ Environment Setup & Configuration

Create a `.env` file in the project root:

```env
# =========================================================
# Core LLM API Key
# =========================================================
OPENAI_API_KEY="sk-proj-your-openai-api-key"

# =========================================================
# Live Tools & Services
# =========================================================
TAVILY_API_KEY="tvly-your-tavily-api-key"
AVIATIONSTACK_API_KEY="your-aviationstack-api-key"
OPENWEATHER_API_KEY="your-openweathermap-api-key"

# =========================================================
# PostgreSQL Database (Supabase Session Pooler URL)
# =========================================================
DATABASE_URL="postgresql://postgres.[REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres?sslmode=require"

# =========================================================
# Defaults
# =========================================================
DEFAULT_ORIGIN_IATA="DEL"

# =========================================================
# LangSmith Observability & Tracing
# =========================================================
LANGSMITH_TRACING="true"
LANGCHAIN_TRACING_V2="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="lsv2_pt_your-langsmith-key"
LANGSMITH_PROJECT="TripMateAI"
```

---

## 📥 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Vaibhavjain22/TripMateAI.git
cd TripMateAI
```

### 2. Set Up Virtual Environment
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

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure `uvx` is Available (for AviationStack MCP)
The AviationStack MCP server runs via `uvx`. Ensure `uv` is installed:
```bash
pip install uv
uvx --version
```

### 5. Launch the Application
```bash
uvicorn app:app --host 127.0.0.1 --port 5050 --reload
```

Open **`http://127.0.0.1:5050`** in your browser.

---

## 📡 API Reference

### 1. Generate Travel Plan
`POST /api/travel`

Starts a new travel planning thread and pauses at the draft review stage.

**Request Body:**
```json
{
  "message": "Plan a 7-day trip to Japan from Bangladesh with budget hotels under 2 lakhs.",
  "thread_id": null
}
```

**Response (Draft Stage — Pending Review):**
```json
{
  "success": true,
  "thread_id": "user_a1b2c3d4e5f6",
  "answer": "### Draft Itinerary: 7-Day Japan Adventure\n...",
  "requires_approval": true,
  "approval_request": "Please review the generated draft itinerary. Approve it to create the final polished plan, or provide feedback for revision.",
  "selected_agents": [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
  ],
  "trip_constraints": {
    "destination": "Japan",
    "origin": "Bangladesh",
    "duration": "7 days",
    "budget": "2 lakhs"
  },
  "guardrail_allowed": true,
  "llm_calls": 5
}
```

---

### 2. Submit Human Review (Approve / Revise)
`POST /api/travel/approve`

Resumes a paused thread to produce the finalized dossier.

**Request Body (Approve):**
```json
{
  "thread_id": "user_a1b2c3d4e5f6",
  "approved": true,
  "feedback": ""
}
```

**Request Body (Request Revision):**
```json
{
  "thread_id": "user_a1b2c3d4e5f6",
  "approved": false,
  "feedback": "Please swap out Day 3 for more historic temple visits in Kyoto and suggest budget ramen shops."
}
```

**Response (Finalized Plan):**
```json
{
  "success": true,
  "thread_id": "user_a1b2c3d4e5f6",
  "answer": "# 🇯🇵 Complete 7-Day Japan Travel Dossier\n\n## 1. Trip Summary\n...\n## 2. Flight Information\n...\n## 3. Hotel Suggestions\n...\n## 4. Weather Information\n...\n## 5. Day-by-Day Itinerary\n...\n## 6. Estimated Budget\n...\n## 7. Final Recommendations\n...",
  "requires_approval": false,
  "approved": true,
  "llm_calls": 6
}
```

---

### 3. Health Check
`GET /health`

```json
{
  "status": "ok",
  "message": "TripMate AI API is running",
  "features": [
    "supervisor_agent",
    "input_guardrail",
    "human_in_the_loop"
  ]
}
```

---

## 💾 State Persistence & Checkpointing

TripMateAI relies on `langgraph-checkpoint-postgres` to manage conversation memory in PostgreSQL:

- **`checkpoints`**: Stores thread execution history, active state, and configuration versions.
- **`checkpoint_blobs`**: Stores binary/JSON serialized intermediate outputs (flight tables, hotel listings, weather maps).
- **`checkpoint_writes`**: Records state mutations and pending interrupts.

Every user session operates within an isolated `thread_id`. When an interrupt occurs, the thread can be paused indefinitely without holding memory in Python, enabling multi-tenant horizontal scaling.

---

## 📊 Observability with LangSmith

Full tracing is enabled across all LangGraph nodes:

```text
[TripMateAI Run]
├── supervisor (1.1s) -> JSON constraints extracted
├── flight_agent (1.4s) -> AviationStack MCP departures retrieved
├── hotel_agent (1.8s) -> Tavily MCP hotel search completed
├── weather_agent (0.9s) -> FastMCP temperature & forecast retrieved
├── budget_agent (1.2s) -> Budget feasibility computed
├── itinerary_agent (2.1s) -> Draft itinerary compiled
├── human_approval (INTERRUPTED) -> Paused for user interaction
└── final_agent (3.2s) -> Final 7-part travel dossier generated
```

View trace graphs, step latencies, and token expenditures at [smith.langchain.com](https://smith.langchain.com).

---

## 🗺️ Future Roadmap

- [ ] **Parallel Specialist Fan-Out**: Execute independent specialists (`flight`, `hotel`, `weather`) concurrently via LangGraph branch fan-out to reduce initial latency.
- [ ] **Server-Sent Events (SSE) Streaming**: Stream token chunks and intermediate agent thought badges directly to the browser UI.
- [ ] **Multi-City Route Optimization**: Add topological sorting for multi-destination train/flight logistics.
- [ ] **Docker Deployment**: Package FastAPI, MCP tools, and dependencies into a production-ready `Dockerfile` and `docker-compose.yml`.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Architected with ❤️ by [Vaibhav Jain](https://github.com/Vaibhavjain22)*
