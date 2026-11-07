# Joke Agent Lab

A small local learning project built with Python 3.13, Streamlit, FastAPI, LangChain,
LangGraph, and Ollama.

The backend runs a LangGraph tool-calling agent. All model inference is performed by a
local Ollama service through its OpenAI-compatible Chat Completions API.

Default model:

- `qwen3.8:27b-mlx`

The three joke-generation tools are:

- `tell_ussr_joke`
- `tell_stirlitz_joke`
- `tell_capitalism_joke`

Each tool receives `count`, `language`, and `keywords`. The agent extracts useful keywords
from the user's request and passes them to the selected tool. The tool's generation prompt
uses those keywords to guide the local model's choice of joke subject or comedic angle.

For other joke topics and normal questions, the agent responds directly without calling
these three tools.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/download) installed and running locally
- The `qwen3.8:27b-mlx` model downloaded in Ollama

No OpenAI account or OpenAI API key is required.

## Prepare Ollama

Start the Ollama service using the normal Ollama application/service for your operating
system. Confirm that its local API is available:

```bash
curl http://127.0.0.1:11434/api/version
```

Pull the exact model used by this project:

```bash
ollama pull qwen3.8:27b-mlx
ollama list
```

Confirm that the model is visible through the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:11434/v1/models
```

Before running the application, verify that the installed Ollama version and this model
support OpenAI-compatible tool calls. Use the manual tool-call smoke test in Section 6.2
of this specification if tool calls are not being returned by the agent.

## Setup the Python project

Create the local environment file:

```bash
cp .env.example .env
```

The defaults use:

- Ollama base URL: `http://127.0.0.1:11434/v1`
- Ollama model: `qwen3.8:27b-mlx`
- Placeholder API key: `ollama` (ignored by the local Ollama server)

Change `OLLAMA_BASE_URL` or `OLLAMA_MODEL` only when intentionally using a different local
Ollama endpoint or model. Do not replace the placeholder key with an OpenAI API key.

Install all dependencies, including development tools:

```bash
uv sync --all-groups
```

## Run the backend

If you'd like to have traces in LangSmith then get your api key in LangSmith and fill in the LangSmith section in .env file

From the repository root, in terminal 1:

```bash
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000 --env-file .env
```

Wihout LangSmith, you can run the backend without the `--env-file .env` flag.

Useful endpoints:

- Health: <http://127.0.0.1:8000/health>
- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>

## Run the frontend

From the repository root, in terminal 2:

```bash
uv run streamlit run frontend/streamlit_app.py
```

Open <http://localhost:8501>.

## Run automated checks

```bash
uv run ruff check .
uv run pytest -q
```

## OpenTelemetry tracing

```bash
docker run -d --name joke-agent-jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/jaeger:latest
```

Open the [Jaeger UI](http://localhost:16686)

## Architecture

```text
Browser
  |
  v
Streamlit frontend
  | HTTP / JSON
  v
FastAPI backend
  |
  v
LangGraph StateGraph
  |
  +--> Agent node (ChatOpenAI -> local Ollama /v1/chat/completions)
  |       |
  |       +--> ToolNode
  |              +--> tell_ussr_joke ----+
  |              +--> tell_stirlitz_joke -+--> local Ollama joke-generation call
  |              +--> tell_capitalism_joke+
  |       |
  |       +<-- Agent node after tool result
  |
  +--> InMemorySaver checkpoint by session_id/thread_id
```

## Notes and limitations

- Inference stays local on the configured Ollama server. No public OpenAI endpoint is used.
- The API key value `ollama` is a required client placeholder and is ignored by local Ollama.
- Conversation checkpoints are stored in process memory and disappear when FastAPI restarts.
- Run one backend process only. Do not start multiple Uvicorn workers because each worker
  would have a separate in-memory checkpointer.
- The frontend stores the visible transcript in Streamlit session state. The backend stores
  the agent conversation state in LangGraph checkpoints.
- The `/health` endpoint does not initialize the LLM and can work while Ollama is stopped.
- The first message can take longer while Ollama loads the model into memory.
- LangSmith and OpenTelemetry are intentionally not configured in Phase 1.
