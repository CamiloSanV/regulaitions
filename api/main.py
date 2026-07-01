"""
api/main.py
───────────
regulAItions API with:
  - POST /chat          → full response at once
  - POST /chat/stream   → SSE streaming
  - GET  /health        → healthcheck for Render
  - GET  /sources       → list indexed regulations
"""

import json
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="regulAItions API",
    description="AI regulation compliance agent for EU AI Act and GDPR",
    version="0.1.0",
)

# CORS — allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sessions in memory ─────────────────────────────────────────────────────────
sessions: dict[str, object] = {}

def get_or_create_session(session_id: str):
    if session_id not in sessions:
        from agent.agent import AgentSession
        sessions[session_id] = AgentSession()
    return sessions[session_id]


# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[dict]
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/sources")
def list_sources():
    return {
        "sources": [
            {"id": "EU_AI_Act", "name": "EU AI Act", "version": "2024-Q3", "effective_date": "2024-08-01"},
            {"id": "GDPR",      "name": "GDPR",       "version": "2018-Q2", "effective_date": "2018-05-25"},
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Full response at once — use for testing in Swagger."""
    try:
        session = get_or_create_session(request.session_id)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, session.chat, request.message)
        return ChatResponse(
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE streaming endpoint.
    
    Event types the frontend receives:
      {"type": "thinking"}                          → agent started
      {"type": "tool_call", "tool": "...", "input": "..."}  → tool being called
      {"type": "token", "content": "..."}           → response word by word
      {"type": "done", "tool_calls": [...]}         → finished
      {"type": "error", "content": "..."}           → something went wrong
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            session = get_or_create_session(request.session_id)

            # 1. Notify frontend the agent is starting
            yield json.dumps({"type": "thinking"})

            # 2. Send keepalive pings every 3 seconds while agent runs
            #    This prevents the browser from closing the SSE connection
            result_holder = {}
            error_holder = {}
            done_event = asyncio.Event()

            async def run_agent():
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, session.chat, request.message
                    )
                    result_holder["result"] = result
                except Exception as e:
                    error_holder["error"] = str(e)
                finally:
                    done_event.set()

            # Start agent in background
            agent_task = asyncio.create_task(run_agent())

            # Send keepalives while waiting
            while not done_event.is_set():
                try:
                    await asyncio.wait_for(done_event.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    # Send a comment to keep the connection alive
                    yield json.dumps({"type": "thinking"})

            await agent_task

            # 3. Handle errors
            if "error" in error_holder:
                yield json.dumps({"type": "error", "content": error_holder["error"]})
                return

            result = result_holder["result"]

            # 4. Send tool calls
            for tc in result["tool_calls"]:
                yield json.dumps({
                    "type": "tool_call",
                    "tool": tc["tool"],
                    "input": str(tc["input"])[:100],
                })
                await asyncio.sleep(0.05)

            # 5. Stream answer word by word
            answer = result["answer"]
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield json.dumps({"type": "token", "content": chunk})
                await asyncio.sleep(0.02)

            # 6. Done signal
            yield json.dumps({
                "type": "done",
                "tool_calls": result["tool_calls"],
            })

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)})

    return EventSourceResponse(generate())


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"cleared": session_id}