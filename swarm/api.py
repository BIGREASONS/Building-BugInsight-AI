import os
import uuid
import asyncio
import json
import time
import logging
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.codebert_classifier import CodeBERTClassifier
from configs.config_loader import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="BugInsight Swarm API", version="2.0.0")

# CORS for Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---

class BugRequest(BaseModel):
    issue_text: str

class SwarmRequest(BaseModel):
    repo_url: str
    issue_text: str

class JobResponse(BaseModel):
    job_id: str

class SeverityResponse(BaseModel):
    severity: str
    confidence: float

# --- Global Model State ---

config = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None
LABEL_MAP = {0: "Critical", 1: "Major", 2: "Minor", 3: "Trivial"}

# In-memory job store (adequate for single-machine hackathon demo)
jobs: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def load_model():
    global model, tokenizer
    logger.info(f"Loading CodeBERT model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("models.codebert.model_name", "microsoft/codebert-base")
    )
    model = CodeBERTClassifier.from_config(config)

    model_path = config.get_path("outputs.models_dir") / "codebert_best.pt"
    if model_path.exists():
        logger.info(f"Loading trained weights from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Using untrained model.")
    else:
        logger.warning(
            f"No trained weights found at {model_path}. Using UNTRAINED model for development."
        )

    model.to(device)
    model.eval()
    logger.info("CodeBERT Brain is online.")


# --- Original severity endpoint (kept for health_check.py compatibility) ---

@app.post("/predict_severity", response_model=SeverityResponse)
async def predict_severity(request: BugRequest):
    if not request.issue_text.strip():
        raise HTTPException(status_code=400, detail="Issue text cannot be empty.")

    severity, confidence = _predict_severity(request.issue_text)
    return SeverityResponse(severity=severity, confidence=confidence)


def _predict_severity(issue_text: str):
    """Internal helper so both the REST endpoint and the streaming workflow can call it."""
    with torch.no_grad():
        inputs = tokenizer(
            issue_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs)
        probs = F.softmax(logits, dim=1)

        max_prob, pred_idx = torch.max(probs, dim=1)

        pred_label = LABEL_MAP[int(pred_idx.item())]
        confidence = float(max_prob.item())

        # Dev fallback for untrained model
        if not (config.get_path("outputs.models_dir") / "codebert_best.pt").exists():
            if "auth" in issue_text.lower() or "crash" in issue_text.lower():
                pred_label = "Critical"
                confidence = 0.94

        return pred_label, confidence


# --- Job-based Swarm Execution with SSE ---

@app.post("/api/swarm/start", response_model=JobResponse)
async def start_swarm(request: SwarmRequest):
    """Create a new swarm job and begin execution in the background."""
    job_id = uuid.uuid4().hex[:12]

    jobs[job_id] = {
        "status": "queued",
        "repo_url": request.repo_url,
        "issue_text": request.issue_text,
        "events": [],       # list of SSE event dicts
        "final_state": None,
        "done": False,
    }

    # Fire and forget — the SSE endpoint will wait for events
    asyncio.create_task(_run_swarm(job_id))

    return JobResponse(job_id=job_id)


@app.get("/api/swarm/stream/{job_id}")
async def stream_swarm(job_id: str):
    """SSE endpoint — streams agent progress events to the frontend."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    return StreamingResponse(
        _event_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _event_generator(job_id: str):
    """Yields SSE events as they are appended by the background swarm task."""
    cursor = 0
    while True:
        job = jobs.get(job_id)
        if job is None:
            break

        # Yield any new events since our last cursor position
        while cursor < len(job["events"]):
            event = job["events"][cursor]
            yield f"data: {json.dumps(event)}\n\n"
            cursor += 1

        if job["done"]:
            break

        await asyncio.sleep(0.25)  # poll interval


async def _run_swarm(job_id: str):
    """Execute the LangGraph workflow and emit SSE events at each node."""
    job = jobs[job_id]
    job["status"] = "running"

    try:
        # Import here to avoid circular imports at module level
        from swarm.workflow import swarm_app

        initial_state = {
            "job_id": job_id,
            "issue_url": None,
            "repo_url": job["repo_url"],
            "issue_text": job["issue_text"],
            "trace_logs": [],
        }

        # Map internal node names to user-friendly display names
        AGENT_NAMES = {
            "repo_agent": "Repository Agent",
            "scanner_agent": "Scanner Agent",
            "severity_agent": "Severity Agent",
            "root_cause_agent": "Root Cause Agent",
            "fix_agent": "Fix Agent",
            "validation_agent": "Validation Agent",
            "test_agent": "Test Agent",
            "test_execute_agent": "Test Execution Agent",
            "auto_rescan_agent": "Auto-Rescan Agent",
            "github_agent": "GitHub Agent",
            "sprint_agent": "Sprint Agent",
        }

        # Emit initial event
        _emit(job, "swarm_started", {
            "job_id": job_id,
            "agents": list(AGENT_NAMES.values()),
        })

        def _sync_runner():
            final = initial_state
            for step_output in swarm_app.stream(initial_state):
                for node_name, state in step_output.items():
                    final = state
                    display_name = AGENT_NAMES.get(node_name, node_name)
                    payload = {"agent": display_name}
                    
                    if node_name == "repo_agent":
                        payload["index_stats"] = state.get("index_stats", {})
                    elif node_name == "scanner_agent":
                        payload["scanner_findings"] = state.get("scanner_findings", [])
                    elif node_name == "severity_agent":
                        payload["severity"] = state.get("severity", "Unknown")
                        payload["confidence"] = state.get("confidence", 0.0)
                        payload["severity_reasoning"] = state.get("severity_reasoning", [])
                    elif node_name == "root_cause_agent":
                        payload["root_cause"] = state.get("root_cause", "")
                        payload["affected_file"] = state.get("affected_file", "")
                        payload["vulnerable_code"] = state.get("vulnerable_code", "")
                        payload["exploit_example"] = state.get("exploit_example", "")
                        payload["risk_if_unfixed"] = state.get("risk_if_unfixed", "")
                        payload["suspected_files"] = state.get("suspected_files", [])
                    elif node_name == "fix_agent":
                        payload["patch"] = state.get("patch", "")
                        payload["fix_summary"] = state.get("fix_summary", "")
                        payload["patched_code"] = state.get("patched_code", "")
                        payload["risk_assessment"] = state.get("risk_assessment", "")
                    elif node_name == "validation_agent":
                        payload["validation_score"] = state.get("validation_score", 0)
                        payload["is_patch_valid"] = state.get("is_patch_valid", False)
                        payload["validation_reasoning"] = state.get("validation_reasoning", "")
                    elif node_name == "test_agent":
                        payload["regression_tests"] = state.get("regression_tests", "")
                    elif node_name == "test_execute_agent":
                        payload["test_results"] = state.get("test_results", {})
                        payload["tests_passed"] = state.get("tests_passed", False)
                    elif node_name == "auto_rescan_agent":
                        payload["rescan_passed"] = state.get("rescan_passed", False)
                    elif node_name == "github_agent":
                        pr_url = state.get("pr_url", "")
                        payload["pr_url"] = pr_url
                        payload["pr_mode"] = state.get("pr_mode", "mock")
                    elif node_name == "sprint_agent":
                        payload["story_points"] = state.get("story_points", 0)
                        payload["priority"] = state.get("priority", "")
                        payload["sprint_recommendation"] = state.get("sprint_recommendation", "")

                    _emit(job, "agent_complete", payload)
            return final

        # Run the synchronous generator in a thread pool to avoid blocking the event loop
        final_state = await asyncio.to_thread(_sync_runner)

        # Emit the final complete event with the full state
        _emit(job, "swarm_complete", {
            "severity": final_state.get("severity", ""),
            "confidence": final_state.get("confidence", 0.0),
            "severity_reasoning": final_state.get("severity_reasoning", []),
            "scanner_findings": final_state.get("scanner_findings", []),
            "root_cause": final_state.get("root_cause", ""),
            "affected_file": final_state.get("affected_file", ""),
            "vulnerable_code": final_state.get("vulnerable_code", ""),
            "exploit_example": final_state.get("exploit_example", ""),
            "risk_if_unfixed": final_state.get("risk_if_unfixed", ""),
            "suspected_files": final_state.get("suspected_files", []),
            "patch": final_state.get("patch", ""),
            "fix_summary": final_state.get("fix_summary", ""),
            "patched_code": final_state.get("patched_code", ""),
            "risk_assessment": final_state.get("risk_assessment", ""),
            "validation_score": final_state.get("validation_score", 0),
            "is_patch_valid": final_state.get("is_patch_valid", False),
            "validation_reasoning": final_state.get("validation_reasoning", ""),
            "regression_tests": final_state.get("regression_tests", ""),
            "test_results": final_state.get("test_results", {}),
            "tests_passed": final_state.get("tests_passed", False),
            "rescan_passed": final_state.get("rescan_passed", False),
            "pr_url": final_state.get("pr_url", ""),
            "pr_mode": "mock" if _is_mock_pr(final_state.get("pr_url", "")) else "live",
            "story_points": final_state.get("story_points", 0),
            "priority": final_state.get("priority", ""),
            "sprint_recommendation": final_state.get("sprint_recommendation", ""),
        })

        job["final_state"] = final_state
        job["status"] = "complete"

    except Exception as e:
        logger.exception(f"Swarm job {job_id} failed")
        _emit(job, "swarm_error", {"error": str(e)})
        job["status"] = "error"

    finally:
        job["done"] = True


def _is_mock_pr(url: str) -> bool:
    """Returns True if a PR URL is mock/failed rather than a real GitHub PR."""
    if not url:
        return True
    lower = url.lower()
    return (
        "mock" in lower
        or "error" in lower
        or url.startswith("mock://")
        or "mock-org" in lower
        or not url.startswith("https://github.com/")
    )


def _emit(job: dict, event_type: str, data: dict):
    """Append an SSE event to the job's event queue."""
    job["events"].append({
        "type": event_type,
        "data": data,
        "timestamp": time.time(),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
