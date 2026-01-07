import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.rag.ingest import ingest_resume
from backend.agent.job_agent_graph import build_job_agent_graph


# ----------------------------
# App setup
# ----------------------------
app = FastAPI(title="AI Job Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Build graph ONCE (important for speed)
job_graph = build_job_agent_graph()


# ----------------------------
# Run Agent Endpoint
# ----------------------------
@app.post("/run-agent")
async def run_agent(resume: UploadFile = File(...)):
    """
    1. Save resume
    2. Run RAG ingestion
    3. Invoke LangGraph job agent
    """

    # 1️⃣ Save resume with unique name
    resume_id = str(uuid.uuid4())
    resume_path = os.path.join(UPLOAD_DIR, f"{resume_id}_{resume.filename}")

    with open(resume_path, "wb") as f:
        f.write(await resume.read())

    # 2️⃣ RAG ingestion
    ingest_resume(resume_path)

    # 3️⃣ Initial state (minimal)
    initial_state = {
        "skills": [],
        "experience_level": "",
        "preferred_roles": [],
        "location": "",
    }

    # 4️⃣ Run agent graph (blocking by design)
    final_state = job_graph.invoke(initial_state)

    return {
        "status": "success",
        "total_jobs": len(final_state.get("ranked_jobs", [])),
        "ranked_jobs": final_state.get("ranked_jobs", []),
        "cover_letters": final_state.get("cover_letters", []),
    }
