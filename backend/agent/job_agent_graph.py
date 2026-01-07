from typing import List, Dict, Any
from typing_extensions import TypedDict
import numpy as np
from langgraph.graph import StateGraph, END

# =========================================================
# STATE DEFINITION
# =========================================================

class JobAgentState(TypedDict):
    # Candidate profile (from RAG)
    skills: List[str]
    experience_level: str
    preferred_roles: List[str]
    location: str

    # Job search planning
    search_queries: List[str]

    # Job data
    raw_jobs: List[Dict[str, Any]]
    parsed_jobs: List[Dict[str, Any]]
    scored_jobs: List[Dict[str, Any]]
    ranked_jobs: List[Dict[str, Any]]

    # Output
    cover_letters: Dict[str, str]


# =========================================================
# =========================================================
# node 1
from pydantic import BaseModel
from typing import List
class CandidateProfile(BaseModel):
    skills: List[str]
    experience_level: str
    preferred_roles: List[str]
    location: str
from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(
    api_key="05M3e310UpAluqszzJMayblJMIvViXqf",
    model="mistral-small-latest"
)


from backend.rag.query import query_resume

def extract_candidate_profile(state: JobAgentState) -> JobAgentState:
    """
    Node 1:
    - Query resume vector DB
    - Use Mistral LLM with structured output
    - Populate candidate profile
    """

    # 1. Retrieve resume context
    resume_chunks = query_resume(
        "Extract skills, experience level, preferred roles, and location from this resume"
    )

    resume_context = "\n".join(resume_chunks)

    # 2. Structured LLM call
    structured_llm = llm.with_structured_output(CandidateProfile)

    profile: CandidateProfile = structured_llm.invoke(
        f"""
        You are an expert resume analyzer.

        Resume:
        --------
        {resume_context}

        Instructions:
        - Extract technical skills only
        - Infer experience level (Fresher / Intern / 0-1 / 1-3 / 3+)
        - Infer preferred job roles
        - Infer location if mentioned, else "Not specified"
        """
    )
    print("NODE:1")
    for key, value in state.items():
        print(f"{key} : {value}")

    # 3. Update agent state
    return {
        **state,
        "skills": profile.skills,
        "experience_level": profile.experience_level,
        "preferred_roles": profile.preferred_roles,
        "location": profile.location,
    }

# node 2
# -----------------------------------
# Node 2: Build Job Search Plan (Soft)
# -----------------------------------

def build_job_search_plan(state: JobAgentState) -> JobAgentState:
    """
    Node 2:
    Build SIMPLE, HIGH-PROBABILITY Gmail search queries
    (no location, no hard filters)
    """

    roles = state.get("preferred_roles", [])
    experience = state.get("experience_level", "").lower()

    search_queries = []

    # Normalize experience
    if "intern" in experience or "fresher" in experience:
        exp_keywords = ["Intern", "Internship"]
    else:
        exp_keywords = ["Junior"]

    # 🔹 Common role aliases (important for inbox matching)
    ROLE_ALIASES = {
        "Software Engineer": ["Software Engineer", "Software Developer"],
        "Web Developer": ["Web Developer", "Frontend Developer"],
        "Data Scientist": ["Data Scientist", "Data Analyst"],
        "Machine Learning Engineer": ["Machine Learning", "ML Engineer"],
        "AI/ML Engineer": ["AI Engineer", "ML Engineer"]
    }

    for role in roles:
        aliases = ROLE_ALIASES.get(role, [role])

        for alias in aliases:
            for exp in exp_keywords:
                search_queries.append(f"{alias} {exp}")

    # 🔹 Hard fallback (guaranteed inbox matches)
    search_queries.extend([
        "Software Intern",
        "Internship",
        "Hiring Intern"
    ])

    # 🔹 Deduplicate + LIMIT (very important)
    search_queries = list(dict.fromkeys(search_queries))[:3]

    print("✅ NODE 2 | Search Queries Generated:")
    for q in search_queries:
        print("  -", q)

    return {
        **state,
        "search_queries": search_queries
    }

# # node 3//////////--------------------
# import os
# import base64
# import re
# from email import message_from_bytes
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# import pickle

# SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# import os
# import pickle
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from google.auth.transport.requests import Request

# SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# BASE_DIR = os.path.dirname(__file__)
# TOKEN_PATH = os.path.join(BASE_DIR, "token.pickle")
# CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")

# def get_gmail_service():
#     creds = None

#     if os.path.exists(TOKEN_PATH):
#         with open(TOKEN_PATH, "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 CREDS_PATH, SCOPES
#             )
#             creds = flow.run_local_server(port=0)

#         with open(TOKEN_PATH, "wb") as token:
#             pickle.dump(creds, token)

#     return build("gmail", "v1", credentials=creds)

# def fetch_jobs_from_email(search_query: str):
#     """
#     Fetch job postings from Gmail job alert emails
#     """
#     service = get_gmail_service()

#     # Search job alert emails
#     query = f"{search_query} subject:(job OR opening OR hiring)"
#     try:
#         results = service.users().messages().list(
#             userId="me",
#             q=query,
#             maxResults=5
#         ).execute(num_retries=3)
#     except Exception as e:
#         print("⚠️ Gmail API timeout or network error:", e)
#         return []


#     messages = results.get("messages", [])
#     jobs = []

#     for msg in messages:
#         msg_data = service.users().messages().get(
#             userId="me",
#             id=msg["id"],
#             format="raw"
#         ).execute()

#         raw_msg = base64.urlsafe_b64decode(msg_data["raw"])
#         email_msg = message_from_bytes(raw_msg)

#         body = ""
#         if email_msg.is_multipart():
#             for part in email_msg.walk():
#                 if part.get_content_type() == "text/plain":
#                     body += part.get_payload(decode=True).decode(errors="ignore")
#         else:
#             body = email_msg.get_payload(decode=True).decode(errors="ignore")

#         jobs.append({
#             "source": "email",
#             "query": search_query,
#             "title": search_query,
#             "company": "Unknown (Email Alert)",
#             "location": "Not specified",
#             "description": body[:1000]
#         })

#     return jobs
# # company api
# import requests

# COMPANY_APIS = {
#     "stripe": "https://boards-api.greenhouse.io/v1/boards/stripe/jobs",
#     "airbnb": "https://boards-api.greenhouse.io/v1/boards/airbnb/jobs",
# }

# def fetch_jobs_from_career_api(search_query: str):
#     jobs = []
#     query_words = search_query.lower().split()

#     for company, url in COMPANY_APIS.items():
#         try:
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()
#             data = response.json()
#         except Exception as e:
#             print(f"⚠️ Career API error for {company}: {e}")
#             continue

#         for job in data.get("jobs", []):
#             title = job.get("title", "").lower()

#             if any(word in title for word in query_words):
#                 description = job.get("content", "")
#                 description = description.replace("\n", " ").strip()

#                 jobs.append({
#                     "source": "career_api",
#                     "query": search_query,
#                     "title": job.get("title"),
#                     "company": company.capitalize(),
#                     "location": job.get("location", {}).get("name", "Not specified"),
#                     "description": description[:1000],
#                     "apply_url": job.get("absolute_url")
#                 })

#     return jobs

# # /// final
# def fetch_jobs(state: JobAgentState) -> JobAgentState:
#     """
#     Node 3:
#     - Fetch jobs from Gmail + Career APIs
#     - Merge & deduplicate
#     """

#     queries = state.get("search_queries", [])
#     all_jobs = []

#     for query in queries:
#         all_jobs.extend(fetch_jobs_from_email(query))
#         all_jobs.extend(fetch_jobs_from_career_api(query))

#     # Deduplicate
#     seen = set()
#     unique_jobs = []

#     for job in all_jobs:
#         key = (job.get("title"), job.get("company"))
#         if key not in seen:
#             seen.add(key)
#             unique_jobs.append(job)
#     print("NODE:3")
#     for key, value in state.items():
#         print(f"{key} : {value}")

#     return {
#         **state,
#         "raw_jobs": unique_jobs
#     }


# -------------------------------
# Node 3: Fetch Jobs (EMAIL ONLY)
# -------------------------------

import os
import base64
import pickle
from email import message_from_bytes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from typing import Dict, Any, List
# from backend.agent.state import JobAgentState   # adjust import if needed

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

BASE_DIR = os.path.dirname(__file__)
TOKEN_PATH = os.path.join(BASE_DIR, "token.pickle")
CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")


def get_gmail_service():
    creds = None

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    return build("gmail", "v1", credentials=creds)


def fetch_jobs_from_email(search_query: str) -> List[Dict[str, Any]]:
    """
    Fetch job postings from Gmail job alert emails
    """
    service = get_gmail_service()

    query = f'{search_query} subject:(job OR opening OR hiring)'

    try:
        results = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=5
        ).execute(num_retries=2)
    except Exception as e:
        print("⚠️ Gmail fetch failed:", e)
        return []

    messages = results.get("messages", [])
    jobs = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="raw"
        ).execute()

        raw_msg = base64.urlsafe_b64decode(msg_data["raw"])
        email_msg = message_from_bytes(raw_msg)

        body = ""
        if email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = email_msg.get_payload(decode=True).decode(errors="ignore")

        jobs.append({
            "source": "email",
            "query": search_query,
            "title": search_query,
            "company": "Unknown (Email Alert)",
            "location": "Not specified",
            "description": body[:1000],
            "apply_url": ""
        })

    return jobs


def fetch_jobs(state: JobAgentState) -> JobAgentState:
    """
    Node 3:
    - Fetch jobs ONLY from Gmail
    - Deduplicate
    """

    queries = state.get("search_queries", [])[:2]  # HARD LIMIT
    all_jobs = []

    for query in queries:
        all_jobs.extend(fetch_jobs_from_email(query))

    # Deduplicate
    seen = set()
    unique_jobs = []

    for job in all_jobs:
        key = (job["title"], job["company"])
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    print("✅ Node 3 completed | Jobs fetched:", len(unique_jobs))
    for key, value in state.items():
        print(f"{key} : {value}")
    return {
        **state,
        "raw_jobs": unique_jobs[:5]  # HARD LIMIT
    }

# # ////----------------------

# node 4
# node 4
import re
from typing import Dict, List, TypedDict

from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------------
# LangGraph State
# -----------------------------------
class JobAgentState(TypedDict, total=False):
    raw_jobs: List[Dict]
    parsed_jobs: List[Dict]


# -----------------------------------
# LLM setup (lightweight, cheap)
# -----------------------------------
llm = ChatMistralAI(
    api_key="05M3e310UpAluqszzJMayblJMIvViXqf",
    model="mistral-small-latest"
)
skill_parser = JsonOutputParser()

skill_prompt = ChatPromptTemplate.from_template(
    """
You are an expert technical recruiter.

Extract ONLY technical skills from the job description below.

Rules:
- Return a JSON list of skills
- Normalize names (e.g. "PyTorch Lightning" → "PyTorch")
- No explanations
- No duplicates
- Use lowercase

Job Description:
{description}
"""
)


# -----------------------------------
# Node 4: Parse & Enrich Jobs
# -----------------------------------
def parse_jobs(state: JobAgentState) -> JobAgentState:
    """
    Node 4:
    - Normalize job fields
    - Use LLM to extract & normalize skills
    """

    raw_jobs = state.get("raw_jobs", [])
    parsed_jobs = []

    for job in raw_jobs:
        description = (job.get("description") or "").strip()

        # ---- LLM skill extraction ----
        try:
            chain = skill_prompt | llm | skill_parser
            skills = chain.invoke({"description": description})

            # Ensure list[str]
            if not isinstance(skills, list):
                skills = []

        except Exception as e:
            print("⚠️ Skill extraction failed:", e)
            skills = []

        parsed_jobs.append({
            "title": job.get("title", "").strip(),
            "company": job.get("company", "").strip(),
            "location": job.get("location", "Not specified"),
            "skills": sorted(set(skills)),
            "description": description,
            "apply_url": job.get("apply_url", ""),
            "source": job.get("source", ""),
            "query": job.get("query", "")
        })
    print("NODE:4")
    for key, value in state.items():
        print(f"{key} : {value}")

    return {
        **state,
        "parsed_jobs": parsed_jobs
    }

# node 5import numpy as np
from typing import Dict, List, TypedDict

from backend.rag.embeddings import get_embeddings
from backend.rag.query import query_resume


# -----------------------------
# LangGraph State
# -----------------------------
class JobAgentState(TypedDict, total=False):
    parsed_jobs: List[Dict]
    scored_jobs: List[Dict]


# -----------------------------
# Utility: cosine similarity
# -----------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -----------------------------
# Node 5: Match Jobs
# -----------------------------
def match_jobs(state: JobAgentState) -> JobAgentState:
    """
    Node 5:
    - Build resume semantic context
    - Match resume with parsed jobs
    - Compute semantic similarity scores
    """

    parsed_jobs = state.get("parsed_jobs", [])
    scored_jobs = []

    if not parsed_jobs:
        return {**state, "scored_jobs": []}

    # ---------------------------------
    # 1. Build resume semantic context
    # ---------------------------------
    resume_chunks = query_resume(
        "Summarize the candidate's skills, experience, and strengths"
    )

    if not resume_chunks:
        resume_text = ""
    else:
        resume_text = " ".join(resume_chunks)

    resume_embedding = get_embeddings([resume_text])[0]
    resume_embedding = np.array(resume_embedding, dtype=np.float32)

    # ---------------------------------
    # 2. Score each job
    # ---------------------------------
    for job in parsed_jobs:
        job_text = f"{job.get('title', '')}. {job.get('description', '')}"

        job_embedding = get_embeddings([job_text])[0]
        job_embedding = np.array(job_embedding, dtype=np.float32)

        similarity = cosine_similarity(resume_embedding, job_embedding)

        scored_jobs.append({
            **job,
            "match_score": round(similarity * 100, 2)
        })

    # ---------------------------------
    # 3. Sort by relevance
    # ---------------------------------
    scored_jobs.sort(
        key=lambda x: x["match_score"],
        reverse=True
    )
    print("NODE:5")
    for key, value in state.items():
        print(f"{key} : {value}")

    return {
        **state,
        "scored_jobs": scored_jobs
    }


# node 6
from typing import Dict, List, TypedDict


# -----------------------------
# LangGraph State
# -----------------------------
class JobAgentState(TypedDict, total=False):
    scored_jobs: List[Dict]
    ranked_jobs: List[Dict]


# -----------------------------
# Node 6: Rank Jobs
# -----------------------------
def rank_jobs(state: JobAgentState) -> JobAgentState:
    """
    Node 6:
    - Filter low-relevance jobs
    - Rank jobs by semantic match score
    - Select top-N jobs
    """

    scored_jobs = state.get("scored_jobs", [])

    # -----------------------------
    # Configuration (easy to tune)
    # -----------------------------
    MIN_MATCH_SCORE = 60.0   # minimum relevance (%)
    TOP_N = 10               # max jobs to keep

    if not scored_jobs:
        return {
            **state,
            "ranked_jobs": []
        }

    # -----------------------------
    # 1. Filter by minimum score
    # -----------------------------
    filtered_jobs = [
        job for job in scored_jobs
        if job.get("match_score", 0.0) >= MIN_MATCH_SCORE
    ]

    # -----------------------------
    # 2. Sort by score (descending)
    # -----------------------------
    ranked_jobs = sorted(
        filtered_jobs,
        key=lambda x: x.get("match_score", 0.0),
        reverse=True
    )

    # -----------------------------
    # 3. Keep top-N
    # -----------------------------
    ranked_jobs = ranked_jobs[:TOP_N]
    for key, value in state.items():
        print(f"{key} : {value}")

    return {
        **state,
        "ranked_jobs": ranked_jobs
    }



# node 7
from typing import Dict, List, TypedDict

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.rag.query import query_resume


# -----------------------------------
# LangGraph State
# -----------------------------------
class JobAgentState(TypedDict, total=False):
    ranked_jobs: List[Dict]
    cover_letters: List[Dict]


# -----------------------------------
# LLM setup
# -----------------------------------


parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
You are an expert career assistant.

Write a professional, concise, and personalized cover letter.

Guidelines:
- 3 short paragraphs
- Professional tone
- No emojis
- No placeholders
- No markdown
- No address header
- Focus on skills match and motivation

Candidate Resume Summary:
{resume_summary}

Job Title: {job_title}
Company: {company}
Job Description:
{job_description}

Cover Letter:
"""
)


# -----------------------------------
# Node 7: Generate Cover Letters
# -----------------------------------
def generate_cover_letters(state: JobAgentState) -> JobAgentState:
    """
    Node 7:
    - Generate personalized cover letters for ranked jobs
    """

    ranked_jobs = state.get("ranked_jobs", [])
    cover_letters = []

    if not ranked_jobs:
        return {**state, "cover_letters": []}

    # ---------------------------------
    # 1. Build resume summary once
    # ---------------------------------
    resume_chunks = query_resume(
        "Summarize the candidate's skills, experience, and strengths in 5–6 lines"
    )

    resume_summary = " ".join(resume_chunks) if resume_chunks else ""

    # ---------------------------------
    # 2. Generate cover letter per job
    # ---------------------------------
    for job in ranked_jobs:
        try:
            chain = prompt | llm | parser

            letter = chain.invoke({
                "resume_summary": resume_summary,
                "job_title": job.get("title", ""),
                "company": job.get("company", ""),
                "job_description": job.get("description", "")
            })

        except Exception as e:
            print("⚠️ Cover letter generation failed:", e)
            letter = ""

        cover_letters.append({
            **job,
            "cover_letter": letter.strip()
        })
    print("NODE:7")
    for key, value in state.items():
        print(f"{key} : {value}")

    return {
        **state,
        "cover_letters": cover_letters
    }



# node 8
# pip install gspread google-auth
import gspread
from typing import Dict, List, TypedDict
from google.oauth2.service_account import Credentials


# -----------------------------------
# LangGraph State
# -----------------------------------
class JobAgentState(TypedDict, total=False):
    ranked_jobs: List[Dict]
    cover_letters: List[Dict]


# -----------------------------------
# Google Sheets Client
# -----------------------------------
def get_gsheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_file(
        "backend/service_account.json",
        scopes=scopes
    )

    return gspread.authorize(creds)


# -----------------------------------
# Node 8: Save Results
# -----------------------------------
def save_results(state: JobAgentState) -> JobAgentState:
    """
    Node 8:
    - Save ranked jobs + cover letters to Google Sheets
    """

    ranked_jobs = state.get("ranked_jobs", [])
    cover_letters = state.get("cover_letters", [])

    if not ranked_jobs:
        print("⚠️ No jobs to save")
        return state

    client = get_gsheet_client()

    # Open or create sheet
    try:
        sheet = client.open("AI Job Applications").sheet1
    except gspread.SpreadsheetNotFound:
        sheet = client.create("AI Job Applications").sheet1

    # Header
    headers = [
        "Company",
        "Role",
        "Location",
        "Match Score",
        "Source",
        "Apply URL",
        "Cover Letter"
    ]

    existing_headers = sheet.row_values(1)
    if existing_headers != headers:
        sheet.insert_row(headers, index=1)

    # Append rows
    for job in cover_letters:
        row = [
            job.get("company", ""),
            job.get("title", ""),
            job.get("location", ""),
            job.get("match_score", ""),
            job.get("source", ""),
            job.get("apply_url", ""),
            job.get("cover_letter", "")
        ]

        sheet.append_row(row, value_input_option="RAW")

    print("✅ Results saved to Google Sheets")
    print("NODE:8")
    return state



# =========================================================
# LANGGRAPH DEFINITION
# =========================================================

def build_job_agent_graph():
    graph = StateGraph(JobAgentState)

    # -------- Nodes --------
    graph.add_node("extract_candidate_profile", extract_candidate_profile)
    graph.add_node("build_job_search_plan", build_job_search_plan)
    graph.add_node("fetch_jobs", fetch_jobs)
    graph.add_node("parse_jobs", parse_jobs)
    graph.add_node("match_jobs", match_jobs)
    graph.add_node("rank_jobs", rank_jobs)
    graph.add_node("generate_cover_letters", generate_cover_letters)
    graph.add_node("save_results", save_results)

    # -------- Edges --------
    graph.set_entry_point("extract_candidate_profile")

    graph.add_edge("extract_candidate_profile", "build_job_search_plan")
    graph.add_edge("build_job_search_plan", "fetch_jobs")
    graph.add_edge("fetch_jobs", "parse_jobs")
    graph.add_edge("parse_jobs", "match_jobs")
    graph.add_edge("match_jobs", "rank_jobs")
    graph.add_edge("rank_jobs", "generate_cover_letters")
    graph.add_edge("generate_cover_letters", "save_results")
    graph.add_edge("save_results", END)

    return graph.compile()


# =========================================================
# OPTIONAL: VISUALIZE GRAPH
# =========================================================

if __name__ == "__main__":
    graph = build_job_agent_graph()
    from backend.rag.ingest import ingest_resume
    from backend.rag.query import query_resume

    ingest_resume(r"\Users\user\Desktop\LG\JOB\backend\resume.pdf")
    initial_state = {
    "skills": [],
    "experience_level": "",
    "preferred_roles": [],
    "location": ""
}

    final_state = graph.invoke(initial_state)
    print(final_state.keys())



