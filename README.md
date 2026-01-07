# AI Job Application Assistant

**Automated Job Discovery, Semantic Matching, and Application Intelligence**

---

## Overview

The **AI Job Application Assistant** is an end-to-end AI system designed to automate and optimize the job application workflow.
It intelligently discovers relevant job opportunities, semantically matches them against a candidate’s resume, generates personalized cover letters, and persists results for structured tracking.

The project demonstrates a **production-style agentic AI architecture**, combining **Retrieval-Augmented Generation (RAG)**, **LLM-driven reasoning**, and **real-world API integrations**.

---

## Problem Addressed

Modern job searching is inefficient due to:

* Fragmented job sources (emails, career pages)
* Manual resume relevance checks
* Repetitive cover letter writing
* Poor tracking of applications

This system reduces manual effort by **automating discovery, evaluation, and preparation**, while keeping the human in control.

---

## Key Capabilities

### Resume Intelligence

* Resume ingestion and semantic indexing
* Structured extraction of skills, experience level, role preferences, and location
* Contextual understanding using RAG

### Job Discovery

* Email-based job alerts (Gmail API)
* Public company career APIs (Greenhouse boards)
* Profile-driven query planning

### Matching & Ranking

* Embedding-based semantic similarity scoring
* Resume–job relevance computation
* Filtering, normalization, and deduplication
* Ranked job recommendations

### Application Assistance

* Personalized cover letter generation using LLMs
* Context-aware writing aligned to job descriptions

### Persistence & Tracking

* Automated storage of ranked jobs and generated cover letters
* Centralized tracking via Google Sheets

---

## High-Level Architecture

* **Frontend**: Streamlit UI for resume upload and execution
* **Backend**: FastAPI service layer
* **AI Orchestration**: LangGraph (state-driven agent execution)
* **LLMs & Embeddings**: Mistral AI
* **Retrieval Layer**: FAISS vector store
* **External Integrations**: Gmail API, Google Sheets API, Career APIs

The system is designed with **clear separation of concerns**, enabling scalability and future extensibility.

---

## Agent Node Flow

The job agent executes as a **deterministic state graph**, where each node is responsible for a single, well-defined task:

1. **Extract Candidate Profile**
   Parses and structures resume data using RAG.

2. **Build Job Search Plan**
   Generates realistic, high-signal search queries based on the candidate profile.

3. **Fetch Jobs**
   Collects job postings from email alerts and public career APIs.

4. **Parse Jobs**
   Normalizes job data and extracts relevant skills and metadata.

5. **Match Jobs**
   Computes semantic similarity between resume context and job descriptions.

6. **Rank Jobs**
   Filters and ranks opportunities based on relevance score.

7. **Generate Cover Letters**
   Produces personalized, role-specific cover letters using an LLM.

8. **Save Results**
   Persists ranked jobs and cover letters to Google Sheets.

---

## Project Structure

```
AI-Job-Application-Assistant/
│
├── backend/
│   ├── api.py                      # FastAPI entry point
│   │
│   ├── rag/
│   │   ├── ingest.py               # Resume ingestion & vectorization
│   │   ├── query.py                # RAG-based resume querying
│   │   └── embeddings.py           # Mistral embedding integration
│   │
│   ├── agent/
│   │   ├── job_agent_graph.py      # LangGraph definition
│   │   ├── test_gmail_fetch.py
│   │   |---gshee.py
│   │       
│   │      
│   │
│   └── utils/
│       ├── pdf_loader.py
│       └── text_splitter.py
│
├── frontend/
│   └── frontend01.py               # Streamlit UI
│
├── uploads/                        # Uploaded resumes (ignored)
├── vectors/                        # Vector store (ignored)
├── .gitignore
├── api.py
└── README.md


SYSTEM ARCHITECTURE 
Streamlit UI
     ↓
FastAPI Backend
     ↓
Resume Ingestion (RAG)
     ↓
LangGraph Job Agent
     ↓
Job Fetching (Email + APIs)
     ↓
Parsing & Skill Extraction
     ↓
Semantic Matching
     ↓
Ranking
     ↓
Cover Letter Generation
     ↓
Google Sheets Storage

```

---
## Technology Stack

**Languages & Frameworks**

* Python
* FastAPI
* Streamlit

**AI & ML**

* LangChain
* LangGraph
* Mistral AI (LLM & embeddings)
* FAISS
* Retrieval-Augmented Generation (RAG)

**Integrations**

* Gmail API
* Google Sheets API
* Greenhouse Career APIs

**Engineering Practices**

* Typed state management
* Modular node-based design
* Secure secret handling
* API-first architecture

---
## Use Cases

* AI-driven job search assistants
* Resume–job matching systems
* Agentic workflow experimentation
* Portfolio-grade AI engineering projects
* Career automation research

---

## Author

**Umer Rafiq**
B.Tech (Computer Science & Engineering)

🔗 GitHub:
[https://github.com/umerrafiq04/AI-Job-Application-Assistant](https://github.com/umerrafiq04/AI-Job-Application-Assistant)



