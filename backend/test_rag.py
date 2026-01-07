# backend/test_rag.py
from backend.rag.ingest import ingest_resume
from backend.rag.query import query_resume

ingest_resume(r"\Users\user\Desktop\LG\JOB\backend\resume.pdf")

results = query_resume("What skills does the candidate have?")
print(results)
