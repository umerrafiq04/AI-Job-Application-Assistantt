from backend.agent.job_agent_graph import fetch_jobs_from_email

jobs =fetch_jobs_from_email("LinkedIn")

print(f"Found {len(jobs)} job emails:\n")

for i, job in enumerate(jobs, start=1):
    print(f"{i}. Source: {job['source']}")
    print(f"   Title: {job['title']}")
    print(f"   Company: {job['company']}")
    print(f"   Preview: {job['description'][:200]}")
    print("-" * 50)
