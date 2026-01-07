import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/run-agent"

st.set_page_config(page_title="AI Job Agent", layout="wide")

st.title("🤖 AI Job Application Agent")
st.caption("Upload resume → find jobs → generate cover letters")

st.divider()

# ----------------------------
# Upload Resume
# ----------------------------
resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

run_button = st.button("🚀 Run Job Agent")

# ----------------------------
# Run Agent
# ----------------------------
if run_button:
    if resume_file is None:
        st.warning("Please upload a resume first.")
    else:
        with st.spinner("Running agent (this may take 30–90 seconds)..."):
            files = {
                "resume": (
                    resume_file.name,
                    resume_file.getvalue(),
                    "application/pdf"
                )
            }

            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error("Agent failed to run.")
            st.text(response.text)
        else:
            data = response.json()

            st.success(f"Found {data['total_jobs']} matching jobs")

            for idx, job in enumerate(data["ranked_jobs"], start=1):
                with st.expander(f"{idx}. {job['title']} @ {job['company']} ({job['match_score']}%)"):
                    st.markdown(f"**Location:** {job.get('location','')}")
                    st.markdown(f"**Source:** {job.get('source','')}")
                    st.markdown(f"[Apply Link]({job.get('apply_url','')})")

                    st.subheader("Cover Letter")
                    cover_letter = data["cover_letters"][idx - 1].get("cover_letter", "")
                    st.text_area(
                        label="",
                        value=cover_letter,
                        height=220
                    )
