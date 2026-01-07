import os
import faiss
import pickle
import numpy as np
from backend.utils.pdf_loader import load_pdf_text
from backend.utils.text_splitter import split_text
from backend.rag.embeddings import get_embeddings

VECTORSTORE_PATH = "vectorstore/resume_index"


def ingest_resume(pdf_path: str):
    text = load_pdf_text(pdf_path)
    chunks = split_text(text)

    embeddings = get_embeddings(chunks)

    # ✅ FIX HERE
    dim = len(embeddings[0])

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    with open(f"{VECTORSTORE_PATH}/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(index, f"{VECTORSTORE_PATH}/index.faiss")

    print("✅ Resume ingested successfully")
