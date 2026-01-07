import faiss
import pickle
import numpy as np

from backend.rag.embeddings import get_embeddings

VECTORSTORE_PATH = "vectorstore/resume_index"


def query_resume(query: str, k: int = 4):
    """
    Retrieve relevant resume chunks from vector store
    """

    # 1. Load index
    index = faiss.read_index(f"{VECTORSTORE_PATH}/index.faiss")

    # 2. Load documents
    with open(f"{VECTORSTORE_PATH}/docs.pkl", "rb") as f:
        documents = pickle.load(f)

    # 3. Get query embedding (list → numpy)
    query_embedding = get_embeddings([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # 4. Search
    distances, indices = index.search(query_embedding, k)

    # 5. Return matching chunks
    return [documents[i] for i in indices[0]]
