from langchain_mistralai import MistralAIEmbeddings


MISTRAL_API_KEY = "05M3e310"

def get_embeddings(texts: list[str]):
    embeddings = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model="mistral-embed"
    )
    return embeddings.embed_documents(texts)

