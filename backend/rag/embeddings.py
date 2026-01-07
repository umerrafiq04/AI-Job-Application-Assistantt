from langchain_mistralai import MistralAIEmbeddings

# ⚠️ Hard-coded for now (OK for local testing)
MISTRAL_API_KEY = "05M3e310UpAluqszzJMayblJMIvViXqf"

def get_embeddings(texts: list[str]):
    embeddings = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model="mistral-embed"
    )
    return embeddings.embed_documents(texts)

