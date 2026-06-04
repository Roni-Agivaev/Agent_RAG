import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
EMBEDDING_DIMENSION = 1536

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K = 7

PINECONE_INDEX_NAME = "medium-rag"


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=LLMOD_API_KEY,
        openai_api_base=LLMOD_BASE_URL,
        chunk_size=256,
    )


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=LLMOD_API_KEY,
        openai_api_base=LLMOD_BASE_URL,
        temperature=1,
    )
