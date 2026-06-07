import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
LLMOD_API_KEY    = os.getenv("LLMOD_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLMOD_BASE_URL   = "https://api.llmod.ai/v1"

EMBEDDING_MODEL   = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL        = "4UHRUIN-gpt-5-mini"
PINECONE_INDEX    = "medium-rag"

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 100
TOP_K         = 8

# ── helpers ───────────────────────────────────────────────────────────────────
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

def get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI()

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    "'I don't know based on the provided Medium articles data.' "
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful."
)

class PromptRequest(BaseModel):
    question: str = Field(example="Your natural language question here")

@app.post("/api/prompt")
def prompt(req: PromptRequest):
    # 0. validate input
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="'question' must be a non-empty string.")

    try:
        # 1. embed question
        question_vec = get_embeddings().embed_query(req.question)

        # 2. retrieve from Pinecone
        results = get_index().query(vector=question_vec, top_k=TOP_K, include_metadata=True)

        # 3. deduplicate — keep highest-scoring chunk per article
        seen: dict = {}
        for match in results.matches:
            aid = match.metadata["article_id"]
            if aid not in seen or match.score > seen[aid].score:
                seen[aid] = match

        deduped = sorted(seen.values(), key=lambda m: m.score, reverse=True)

        # 4. build augmented prompt
        context_parts = []
        for match in deduped:
            m = match.metadata
            # Strip the prepended "Title:/Authors:/Tags:" header from the chunk
            # (added by build_embed_text during indexing) to avoid duplication
            chunk_text = m["chunk"]
            if chunk_text.startswith("Title:"):
                lines = chunk_text.splitlines()
                # Skip header lines until the first blank line
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        chunk_text = "\n".join(lines[i + 1:]).strip()
                        break
            context_parts.append(
                f"[Article: {m['title']}]\nAuthors: {m['authors']}\nTags: {m['tags']}\n{chunk_text}"
            )
        context_str = "\n\n".join(context_parts)
        user_prompt  = f"{context_str}\n\nQuestion: {req.question}"

        # 5. call LLM
        chain    = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")]) | get_llm()
        response = chain.invoke({"system": SYSTEM_PROMPT, "user": user_prompt})

        # 6. return
        return {
            "response": response.content,
            "context": [
                {
                    "article_id": m.metadata["article_id"],
                    "title":      m.metadata["title"],
                    "chunk":      m.metadata["chunk"],
                    "score":      m.score,
                }
                for m in deduped
            ],
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User":   user_prompt,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/stats")
def stats():
    return {
        "chunk_size":    CHUNK_SIZE,
        "overlap_ratio": round(CHUNK_OVERLAP / CHUNK_SIZE, 2),
        "top_k":         TOP_K,
    }
