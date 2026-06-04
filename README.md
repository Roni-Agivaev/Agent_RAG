# Medium Article RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about a corpus of ~7,600 English Medium articles. Built for the Technion Individual RAG Assignment.

## Live Demo

**API Base URL:** `https://agent-rag-delta.vercel.app`

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python) |
| Deployment | Vercel |
| Vector DB | Pinecone (cosine, 1536 dims) |
| Embeddings | `4UHRUIN-text-embedding-3-small` |
| LLM | `4UHRUIN-gpt-5-mini` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (token-based via tiktoken) |

## API Endpoints

### `POST /api/prompt`
Submit a natural language question. The system embeds the question, retrieves the most relevant article chunks from Pinecone, deduplicates by article, and generates an answer grounded strictly in the retrieved context.

**Request:**
```json
{
  "question": "Find an article about building better habits"
}
```

**Response:**
```json
{
  "response": "Natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "Retrieved article passage",
      "score": 0.8523
    }
  ],
  "Augmented_prompt": {
    "System": "System prompt used",
    "User": "Full user prompt with retrieved context"
  }
}
```

### `GET /api/stats`
Returns the current RAG hyperparameters.

**Response:**
```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 7
}
```

## RAG Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `chunk_size` | 512 tokens | Token-based via `cl100k_base` encoding |
| `overlap_ratio` | 0.2 (20%) | 100 token overlap between chunks |
| `top_k` | 7 | Pinecone retrieves top 7 chunks, then deduplicated to distinct articles |

## Key Design Decisions

- **Token-based chunking** — uses `tiktoken` (`cl100k_base`) so chunk sizes are measured in real tokens, matching the embedding model's context window
- **Per-article deduplication** — after retrieval, only the highest-scoring chunk per article is kept, ensuring the LLM sees diverse sources rather than multiple chunks from the same article
- **Strict grounding** — the system prompt forbids the model from using any external knowledge; if the answer isn't in the retrieved context, it responds: *"I don't know based on the provided Medium articles data."*

## Dataset

~7,600 English articles from the [Medium Articles dataset](https://www.kaggle.com/datasets/fabiochiusano/medium-articles-dataset).

CSV schema: `title, text, url, authors, timestamp, tags`

## Local Setup

1. Clone the repo and create a virtual environment:
```bash
git clone https://github.com/Roni-Agivaev/Agent_RAG.git
cd Agent_RAG
python -m venv .venv
.venv\Scripts\activate      # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install pandas langchain-text-splitters tiktoken  # for indexing script
```

3. Create a `.env` file:
```
LLMOD_API_KEY=your_llmod_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

4. Index the dataset (one-time):
```bash
python scripts/index_data.py
```

5. Run locally:
```bash
uvicorn api.index:app --reload
```

## Project Structure

```
├── api/
│   └── index.py          # FastAPI app — /api/prompt and /api/stats endpoints
├── scripts/
│   └── index_data.py     # One-time script to chunk, embed, and upsert to Pinecone
├── config.py             # Shared config and LLM/embedding client helpers
├── requirements.txt      # Vercel runtime dependencies
└── vercel.json           # Vercel deployment config
```
