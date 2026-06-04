import ast
import sys
import os
import time

import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
# Add parent dir to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    get_embeddings,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def parse_list_field(value: str) -> list:
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() >= 100].reset_index(drop=True)

    df["authors"] = df["authors"].apply(
        lambda v: ", ".join(parse_list_field(str(v))) if pd.notna(v) else ""
    )
    df["tags"] = df["tags"].apply(
        lambda v: ", ".join(parse_list_field(str(v))) if pd.notna(v) else ""
    )
    df["article_id"] = df.index.astype(str)

    return df


def build_embed_text(row) -> str:
    return f"Title: {row['title']}\nAuthors: {row['authors']}\nTags: {row['tags']}\n\n{row['text']}"


def chunk_articles(df: pd.DataFrame):
    enc = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda text: len(enc.encode(text)),
    )

    records = []
    for _, row in df.iterrows():
        full_text = build_embed_text(row)
        chunks = splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"{row['article_id']}_{i}",
                    "article_id": row["article_id"],
                    "title": str(row.get("title", "")),
                    "authors": str(row.get("authors", "")),
                    "tags": str(row.get("tags", "")),
                    "url": str(row.get("url", "")),
                    "timestamp": str(row.get("timestamp", "")),
                    "chunk": chunk,
                }
            )
    return records


def ensure_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        # Clear all existing vectors so old chunks don't mix with new ones
        print("  Clearing existing vectors from index...")
        pc.Index(PINECONE_INDEX_NAME).delete(delete_all=True)
    return pc.Index(PINECONE_INDEX_NAME)


def upsert_in_batches(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted {min(i + batch_size, len(vectors))} / {len(vectors)}")


def main():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medium-english-50mb.csv")

    print("Loading CSV...")
    df = load_and_clean(csv_path)
    print(f"  {len(df)} articles after filtering")

    print("Chunking...")
    records = chunk_articles(df)
    print(f"  {len(records)} chunks total")

    print("Embedding...")
    embeddings = get_embeddings()
    texts = [r["chunk"] for r in records]
    vectors_raw = embeddings.embed_documents(texts)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = ensure_index(pc)

    print("Upserting to Pinecone...")
    pinecone_vectors = [
        {
            "id": rec["chunk_id"],
            "values": vec,
            "metadata": {
                "article_id": rec["article_id"],
                "title": rec["title"],
                "authors": rec["authors"],
                "tags": rec["tags"],
                "url": rec["url"],
                "timestamp": rec["timestamp"],
                "chunk": rec["chunk"],
            },
        }
        for rec, vec in zip(records, vectors_raw)
    ]
    upsert_in_batches(index, pinecone_vectors)

    print("Done.")


if __name__ == "__main__":
    main()
