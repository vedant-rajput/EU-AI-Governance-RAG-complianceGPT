import json
import os
import sys
import uuid

# Resolve project root so imports work correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.logger import get_logger

logger = get_logger(__name__)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: Load from corpus.json
def load_corpus(filepath="data/corpus.json"):
    with open(filepath) as f:
        return json.load(f)

# Step 2: Chunk with metadata
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def chunk_corpus(corpus, chunk_size=500, overlap=100):
    chunks = []
    for entry in corpus:
        text_chunks = chunk_text(entry["text"], chunk_size, overlap)
        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "page": entry["page"],
                "source": entry["source"]
            })
    return chunks

# Step 3: Embed chunks
def get_embeddings(texts):
    if not texts:
        return []

    response = client.models.embed_content(
        model="gemini-embedding-001", 
        contents=texts,
    )
    # Bypass OpenAI specific cost logger
    logger.info("Embedded chunk batch with Google Gemini.")
    return [item.values for item in response.embeddings]

import time

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = []
    # Batch process in sizes of 50
    for i in range(0, len(texts), 50):
        print(f"Embedding chunks {i} to {min(i+50, len(texts))}...")
        for attempt in range(5):
            try:
                embeddings.extend(get_embeddings(texts[i:i+50]))
                time.sleep(1)  # small pause to help pacing
                break
            except Exception as e:
                logger.warning(f"Rate limit hit ({e}). Sleeping 40 seconds...")
                time.sleep(40)
    return embeddings

# Step 4: Build and save Qdrant index
def main():
    logger.info("Loading corpus...")
    corpus = load_corpus()

    logger.info("Chunking corpus...")
    chunks = chunk_corpus(corpus)
    logger.info(f"Total chunks: {len(chunks)}")

    logger.info("Generating embeddings (this may take a minute)...")
    embeddings = embed_chunks(chunks)

    logger.info("Building Qdrant index...")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qclient = QdrantClient(url=qdrant_url)

    collection_name = "rag_regulators"

    # Recreate collection
    if qclient.collection_exists(collection_name):
        qclient.delete_collection(collection_name)

    qclient.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
    )

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "chunk_id": i
                }
            )
        )

    qclient.upload_points(collection_name=collection_name, points=points)

    logger.info("Saving chunks list for BM25...")
    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    logger.info("Done!")

if __name__ == "__main__":
    main()
