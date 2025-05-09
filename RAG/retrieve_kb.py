import os
import sys
import argparse
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
RAG_COLLECTION_NAME = "pdf_knowledge_base"
DENSE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --- Retrieval Function ---
def retrieve_knowledge_chunks(
    search_query_text: str, # The final text query for semantic/keyword search
    qdrant_filter: qdrant_models.Filter | None = None, # Optional pre-constructed Qdrant filter
    top_k: int = 5
) -> list[Document]:
    """
    Performs Hybrid Search with optional Metadata Filtering against the RAG KB.
    This is a focused retrieval block. Query synthesis and filter creation
    should happen before calling this function.

    Args:
        search_query_text (str): The final text query for embedding and sparse search.
        qdrant_filter (qdrant_models.Filter | None): An optional, pre-constructed Qdrant filter object.
        top_k (int): Number of relevant chunks to retrieve.

    Returns:
        list[Document]: A list of LangChain Document objects representing relevant chunks.
    """
    print(f"Retrieving chunks for query: '{search_query_text[:100]}...'")
    if qdrant_filter:
        print(f"  Applying Qdrant filter: {qdrant_filter.dict()}", file=sys.stderr)

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=20.0)
        dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_EMBEDDING_MODEL)

        vector_store = Qdrant(
            client=client,
            collection_name=RAG_COLLECTION_NAME,
            embeddings=dense_embeddings,
            content_payload_key="page_content", # Ensure this matches ingestion
            metadata_payload_key="metadata"
        )

        # --- Execute Hybrid Search ---
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': top_k,
                'filter': qdrant_filter # Pass the pre-constructed filter
            }
        )

        retrieved_documents = retriever.invoke(search_query_text)

        print(f"  Retrieved {len(retrieved_documents)} documents via hybrid search.")
        return retrieved_documents

    except Exception as e:
        print(f"Error during retrieval: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve relevant KB chunks from Qdrant. (Standalone Test)")
    parser.add_argument("query_text", help="Text query for searching the KB.")
    parser.add_argument("--topic_filter", help="Optional topic to filter by (e.g., Rhythm, MusicTheory).", default=None)
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    test_filter = None
    if args.topic_filter:
        test_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.topic", # Assumes 'topic' field in metadata
                    match=qdrant_models.MatchValue(value=args.topic_filter)
                )
            ]
        )

    retrieved_chunks = retrieve_knowledge_chunks(
        search_query_text=args.query_text,
        qdrant_filter=test_filter,
        top_k=args.top_k
    )

    print(f"\n--- Top {len(retrieved_chunks)} Retrieved Chunks ---")
    if not retrieved_chunks:
        print("No relevant chunks found.")
    else:
        for i, doc in enumerate(retrieved_chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"  Content: {doc.page_content[:300]}...")