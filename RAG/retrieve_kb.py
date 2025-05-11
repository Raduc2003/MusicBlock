#!/usr/bin/env python3
import os
import sys
import argparse
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
RAG_COLLECTION_NAME = "pdf_knowledge_base_hybrid3"
DENSE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# --- Retrieval Function ---
def retrieve_knowledge_chunks(
    search_query_text: str,
    qdrant_filter: qdrant_models.Filter | None = None,
    dense_embedder: HuggingFaceEmbeddings = None,
    sparse_embedder: FastEmbedSparse = None,
    top_k: int = 5
) -> list[Document]:
    """
    Retrieve top_k chunks from an existing Qdrant collection using hybrid search.
    """
    print(f"Retrieving chunks for query: '{search_query_text[:100]}' using HYBRID mode")
    if qdrant_filter:
        print(f"  Applying Qdrant filter: {qdrant_filter.dict()}", file=sys.stderr)

   

    # Build hybrid-enabled vector store, expose all payload fields as metadata
    vector_store = QdrantVectorStore.from_existing_collection(
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        prefer_grpc=False,
        grpc_port=QDRANT_PORT,
        collection_name=RAG_COLLECTION_NAME,
        embedding=dense_embedder,
        sparse_embedding=sparse_embedder,
        retrieval_mode=RetrievalMode.HYBRID
        ,
        content_payload_key="page_content",
        metadata_payload_key="metadata",        # Expose all other payload fields directly
        vector_name="dense",
        sparse_vector_name="sparse"
    )

    # Create retriever and fetch documents
    retriever = vector_store.as_retriever(
        search_kwargs={
            'k': top_k,
            'filter': qdrant_filter
        }
    )
    docs = retriever.invoke(search_query_text)
    print(f"Retrieved {len(docs)} chunks.")
    return docs

# --- Standalone CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve relevant KB chunks from Qdrant using hybrid search."
    )
    parser.add_argument("query_text", help="Text query for searching the KB.")
    parser.add_argument(
        "--topic_filter",
        help="Optional topic to filter by (e.g., Rhythm, MusicTheory).",
        default=None
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve."
    )
    args = parser.parse_args()

    # Build Qdrant filter if needed (match top-level payload field)
    test_filter = None
    if args.topic_filter:
        test_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="topic",
                    match=qdrant_models.MatchValue(value=args.topic_filter)
                )
            ]
        )

    # Retrieve and print
    retrieved_chunks = retrieve_knowledge_chunks(
        search_query_text=args.query_text,
        qdrant_filter=test_filter,
        top_k=args.top_k
    )

    print(f"\n--- Top {len(retrieved_chunks)} Retrieved Chunks ---")
    if not retrieved_chunks:
        print("No relevant chunks found.")
    else:
        for i, doc in enumerate(retrieved_chunks, start=1):
            print(f"\nChunk {i}:")
            # All payload fields are now in metadata
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"  Pages: {doc.metadata.get('grouped_page_numbers', [])}")
            print(f"  Element Types: {doc.metadata.get('grouped_element_types', [])}")
            print(f"  Content: {doc.page_content[:300]}...")

