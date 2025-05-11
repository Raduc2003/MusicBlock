# RAG/rag_tools.py

from qdrant_client.http import models as qdrant_models
from langchain_core.documents import Document
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from retrieve_kb import retrieve_knowledge_chunks

from rag_config import DEFAULT_RETRIEVAL_TOP_K ,DENSE_EMBEDDING_MODEL_NAME, SPARSE_EMBEDDING_MODEL_NAME, QDRANT_URL

# Initialize embedding models globally
print("Initializing embedding models in rag_tools.py...")
try:
    # Specify device for HuggingFaceEmbeddings if you want to control CPU/GPU
    # model_kwargs = {'device': 'cpu'} # To force CPU for dense embeddings
    # encode_kwargs = {'normalize_embeddings': True} # if you had this before
    dense_embedder_global = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL_NAME,
        model_kwargs = {'device': 'cpu'}
    )
    sparse_embedder_global = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL_NAME)
    print("Embedding models initialized globally in rag_tools.")
except Exception as e:
    print(f"FATAL: Could not initialize global embedding models in rag_tools.py: {e}")
    dense_embedder_global = None
    sparse_embedder_global = None



def get_knowledge_from_kb( 
    search_query: str,
    topic: Optional[str] = None,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K
) -> List[Document]:
   
    if not dense_embedder_global or not sparse_embedder_global:
        print("Error: Global embedding models not initialized in rag_tools.")
        return []


    if topic:
        print(f"TOOL: 'get_knowledge_from_kb' called for topic '{topic}' with query: '{search_query[:100]}...' (top_k={top_k})")
    else:
        print(f"TOOL: 'get_knowledge_from_kb' called with no topic filter. Query: '{search_query[:100]}...' (top_k={top_k})")

    q_filter = None
    if topic:
        # Construct the Qdrant filter for the 'metadata.topic' field.
        q_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.topic",  # Targeting payload.metadata.topic
                    match=qdrant_models.MatchValue(value=topic)
                )
            ]
        )
    # If topic is None, q_filter remains None, and retrieve_knowledge_chunks should handle it (i.e., no filter applied)

    try:
        retrieved_docs = retrieve_knowledge_chunks(
            search_query_text=search_query,
            qdrant_filter=q_filter,
            top_k=top_k,
            dense_embedder=dense_embedder_global, 
            sparse_embedder=sparse_embedder_global 
            )
        if topic:
            print(f"TOOL: Retrieved {len(retrieved_docs)} documents for topic '{topic}'.")
        else:
            print(f"TOOL: Retrieved {len(retrieved_docs)} documents (no topic filter).")
        return retrieved_docs
    except Exception as e:
        print(f"Error in 'get_knowledge_from_kb' tool: {e}")
        return []

if __name__ == '__main__':
    print("\n--- Testing rag_tools.py ---")



    sample_query_specific = "how to make a groovy bassline"
    sample_topic_specific = "Rhythm"
    sample_query_general = "how to make a bassline"
    sample_top_k = 2

    print(f"\nTesting 'get_knowledge_from_kb' with specific topic:")
    print(f"  Query: '{sample_query_specific}'")
    print(f"  Topic: '{sample_topic_specific}'")
    print(f"  Top K: {sample_top_k}")

    try:
        documents_specific = get_knowledge_from_kb(sample_query_specific, sample_topic_specific, sample_top_k)
        if documents_specific:
            print(f"\nSuccessfully retrieved {len(documents_specific)} documents (specific topic):")
            for i, doc in enumerate(documents_specific):
                print(f"  Doc {i+1}: Content: {doc.page_content[:100]}... Metadata: {doc.metadata.get('topic')}")
        else:
            print("\nNo documents retrieved for specific topic query.")
    except Exception as e:
        print(f"\nError during specific topic test: {e}")


    print(f"\nTesting 'get_knowledge_from_kb' with NO topic (general search):")
    print(f"  Query: '{sample_query_general}'")
    print(f"  Top K: {sample_top_k}")
    try:
        documents_general = get_knowledge_from_kb(sample_query_general, top_k=sample_top_k) # topic is None
        if documents_general:
            print(f"\nSuccessfully retrieved {len(documents_general)} documents (general search):")
            for i, doc in enumerate(documents_general):
                print(f"  Doc {i+1}: Content: {doc.page_content[:100]}... Metadata: {doc.metadata.get('topic')}") # Print topic to see diversity
        else:
            print("\nNo documents retrieved for general query.")
    except Exception as e:
        print(f"\nError during general search test: {e}")