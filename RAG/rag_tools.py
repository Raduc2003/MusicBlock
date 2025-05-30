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
import os
import html
import requests
from typing import Dict, Any, TypedDict

print("Initializing embedding models in rag_tools.py...")
try:

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
        print(f"TOOL: Full query text: '{search_query}'")
    else:
        print(f"TOOL: 'get_knowledge_from_kb' called with no topic filter. Query: '{search_query[:100]}...' (top_k={top_k})")

    q_filter = None
    if topic:
        print(f"TOOL: Setting up Qdrant filter for topic: '{topic}'")
        q_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.topic",  
                    match=qdrant_models.MatchValue(value=topic)
                )
            ]
        )

    try:
        print(f"TOOL: Calling retrieve_knowledge_chunks...")
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
            
        # Log document details
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'N/A')
            page = doc.metadata.get('page_number', 'N/A')
            content_preview = doc.page_content[:150].replace('\n', ' ')
            print(f"TOOL: Doc {i+1}: {source} (Pg:{page}) - {content_preview}...")
            
        return retrieved_docs
    except Exception as e:
        print(f"Error in 'get_knowledge_from_kb' tool: {e}")
        return []

# --- StackExchange Q&A Retrieval Tool ---
STACKEXCHANGE_API_KEY = os.getenv("STACKEXCHANGE_API_KEY")

class StackExchangeResult(TypedDict):
    source_url: str
    question_title: str
    question_id: int
    answer_html_bodies: List[str] # Raw HTML bodies


def search_stackexchange_qa(
    query: str,
    site: str = 'music.stackexchange.com',
    num_questions: int = 5, 
    num_answers: int = 2
) -> List[StackExchangeResult]: # Return type is a list of our TypedDict
    """
    Fetches top questions and their top answers from a StackExchange site.
    Returns a list of dictionaries, each containing the source URL, question, and answer bodies.
    """
    clean_query = query.strip()
    clean_query = clean_query.strip('"').strip("'").replace('"', '').replace("'", '').replace('"', '').replace('"', '')
    clean_query = clean_query.split('\n')[0].strip()
    clean_query = clean_query.strip('.,!?;:')
    if len(clean_query) > 100:
        clean_query = clean_query[:100].strip()
    
    print(f"TOOL (search_stackexchange_qa): Searching '{site}' for query: '{clean_query}' ({num_questions}q, {num_answers}a)")
    if query != clean_query:
        print(f"TOOL (search_stackexchange_qa): Original query was: '{query}', cleaned to: '{clean_query}'")

    q_params: Dict[str, Any] = {
        'site': site, 'q': clean_query, 'pagesize': num_questions, 'sort': 'relevance', 'order': 'desc',
    }
    if STACKEXCHANGE_API_KEY: q_params['key'] = STACKEXCHANGE_API_KEY

    try:
        print(f"TOOL (search_stackexchange_qa): Making request to StackExchange API...")
        q_resp = requests.get('https://api.stackexchange.com/2.3/search/advanced', params=q_params, timeout=15)
        print(f"TOOL (search_stackexchange_qa): Response status: {q_resp.status_code}")
        q_resp.raise_for_status()
        questions = q_resp.json().get('items', [])
        print(f"TOOL (search_stackexchange_qa): Found {len(questions)} questions")
    except requests.exceptions.RequestException as e:
        print(f"TOOL (search_stackexchange_qa): Error fetching StackExchange questions: {e}")
        return []

    tool_results: List[StackExchangeResult] = []

    for q_item in questions:
        qid = q_item['question_id']
        title = html.unescape(q_item['title'])
        link = q_item['link']
        accepted_answer_id = q_item.get('accepted_answer_id')

    
        a_params: Dict[str, Any] = {
            'site': site, 'sort': 'votes', 'order': 'desc', 'pagesize': num_answers, 'filter': 'withbody'
        }
        if STACKEXCHANGE_API_KEY: a_params['key'] = STACKEXCHANGE_API_KEY
        
        fetched_answers_bodies = []
        try:
            answer_ids_to_fetch_str = ""
            if num_answers == 1 and accepted_answer_id:
                answer_ids_to_fetch_str = str(accepted_answer_id)
            else:
                ans_resp = requests.get(f'https://api.stackexchange.com/2.3/questions/{qid}/answers', params=a_params, timeout=10)
                ans_resp.raise_for_status()
                answers_data = ans_resp.json().get('items', [])
                if accepted_answer_id and num_answers > 1 and not any(a['answer_id'] == accepted_answer_id for a in answers_data):
                    ids_list = [str(accepted_answer_id)] + [str(a['answer_id']) for a in answers_data]
                    answer_ids_to_fetch_str = ";".join(ids_list[:num_answers])
                elif answers_data:
                    answer_ids_to_fetch_str = ";".join(str(a['answer_id']) for a in answers_data[:num_answers])
                elif accepted_answer_id:
                     answer_ids_to_fetch_str = str(accepted_answer_id)


            if answer_ids_to_fetch_str:
                specific_a_params = {'site': site, 'filter': 'withbody', 'order': 'desc', 'sort': 'activity'}
                if STACKEXCHANGE_API_KEY: specific_a_params['key'] = STACKEXCHANGE_API_KEY
                final_ans_resp = requests.get(f'https://api.stackexchange.com/2.3/answers/{answer_ids_to_fetch_str}', params=specific_a_params, timeout=15)
                final_ans_resp.raise_for_status()
                final_answers_data = final_ans_resp.json().get('items', [])
                fetched_answers_bodies = [html.unescape(ans['body']) for ans in final_answers_data]

            tool_results.append({
                'source_url': link, 
                'question_title': title,
                'question_id': qid,
                'answer_html_bodies': fetched_answers_bodies
            })
        except requests.exceptions.RequestException as e:
            print(f"Error fetching StackExchange answers for QID {qid}: {e}")
            tool_results.append({'source_url': link, 'question_title': title, 'question_id': qid, 'answer_html_bodies': []})

    print(f"TOOL (search_stackexchange_qa): Found {len(tool_results)} Q&A items.")
    return tool_results


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