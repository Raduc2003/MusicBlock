#!/usr/bin/env python3
import os
import sys
import argparse
import json
import re
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

# --- Configuration ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 100
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Qdrant settings
QDRANT_URL = os.getenv('QDRANT_URL', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
COLLECTION_NAME = 'pdf_knowledge_base'

# Element type sets
ELEMENTS_TO_KEEP = {"NarrativeText", "ListItem", "Title", "UncategorizedText"}
GROUP_CONSECUTIVE = {"NarrativeText", "ListItem", "UncategorizedText"}

# Helper to prune and extract metadata
def get_element_metadata(el: Element) -> dict:
    md = getattr(el, 'metadata', {})
    if hasattr(md, 'to_dict'):
        md = md.to_dict()
    elif not isinstance(md, dict):
        md = {}
    for key in ('parent_id', 'element_id', 'coordinates', 'category'):
        md.pop(key, None)
    md['page_number'] = md.get('page_number')
    return md

# Combine consecutive elements up to max_chars into Documents
def combine_elements(elements: list[Element], max_chars: int) -> list[Document]:
    combined = []
    buffer = []
    for el in elements:
        current_len = sum(len(x.text) for x in buffer)
        if buffer and (el.category not in GROUP_CONSECUTIVE or current_len + len(el.text) > max_chars):
            content = "\n\n".join(x.text for x in buffer)
            md = get_element_metadata(buffer[0])
            pages = sorted(
                get_element_metadata(x).get('page_number')
                for x in buffer
                if get_element_metadata(x).get('page_number') is not None
            )
            types = sorted(x.category for x in buffer)
            md.update({
                'grouped_page_numbers': pages,
                'grouped_element_types': types,
                'original_element_count': len(buffer)
            })
            combined.append(Document(page_content=content, metadata=md))
            buffer = [el]
        else:
            buffer.append(el)
    if buffer:
        content = "\n\n".join(x.text for x in buffer)
        md = get_element_metadata(buffer[0])
        pages = sorted(
            get_element_metadata(x).get('page_number')
            for x in buffer
            if get_element_metadata(x).get('page_number') is not None
        )
        types = sorted(x.category for x in buffer)
        md.update({
            'grouped_page_numbers': pages,
            'grouped_element_types': types,
            'original_element_count': len(buffer)
        })
        combined.append(Document(page_content=content, metadata=md))
    return combined

# Load all PDFs under root folder into chunks
def load_and_chunk_folder(
    pdf_root: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_size: int = DEFAULT_MIN_CHUNK_SIZE
) -> list[Document]:
    """
    Traverse pdf_root, process each PDF with progress bar,
    handle errors, and summarize results.
    """
    # Collect (topic, path) entries
    pdf_entries = []
    for root, _, files in os.walk(pdf_root):
        topic = os.path.relpath(root, pdf_root)
        for fname in files:
            if fname.lower().endswith('.pdf'):
                pdf_entries.append((topic, os.path.join(root, fname), fname))

    all_chunks = []
    failed = []
    # Process with progress bar
    for topic, pdf_path, fname in tqdm(pdf_entries, desc="Processing PDFs", unit="pdf"):
        try:
            elements = partition(filename=pdf_path, strategy="fast")
        except Exception as e:
            tqdm.write(f"✗ Failed to parse {fname}: {e}")
            failed.append(fname)
            continue
        # Filter out undesired elements
        keep = [el for el in elements
                if el.category in ELEMENTS_TO_KEEP
                and hasattr(el, 'text')
                and len(el.text.strip()) > 10]
        # Combine and split
        combined = combine_elements(keep, chunk_size * 2)
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        preliminary = splitter.split_documents(combined)
        # Smart merge unfinished sentences
        merged, i = [], 0
        while i < len(preliminary):
            curr = preliminary[i]
            text = curr.page_content
            if i + 1 < len(preliminary) and not re.search(r"[\.\!\?]$", text.strip()):
                nxt = preliminary[i+1].page_content
                if len(text) + len(nxt) <= chunk_size:
                    merged.append(Document(page_content=text + ' ' + nxt, metadata=curr.metadata))
                    i += 2
                    continue
            merged.append(curr)
            i += 1
        # Final filter
        final = [c for c in merged if len(c.page_content.strip()) >= min_size]
        all_chunks.extend(final)
    # Report failures
    if failed:
        print(f"Failed to process {len(failed)} PDFs: {', '.join(failed)}")
    print(f"Processed {len(pdf_entries)-len(failed)} PDFs, generated {len(all_chunks)} chunks.")
    return all_chunks

# Upsert chunks into Qdrant with BGE embeddings + BM25 sparse
def upsert_to_qdrant(
    chunks: list[Document],
    url: str = QDRANT_URL,
    port: int = QDRANT_PORT,
    collection_name: str = COLLECTION_NAME
):
    client = QdrantClient(url=url, port=port)
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Determine embedding dimension
    sample = embedder.embed_documents(["test"])
    dims = len(sample[0]) if sample else DEFAULT_CHUNK_SIZE
    # (Re)create collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE)
    )
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedder,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
        distance_strategy="COSINE"
    )
    sparse = FastEmbedSparse("Qdrant/bm25")
    vector_store._sparse_embedding = sparse
    vector_store._retrieval_kwargs = {"search_type": "hybrid"}
    # Upsert with progress bar
    for chunk in tqdm(chunks, desc="Upserting chunks", unit="chunk"):
        vector_store.add_documents([chunk])
    print(f"Upserted {len(chunks)} chunks into '{collection_name}'")

# Main entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest folder of PDFs into Qdrant (BGE + BM25)")
    parser.add_argument('pdf_root', help="Root folder containing PDF subfolders as topics.")
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument('--chunk_overlap', type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument('--min_chunk_size', type=int, default=DEFAULT_MIN_CHUNK_SIZE)
    parser.add_argument('--qdrant_url', default=QDRANT_URL)
    parser.add_argument('--qdrant_port', type=int, default=QDRANT_PORT)
    parser.add_argument('--collection', default=COLLECTION_NAME)
    args = parser.parse_args()
    chunks = load_and_chunk_folder(
        args.pdf_root,
        args.chunk_size,
        args.chunk_overlap,
        args.min_chunk_size
    )
    if not chunks:
        sys.exit(1)
    upsert_to_qdrant(
        chunks,
        args.qdrant_url,
        args.qdrant_port,
        args.collection
    )
