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
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client import QdrantClient, models as qdrant_models
from tqdm import tqdm
import time  # For delay after deleting collection

# --- Configuration ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 100
DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPARSE_EMBEDDING_MODEL_NAME = "Qdrant/bm25"

# Qdrant settings
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
COLLECTION_NAME = 'pdf_knowledge_base_hybrid3'

# Element type sets
ELEMENTS_TO_KEEP = {"NarrativeText", "ListItem", "Title", "UncategorizedText"}
GROUP_CONSECUTIVE = {"NarrativeText", "ListItem", "UncategorizedText"}

# --- Helper Functions ---
def get_element_metadata(el: Element) -> dict:
    md = getattr(el, 'metadata', {})
    if hasattr(md, 'to_dict'):
        md = md.to_dict()
    elif not isinstance(md, dict):
        md = {}

    # Preserve core metadata
    final_md = {}
    if 'page_number' in md:
        final_md['page_number'] = md['page_number']
    if 'category' in md:
        final_md['element_type'] = md['category']
    # Carry through custom fields
    for key in ('source', 'topic'):
        if key in md:
            final_md[key] = md[key]
    return final_md


def combine_elements(elements: list[Element], max_chars: int) -> list[Document]:
    combined = []
    buffer = []
    for el in elements:
        current_len = sum(len(x.text) for x in buffer)
        if buffer and (el.category not in GROUP_CONSECUTIVE or current_len + len(el.text) > max_chars):
            content = "\n\n".join(x.text for x in buffer)
            md = get_element_metadata(buffer[0])
            pages = sorted(set(get_element_metadata(b).get('page_number') for b in buffer if get_element_metadata(b).get('page_number') is not None))
            types = sorted(set(b.category for b in buffer))
            md.update({'grouped_page_numbers': pages,
                       'grouped_element_types': types,
                       'original_element_count': len(buffer)})
            combined.append(Document(page_content=content, metadata=md))
            buffer = [el]
        else:
            buffer.append(el)
    if buffer:
        content = "\n\n".join(x.text for x in buffer)
        md = get_element_metadata(buffer[0])
        pages = sorted(set(get_element_metadata(b).get('page_number') for b in buffer if get_element_metadata(b).get('page_number') is not None))
        types = sorted(set(b.category for b in buffer))
        md.update({'grouped_page_numbers': pages,
                   'grouped_element_types': types,
                   'original_element_count': len(buffer)})
        combined.append(Document(page_content=content, metadata=md))
    return combined


def load_and_chunk_folder(
    pdf_root: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_size: int = DEFAULT_MIN_CHUNK_SIZE
) -> list[Document]:
    pdf_entries = []
    for root, _, files in os.walk(pdf_root):
        topic = os.path.relpath(root, pdf_root)
        if topic == '.': topic = 'General'
        for fname in files:
            if fname.lower().endswith('.pdf'):
                pdf_entries.append((topic, os.path.join(root, fname), fname))

    all_final_chunks = []
    failed_pdfs = []
    for topic, pdf_path, fname in tqdm(pdf_entries, desc="Processing PDFs", unit="pdf"):
        try:
            elements = partition(filename=pdf_path, strategy="auto")
        except Exception as e:
            tqdm.write(f"✗ Failed to parse {fname}: {e}")
            failed_pdfs.append(fname)
            continue

        filtered_elements = []
        for el in elements:
            if el.category in ELEMENTS_TO_KEEP and hasattr(el, 'text') and el.text and len(el.text.strip()) > 10:
                md = get_element_metadata(el)
                md['source'] = os.path.relpath(pdf_path, pdf_root)
                md['topic'] = topic
                el.metadata = md
                filtered_elements.append(el)

        if not filtered_elements:
            continue

        combined_docs = combine_elements(filtered_elements, chunk_size * 2)
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=False,
            add_start_index=True
        )
        preliminary_chunks = splitter.split_documents(combined_docs)

        merged_chunks, i = [], 0
        while i < len(preliminary_chunks):
            curr = preliminary_chunks[i]
            text = curr.page_content
            if i + 1 < len(preliminary_chunks) and not re.search(r"[\.\!\?]$", text.strip()):
                nxt = preliminary_chunks[i+1]
                if len(text) + len(nxt.page_content) + 1 <= chunk_size:
                    merged_chunks.append(Document(page_content=text + ' ' + nxt.page_content, metadata=curr.metadata))
                    i += 2
                    continue
            merged_chunks.append(curr)
            i += 1

        final_chunks = [c for c in merged_chunks if len(c.page_content.strip()) >= min_size]
        all_final_chunks.extend(final_chunks)

    if failed_pdfs:
        print(f"Failed to process {len(failed_pdfs)} PDFs: {', '.join(failed_pdfs)}")
    print(f"Processed {len(pdf_entries)-len(failed_pdfs)} PDFs, generated {len(all_final_chunks)} chunks.")
    return all_final_chunks


def upsert_to_qdrant_hybrid(
    chunks: list[Document],
    url: str = QDRANT_URL,
    port: int = QDRANT_PORT,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False
):
    client = QdrantClient(url=url, port=port, timeout=60.0, prefer_grpc=True)
    dense_embedder = HuggingFaceEmbeddings(model_name=DENSE_EMBEDDING_MODEL_NAME)
    sparse_embedder = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL_NAME)

    sample = dense_embedder.embed_query("test")
    dense_dims = len(sample)

    exists = False
    try:
        client.get_collection(collection_name)
        exists = True
    except:
        pass

    if recreate and exists:
        client.delete_collection(collection_name)
        time.sleep(1)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": qdrant_models.VectorParams(size=dense_dims, distance=qdrant_models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": qdrant_models.SparseVectorParams(
                    index=qdrant_models.SparseIndexParams(on_disk=True)
                )
            },
            optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=0),
            hnsw_config=qdrant_models.HnswConfigDiff(m=16, ef_construct=100),
        )
        from qdrant_client.http import models as rest

        # Create payload indexes with proper enum types
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.topic",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.source",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        

    points = []
    texts = [c.page_content for c in chunks]
    dense_vectors = dense_embedder.embed_documents(texts)

    for i, chunk in enumerate(tqdm(chunks, desc="Preparing points", unit="chunk")):
        sparse_data = sparse_embedder.embed_documents([chunk.page_content])[0]
        q_sparse = qdrant_models.SparseVector(indices=sparse_data.indices, values=sparse_data.values)
        payload = {"metadata":chunk.metadata.copy()}
        payload["page_content"] = chunk.page_content
        points.append(qdrant_models.PointStruct(id=i, payload=payload, vector={"dense": dense_vectors[i], "sparse": q_sparse}))

    BATCH = 1000
    for j in range(0, len(points), BATCH):
        batch = points[j:j+BATCH]
        client.upsert(collection_name=collection_name, points=batch, wait=True)

    print(f"Upserted {len(points)} chunks into '{collection_name}'")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pdf_root')
    p.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument('--chunk_overlap', type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument('--min_chunk_size', type=int, default=DEFAULT_MIN_CHUNK_SIZE)
    p.add_argument('--qdrant_url', default=QDRANT_URL)
    p.add_argument('--qdrant_port', type=int, default=QDRANT_PORT)
    p.add_argument('--collection', default=COLLECTION_NAME)
    p.add_argument('--recreate', action='store_true')
    args = p.parse_args()

    chunks = load_and_chunk_folder(args.pdf_root, args.chunk_size, args.chunk_overlap, args.min_chunk_size)
    if not chunks:
        sys.exit(1)
    upsert_to_qdrant_hybrid(chunks, args.qdrant_url, args.qdrant_port, args.collection, recreate=args.recreate)
