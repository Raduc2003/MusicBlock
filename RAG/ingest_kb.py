#!/usr/bin/env python3
import os
import json
import sys
import argparse
import re
import nltk
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
from langchain.schema import Document

# Download punkt tokenizer (only first run)
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --- Configuration ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
ELEMENT_TYPES_TO_GROUP = {"NarrativeText", "ListItem"}
MISCLASSIFIED_TYPES = {"UncategorizedText"}
SHORT_ELEMENT_MAX_CHARS = 150
PAGE_CONTINUATION_DELTA = 1

# --- Helper Functions ---
def get_metadata(element: Element) -> dict:
    """Extract and prune metadata from an Unstructured element."""
    md = getattr(element, "metadata", {})
    if hasattr(md, "to_dict"):
        md = md.to_dict()
    elif not isinstance(md, dict):
        md = {}
    for key in ("parent_id", "element_id", "coordinates", "category"):
        md.pop(key, None)
    return md


def group_elements(elements: list[Element]) -> list[Document]:
    """
    Group elements semantically into Documents, treating titles, exercises, and figures as boundaries.
    """
    groups = []
    buffer = []

    def flush_buffer():
        if not buffer:
            return
        text = "\n\n".join(item.text for item in buffer)
        base_md = get_metadata(buffer[0])
        pages = sorted({get_metadata(item).get("page_number") for item in buffer if get_metadata(item).get("page_number") is not None})
        types = sorted({item.category for item in buffer})
        struct_types = sorted({item.metadata.get("structural_type") for item in buffer if item.metadata.get("structural_type")})
        base_md.update({
            "grouped_page_numbers": pages,
            "grouped_element_types": types,
            "structural_types": struct_types,
            "original_element_count": len(buffer)
        })
        groups.append(Document(page_content=text, metadata=base_md))
        buffer.clear()

    for el in elements:
        text_strip = el.text.strip()
        # Structural boundaries
        if el.category == "Title":
            flush_buffer()
            el_md = get_metadata(el)
            el_md["structural_type"] = "title"
            el.metadata = el_md
            buffer.append(el)
            continue
        if re.match(r"^Exercise", text_strip, re.IGNORECASE):
            flush_buffer()
            el_md = get_metadata(el)
            el_md["structural_type"] = "exercise"
            el.metadata = el_md
            buffer.append(el)
            continue
        if re.match(r"^Figure\s+\d+(\.\d+)?", text_strip, re.IGNORECASE):
            flush_buffer()
            el_md = get_metadata(el)
            el_md["structural_type"] = "figure_caption"
            el.metadata = el_md
            buffer.append(el)
            continue
        # Default
        buffer.append(el)

    # Flush any remaining
    flush_buffer()
    print(f"Grouped {len(elements)} elements into {len(groups)} documents.")
    return groups


def split_paragraphs_and_sentences(docs: list[Document], chunk_size: int) -> list[Document]:
    """
    Split docs into paragraphs, then sentences, then merge small fragments.
    """
    # Stage 1: Paragraph split
    paras = []
    for doc in docs:
        for para in doc.page_content.split("\n\n"):
            text = para.strip()
            if not text:
                continue
            paras.append(Document(page_content=text, metadata=doc.metadata))

    # Stage 2: Sentence split for long paragraphs
    initial_chunks = []
    for para in paras:
        text = para.page_content
        md = para.metadata
        if len(text) <= chunk_size:
            initial_chunks.append(Document(page_content=text, metadata=md))
        else:
            for sent in sent_tokenize(text):
                if not sent:
                    continue
                initial_chunks.append(Document(page_content=sent, metadata=md))

    # Stage 3: Merge adjacent small chunks
    final_chunks = []
    if initial_chunks:
        buffer = initial_chunks[0]
        for curr in initial_chunks[1:]:
            combined = buffer.page_content + " " + curr.page_content
            if len(combined) <= chunk_size:
                buffer = Document(page_content=combined, metadata=buffer.metadata)
            else:
                final_chunks.append(buffer)
                buffer = curr
        final_chunks.append(buffer)
    return final_chunks


def load_and_chunk(pdf_dir: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Load the first PDF found, group elements, then apply multi-stage splitting.
    """
    for root, _, files in os.walk(pdf_dir):
        for fname in files:
            if not fname.lower().endswith('.pdf'):
                continue
            path = os.path.join(root, fname)
            topic = os.path.relpath(root, pdf_dir) or 'General'
            print(f"Loading PDF: {path} (Topic: {topic})")
            try:
                elements = partition(filename=path)
            except Exception as err:
                print(f"Error parsing {path}: {err}", file=sys.stderr)
                return []
            if not elements:
                print(f"No content extracted from {path}", file=sys.stderr)
                return []
            for el in elements:
                md = get_metadata(el)
                md['source'] = os.path.relpath(path, pdf_dir)
                md['topic'] = topic
                el.metadata = md
            docs = group_elements(elements)
            chunks = split_paragraphs_and_sentences(docs, chunk_size)
            print(f"Produced {len(chunks)} final chunks.")
            return chunks
    print("No PDF files found.", file=sys.stderr)
    return []


def save_json(chunks: list[Document], out_file: str) -> None:
    data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {out_file}")


def save_readable(chunks: list[Document], out_file: str) -> None:
    """
    Save each chunk in a simple format for quick PDF comparison.
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for idx, c in enumerate(chunks, start=1):
            pages = c.metadata.get('grouped_page_numbers') or [c.metadata.get('page_number')]
            pages_str = ",".join(map(str, pages))
            cats_str = ",".join(c.metadata.get('grouped_element_types', []))
            structs_str = ",".join(c.metadata.get('structural_types', []))
            size = len(c.page_content)
            header = (
                f"=== Chunk {idx}: {c.metadata.get('source','N/A')} "
                f"[pages:{pages_str}; cats:{cats_str}; structs:{structs_str}; size:{size} chars] ===\n"
            )
            f.write(header)
            f.write(c.page_content)
            f.write("\n\n")
    print(f"Saved simplified text to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into improved chunked text and JSON.")
    parser.add_argument("pdf_dir", nargs="?", default="./KB", help="Directory with PDFs.")
    parser.add_argument("--json", default="./chunks.json", help="Output JSON path.")
    parser.add_argument("--text", default="./chunks.txt", help="Output simplified text path.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Max chars per chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chars overlap (unused)." )
    args = parser.parse_args()

    chunks = load_and_chunk(args.pdf_dir, args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("No chunks generated.", file=sys.stderr)
        sys.exit(1)
    save_json(chunks, args.json)
    save_readable(chunks, args.text)

if __name__ == "__main__":
    main()
