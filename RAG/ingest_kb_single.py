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

# --- Configuration ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 100

# Element type sets
ELEMENTS_TO_KEEP = {"NarrativeText", "ListItem", "Title", "UncategorizedText"}
GROUP_CONSECUTIVE = {"NarrativeText", "ListItem", "UncategorizedText"}

# --- Helper Functions ---
def get_element_metadata(el: Element) -> dict:
    """Extract and prune metadata from an Unstructured element."""
    md = getattr(el, 'metadata', {})
    if hasattr(md, 'to_dict'):
        md = md.to_dict()
    elif not isinstance(md, dict):
        md = {}
    for key in ('parent_id', 'element_id', 'coordinates', 'category'):
        md.pop(key, None)
    md['page_number'] = md.get('page_number')
    return md


def combine_elements(elements: list[Element], max_chars: int) -> list[Document]:
    """
    Group consecutive elements into Documents, merging until max_chars per group.
    """
    combined = []
    buffer = []
    for el in elements:
        current_len = sum(len(x.text) for x in buffer)
        if buffer and (el.category not in GROUP_CONSECUTIVE or current_len + len(el.text) > max_chars):
            # Flush buffer into a Document
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
    print(f"Combined {len(elements)} elements into {len(combined)} documents.")
    return combined


def load_and_chunk(pdf_file: str, chunk_size: int, chunk_overlap: int, min_size: int) -> list[Document]:
    """
    Load a single PDF file, combine elements, then split with hierarchical and smart merging.
    """
    if not os.path.isfile(pdf_file):
        print(f"PDF file not found: {pdf_file}", file=sys.stderr)
        return []
    topic = os.path.splitext(os.path.basename(pdf_file))[0]
    print(f"Parsing PDF: {pdf_file}")
    try:
        elements = partition(filename=pdf_file)
    except Exception as e:
        print(f"Error parsing {pdf_file}: {e}", file=sys.stderr)
        return []
    if not elements:
        print(f"No content extracted from {pdf_file}", file=sys.stderr)
        return []
    meta = {'source': os.path.basename(pdf_file), 'topic': topic}

    keep = []
    for el in elements:
        if el.category in ELEMENTS_TO_KEEP and hasattr(el, 'text') and len(el.text.strip()) > 10:
            md = get_element_metadata(el)
            md.update(meta)
            md['element_type'] = el.category
            el.metadata = md
            keep.append(el)
    print(f"Filtered to {len(keep)} elements.")

    # Combine coarse documents
    combined = combine_elements(keep, chunk_size * 2)

    # Hierarchical splitting
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    preliminary = splitter.split_documents(combined)
    print(f"Preliminary split into {len(preliminary)} chunks.")

    # Smart merge mid-sentence chunks
    merged = []
    i = 0
    while i < len(preliminary):
        curr = preliminary[i]
        text = curr.page_content
        md = curr.metadata
        # Merge with next if unfinished sentence and fits
        if i + 1 < len(preliminary) and not re.search(r"[\.\!\?]$", text.strip()):
            next_text = preliminary[i + 1].page_content
            if len(text) + len(next_text) <= chunk_size:
                merged.append(Document(page_content=text + ' ' + next_text, metadata=md))
                i += 2
                continue
        merged.append(curr)
        i += 1

    # Filter small chunks and return
    final = [c for c in merged if len(c.page_content.strip()) >= min_size]
    print(f"Merged to {len(final)} final chunks (min size {min_size}).")
    return final


def save_json(chunks: list[Document], out_file: str) -> None:
    """Save chunks as a JSON file."""
    data = [{'page_content': c.page_content, 'metadata': c.metadata} for c in chunks]
    dirpath = os.path.dirname(out_file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {out_file}")


def save_readable(chunks: list[Document], out_file: str) -> None:
    """Save each chunk in a simple readable format."""
    dirpath = os.path.dirname(out_file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for i, c in enumerate(chunks, 1):
            pages = c.metadata.get('grouped_page_numbers') or [c.metadata.get('page_number')]
            pages_str = ",".join(str(p) for p in pages if p is not None)
            size = len(c.page_content)
            header = f"=== Chunk {i}: {c.metadata.get('source','N/A')} [pages:{pages_str}; size:{size} chars] ===\n"
            f.write(header)
            f.write(c.page_content)
            f.write("\n\n")
    print(f"Saved simplified text to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Ingest a single PDF into coherent RAG chunks.")
    parser.add_argument('pdf_file', help="Path to the PDF file to ingest.")
    parser.add_argument('--json', help="Output JSON path.")
    parser.add_argument('--text', help="Output readable text path.")
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE, help="Max chars per chunk.")
    parser.add_argument('--chunk_overlap', type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap chars.")
    parser.add_argument('--min_chunk_size', type=int, default=DEFAULT_MIN_CHUNK_SIZE, help="Min chars per chunk.")
    args = parser.parse_args()

    chunks = load_and_chunk(
        args.pdf_file,
        args.chunk_size,
        args.chunk_overlap,
        args.min_chunk_size
    )
    if not chunks:
        sys.exit(1)
    if args.json:
        save_json(chunks, args.json)
    if args.text:
        save_readable(chunks, args.text)
    if not args.json and not args.text:
        print("No output specified. Use --json and/or --text.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
