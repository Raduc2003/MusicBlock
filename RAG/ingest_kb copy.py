#!/usr/bin/env python3
import os
import json
import argparse
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element # Base class for Unstructured elements
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import math # Not strictly needed but keeping imports used in logic

# --- Configuration ---
FINAL_CHUNK_SIZE = 1000
FINAL_CHUNK_OVERLAP = 100

# --- Define Element Types to Group ---
ELEMENT_TYPES_TO_GROUP = ["NarrativeText", "ListItem"]
# --- Heuristic for Misclassified Elements ---
POTENTIALLY_MISCLASSIFIED_TYPES = ["Title", "UncategorizedText"]
SHORT_ELEMENT_THRESHOLD = 150 # Max characters for a potentially misclassified element
PAGE_CONTINUATION_THRESHOLD = 1 # Max page difference to consider for continuation (0 for same page, 1 for same or next)
# ---

def get_metadata_dict(element: Element) -> dict:
    """Safely get metadata as a dictionary from an Unstructured element."""
    if hasattr(element.metadata, 'to_dict'):
        return element.metadata.to_dict()
    elif isinstance(element.metadata, dict):
        return element.metadata
    else:
        return {} # Return empty dict if metadata is unexpected type

def group_elements_semantically(elements: list[Element]) -> list[Document]:
    """
    Groups consecutive Unstructured elements based on type and a heuristic for short,
    potentially misclassified elements, creating LangChain Documents.
    """
    grouped_documents = []
    current_group = []

    for i, element in enumerate(elements):
        element_type = element.category
        metadata_dict = get_metadata_dict(element)
        current_page = metadata_dict.get('page_number')

        # Get info about the previous element if a group is being built
        previous_element = current_group[-1] if current_group else None
        previous_element_type = previous_element.category if previous_element else None
        previous_metadata_dict = get_metadata_dict(previous_element) if previous_element else {}
        previous_page = previous_metadata_dict.get('page_number')


        # Condition to group with the previous element (logic remains the same)
        is_standard_groupable_type = element_type in ELEMENT_TYPES_TO_GROUP
        is_previous_groupable_type = previous_element_type in ELEMENT_TYPES_TO_GROUP

        is_potentially_misclassified_short_continuation = (
             element_type in POTENTIALLY_MISCLASSIFIED_TYPES and
             len(element.text) <= SHORT_ELEMENT_THRESHOLD and
             previous_element is not None and is_previous_groupable_type and
             (current_page is None or previous_page is None or (current_page >= previous_page and current_page - previous_page <= PAGE_CONTINUATION_THRESHOLD))
        )

        should_group_with_previous = (
            (is_standard_groupable_type and is_previous_groupable_type) or
            is_potentially_misclassified_short_continuation
        )

        # If the current element should group with the previous, append to current_group
        # Otherwise, finalize the current group and start a new one
        if should_group_with_previous:
            current_group.append(element)
        else:
            # Finalize the current group (if not empty)
            if current_group:
                combined_content = "\n\n".join([doc.text for doc in current_group])

                # Combine metadata - get metadata from the first element as base
                # Use get_metadata_dict for safety
                combined_metadata_dict = get_metadata_dict(current_group[0])
                
                # Collect page numbers and element types from all elements in the group
                combined_metadata_dict['grouped_page_numbers'] = sorted(list(set([get_metadata_dict(el).get('page_number') for el in current_group if get_metadata_dict(el).get('page_number') is not None])))
                combined_metadata_dict['grouped_element_types'] = sorted(list(set([el.category for el in current_group if el.category is not None])))
                combined_metadata_dict['original_element_count'] = len(current_group)

                # Clean up some Unstructured metadata keys
                combined_metadata_dict.pop('parent_id', None)
                combined_metadata_dict.pop('element_id', None)
                combined_metadata_dict.pop('coordinates', None)
                combined_metadata_dict.pop('category', None)

                grouped_documents.append(Document(page_content=combined_content, metadata=combined_metadata_dict))

            # Start a new group with the current element
            current_group = [element]

    # After the loop, finalize the last group
    if current_group:
        combined_content = "\n\n".join([doc.text for doc in current_group])
        combined_metadata_dict = get_metadata_dict(current_group[-1]) # Use metadata from the last element for base
        combined_metadata_dict['grouped_page_numbers'] = sorted(list(set([get_metadata_dict(el).get('page_number') for el in current_group if get_metadata_dict(el).get('page_number') is not None])))
        combined_metadata_dict['grouped_element_types'] = sorted(list(set([el.category for el in current_group if el.category is not None])))
        combined_metadata_dict['original_element_count'] = len(current_group)
        combined_metadata_dict.pop('parent_id', None)
        combined_metadata_dict.pop('element_id', None)
        combined_metadata_dict.pop('coordinates', None)
        combined_metadata_dict.pop('category', None)
        grouped_documents.append(Document(page_content=combined_content, metadata=combined_metadata_dict))


    print(f"Grouped {len(elements)} elements into {len(grouped_documents)} larger documents.")
    return grouped_documents


def load_and_group_first_pdf_unstructured(kb_pdfs_path: str):
    # ... (same as before, simplified metadata addition)
    print(f"Loading the first PDF found in {kb_pdfs_path} using Unstructured...")

    found_first_pdf = False
    first_pdf_elements = []
    pdf_level_metadata = {} # Renamed for clarity

    for root, _, files in os.walk(kb_pdfs_path):
        if found_first_pdf: break
        relative_path = os.path.relpath(root, kb_pdfs_path)
        topic = relative_path if relative_path != '.' else 'General'
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                print(f"  Processing: {file_path} (Inferred Topic: {topic})")
                found_first_pdf = True

                try:
                    elements = partition(filename=file_path)
                    if not elements:
                        print(f"    Warning: No elements extracted from {file_path}. Skipping.")
                        return []
                except Exception as e:
                    print(f"    CRITICAL ERROR parsing {file_path} with Unstructured: {e}. Skipping this PDF.")
                    return []

                pdf_level_metadata = {
                    'topic': topic,
                    'source': os.path.relpath(file_path, kb_pdfs_path),
                    'original_filename': file
                }

                first_pdf_elements = elements
                break

    if not found_first_pdf:
        print("No PDF files found.")
        return []

    print(f"Extracted {len(first_pdf_elements)} elements from the first PDF.")

    # --- Metadata Enhancement for Elements (before grouping) ---
    processed_elements = []
    for element in first_pdf_elements:
         # Get element's original metadata as dict
         element_metadata_dict = get_metadata_dict(element)
         # Add PDF-level metadata
         element_metadata_dict.update(pdf_level_metadata)
         # Standardize element_type (use category from original metadata)
         element_metadata_dict['element_type'] = element_metadata_dict.get('category', element.category) # Prefer category from dict if present
         # Ensure page_number is accessible via key
         element_metadata_dict['page_number'] = element_metadata_dict.get('page_number', None)

         # Replace the element's metadata with the updated dictionary
         # Note: This might break if Unstructured elements expect metadata to be a specific object type.
         # A safer approach might be to pass the updated dict alongside the element.
         # Let's refine group_elements_semantically to handle original elements + added metadata.
         
         # Simplified: Just pass the elements and the PDF-level metadata separately to grouping? No, need element-specific metadata.
         # Let's pass a tuple: (element, updated_metadata_dict) to grouping.

         # This loop isn't strictly 'processing' elements, but associating info for grouping
         # The actual metadata update will happen *during* grouping based on the first element's metadata
         pass # No changes needed in this loop anymore, metadata update moves to grouping


    # --- Semantically Group Elements ---
    # Pass original elements directly. Metadata update will happen inside grouping.
    grouped_documents = group_elements_semantically(first_pdf_elements)


    # --- Final Chunking ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FINAL_CHUNK_SIZE,
        chunk_overlap=FINAL_CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    final_chunks = text_splitter.split_documents(grouped_documents)

    print(f"Split grouped documents into {len(final_chunks)} final chunks.")
    return final_chunks


def save_chunks_to_json(chunks: list[Document], output_filepath: str):
    # ... (same as before)
    serializable_chunks = []
    for chunk in chunks: serializable_chunks.append({"page_content": chunk.page_content, "metadata": chunk.metadata})
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f: json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
        print(f"Processed chunks saved to {output_filepath}")
    except Exception as e: print(f"Error saving chunks to JSON file: {e}")

def save_chunks_to_readable_text(chunks: list[Document], output_filepath: str):
    # ... (same as before)
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(f"  Source: {chunk.metadata.get('source', 'N/A')}\n")
                f.write(f"  Topic: {chunk.metadata.get('topic', 'N/A')}\n")
                f.write(f"  Pages: {chunk.metadata.get('grouped_page_numbers', [chunk.metadata.get('page_number', 'N/A')])}\n")
                f.write(f"  Element Types: {chunk.metadata.get('grouped_element_types', [chunk.metadata.get('element_type', 'N/A')])}\n")
                f.write(f"  Original Element Count: {chunk.metadata.get('original_element_count', 1)}\n")
                f.write(f"  Chunk Size: {len(chunk.page_content)} chars\n")
                f.write("---\n")
                f.write(f"{chunk.page_content}\n")
                f.write("\n" + "="*40 + "\n\n")
        print(f"Processed chunks saved to readable text file {output_filepath}")
    except Exception as e: print(f"Error saving chunks to readable text file: {e}")

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load, group elements semantically, and chunk the first PDF found."
    )
    parser.add_argument(
        "kb_pdfs_path", help="Path to the root directory containing KB PDFs (e.g., ./KB).", default="./KB", nargs='?')
    parser.add_argument("--output_json_file", help="Path to save the processed chunks as JSON.")
    parser.add_argument("--output_readable_file", help="Path to save the processed chunks as readable text.")
    parser.add_argument("--sample_chunks_to_print", type=int, default=5, help="Number of sample chunks to print (default: 5).")
    parser.add_argument("--chunk_size", type=int, default=FINAL_CHUNK_SIZE, help=f"Final chunk size (default: {FINAL_CHUNK_SIZE}).")
    parser.add_argument("--chunk_overlap", type=int, default=FINAL_CHUNK_OVERLAP, help=f"Final chunk overlap (default: {FINAL_CHUNK_OVERLAP}).")

    args = parser.parse_args()

    FINAL_CHUNK_SIZE = args.chunk_size
    FINAL_CHUNK_OVERLAP = args.chunk_overlap

    if not os.path.isdir(args.kb_pdfs_path): print(f"Error: Input directory not found at {args.kb_pdfs_path}"); sys.exit(1)

    document_chunks = load_and_group_first_pdf_unstructured(args.kb_pdfs_path)

    if args.output_json_file: save_chunks_to_json(document_chunks, args.output_json_file)
    if args.output_readable_file: save_chunks_to_readable_text(document_chunks, args.output_readable_file)
    if not args.output_json_file and not args.output_readable_file: print("\nNo output file specified. Chunks not saved.");

    print(f"\n--- Sample Chunks (first {args.sample_chunks_to_print}) ---")
    if document_chunks:
        chunks_to_print = document_chunks[:args.sample_chunks_to_print]
        for i, chunk in enumerate(chunks_to_print):
            print(f"--- Chunk {i+1} ---")
            print(f"Metadata: {json.dumps(chunk.metadata, indent=2)}")
            print(f"Content:\n{chunk.page_content}\n")
            print("-" * 20)
    else:
        print("No chunks were created.")

    print("\n--- Processing complete. ---")