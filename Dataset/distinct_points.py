# count_all_distinct_fuzzy.py
import os
from qdrant_client import QdrantClient, models
import sys

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_HTTPS = os.getenv("QDRANT_HTTPS", "false").lower() == "true"

COLLECTION_NAME = "zscore_94"
BATCH_SIZE = 250

# --- Filter Definition ---
# Set to None to scan the ENTIRE collection.
scroll_filter = None

# --- Main Script Logic ---
print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
try:
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        https=QDRANT_HTTPS,
    )
    client.get_collections()
    print("Successfully connected to Qdrant.")

except Exception as e:
     print(f"\nError connecting to Qdrant: {e}")
     print("Please check connection settings.")
     sys.exit(1)

print(f"\nCounting distinct (artist, title) pairs in the *entire* collection '{COLLECTION_NAME}'...")
print("Applying normalization: stripping whitespace and converting to lowercase.")

if scroll_filter:
    try:
        print(f"Applying filter: {scroll_filter.model_dump()}")
    except AttributeError:
        try:
             print(f"Applying filter: {scroll_filter.dict()}")
        except AttributeError:
             print(f"Applying filter: {scroll_filter}")
else:
    print("No filter applied (scanning entire collection).")

# Use a set to store unique normalized pairs automatically
distinct_pairs = set()
next_offset = None
points_processed = 0
batch_num = 0

try:
    while True:
        batch_num += 1
        print(f"  Fetching batch #{batch_num} (limit={BATCH_SIZE})...", end='\r')

        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=BATCH_SIZE,
            offset=next_offset,
            with_payload=["artist", "title"],
            with_vectors=False
        )

        if not points and next_offset is None:
            print(f"\n  Received empty batch and no next offset. Assuming end of scroll.")
            break
        elif not points and next_offset is not None:
             print(f"\n  Received empty batch but received a next offset ({next_offset}). Continuing...")
             continue

        # Process the retrieved batch
        batch_point_count = 0
        for point in points:
            batch_point_count += 1
            points_processed += 1
            payload = point.payload
            artist = payload.get("artist")
            title = payload.get("title")

            # Check if both fields exist and are not None
            if artist is not None and title is not None:
                 # --- NORMALIZATION APPLIED HERE ---
                 # Normalize by stripping whitespace and converting to lowercase
                 # Ensure type is string before calling strip/lower if necessary, though .get should return strings or None
                 try:
                     normalized_pair = (str(artist).strip().lower(), str(title).strip().lower())
                     distinct_pairs.add(normalized_pair)
                 except Exception as norm_error:
                      print(f"\nWarning: Could not normalize pair ({artist}, {title}). Error: {norm_error}. Skipping this pair.")


        print(f"  Processed batch #{batch_num} ({batch_point_count} points). Total points processed: {points_processed}. Found {len(distinct_pairs)} distinct normalized pairs so far.")

        if next_offset is None:
            print(f"\n  Received null next offset. End of scroll reached.")
            break

except Exception as e:
    print(f"\n\nAn error occurred during scrolling: {e}")
    print("Processing stopped. Results might be partial.")
    sys.exit(1)

# --- Final Result ---
distinct_count = len(distinct_pairs)

print("\n--- Counting Complete ---")
print(f"Total points processed from the collection: {points_processed}")
print(f"Found {distinct_count} distinct normalized (case-insensitive, whitespace-stripped) (artist, title) pairs in the entire collection.")

# Optional: Print the first few normalized distinct pairs found
if distinct_pairs:
    print("\nFirst 10 distinct normalized pairs found:")
    sorted_distinct_list = sorted(list(distinct_pairs)) # Sort for consistent display
    for i, pair in enumerate(sorted_distinct_list):
        if i >= 10:
            break
        print(f"  - {pair}")