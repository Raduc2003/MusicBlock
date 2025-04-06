#!/usr/bin/env python3
import csv
import argparse
import os
from collections import defaultdict
import glob # To find CSV files easily

def calculate_coverage(csv_filepath):
    """
    Calculates coverage percentages from a single test results CSV file.

    Args:
        csv_filepath (str): Path to the input CSV file.

    Returns:
        dict: A dictionary containing coverage stats (counts, total, percentages)
              for different thresholds, or None if the file cannot be processed.
              Example: {'top1': {'count': 15, 'total': 30, 'percentage': 50.0}, ...}
    """
    if not os.path.isfile(csv_filepath):
        print(f"Error: File not found at {csv_filepath}")
        return None

    # Columns to calculate coverage for
    coverage_cols = [
        'found_in_top_1',
        'found_in_top_10',
        'found_in_top_20',
        'found_in_top_100'
    ]
    # Map CSV column names to simpler keys for the output dict
    col_map = {
        'found_in_top_1': 'top1',
        'found_in_top_10': 'top10',
        'found_in_top_20': 'top20',
        'found_in_top_100': 'top100'
    }

    counts = defaultdict(int)
    total_rows = 0

    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            # Handle potential empty files or files with just a header
            first_line = csvfile.readline()
            if not first_line:
                 print(f"Warning: CSV file '{os.path.basename(csv_filepath)}' appears to be empty. Skipping.")
                 return None # Treat as unable to process
            csvfile.seek(0) # Go back to the start

            reader = csv.DictReader(csvfile)

            # Verify header contains necessary columns
            header = reader.fieldnames
            if not header:
                print(f"Error: CSV file '{os.path.basename(csv_filepath)}' has no header after first line check. Skipping.")
                return None
            if not all(col in header for col in coverage_cols):
                missing = [col for col in coverage_cols if col not in header]
                print(f"Error: CSV file '{os.path.basename(csv_filepath)}' is missing columns: {missing}. Skipping.")
                return None

            for row in reader:
                # Skip potential empty rows at the end
                if not any(row.values()):
                    continue
                total_rows += 1
                for col in coverage_cols:
                    # Check if the value is exactly the string "True"
                    if row.get(col, '').strip().lower() == 'true':
                        counts[col_map[col]] += 1

    except Exception as e:
        print(f"Error reading or processing CSV file {os.path.basename(csv_filepath)}: {e}")
        return None

    if total_rows == 0:
        print(f"Warning: No valid data rows found in {os.path.basename(csv_filepath)} after header.")
        # Return zero counts and percentages
        results = {}
        for key in col_map.values():
            results[key] = {'count': 0, 'total': 0, 'percentage': 0.0}
        return results

    # Calculate percentages
    results = {}
    for csv_col, key in col_map.items():
        count = counts[key]
        percentage = (count / total_rows) * 100 if total_rows > 0 else 0.0
        results[key] = {
            'count': count,
            'total': total_rows,
            'percentage': percentage
        }

    return results

def analyze_folder(input_folder, output_csv_path):
    """
    Analyzes all CSV files in a folder and writes summary coverage stats to an output CSV.

    Args:
        input_folder (str): Path to the folder containing result CSV files.
        output_csv_path (str): Path to write the summary CSV file.
    """
    # Find all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in folder: {input_folder}")
        return

    print(f"Found {len(csv_files)} CSV files in {input_folder}. Analyzing...")

    # Define the header for the output summary CSV
    output_fieldnames = [
        'input_filename', 'total_cases',
        'top1_count', 'top1_percentage',
        'top10_count', 'top10_percentage',
        'top20_count', 'top20_percentage',
        'top100_count', 'top100_percentage'
    ]

    all_results_data = []
    processed_count = 0
    error_count = 0

    # Process each CSV file
    for csv_path in sorted(csv_files): # Sort for consistent output order
        print(f"--- Analyzing: {os.path.basename(csv_path)} ---")
        coverage_results = calculate_coverage(csv_path)

        if coverage_results:
            # Prepare row data for the output CSV
            total_cases = coverage_results.get('top1', {}).get('total', 0)
            row_data = {
                'input_filename': os.path.basename(csv_path),
                'total_cases': total_cases
            }
            for key, stats in coverage_results.items(): # key is 'top1', 'top10', etc.
                row_data[f'{key}_count'] = stats['count']
                # Format percentage to 2 decimal places for the output CSV
                row_data[f'{key}_percentage'] = f"{stats['percentage']:.2f}"

            all_results_data.append(row_data)
            processed_count += 1
        else:
            print(f"Failed to process {os.path.basename(csv_path)}")
            error_count += 1

    # Write the aggregated results to the output CSV file
    if not all_results_data:
        print("No data collected from any CSV files. Output file will not be created.")
        return

    print(f"\n--- Writing summary to: {output_csv_path} ---")
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(all_results_data)
        print("Summary file written successfully.")
    except Exception as e:
        print(f"Error writing summary CSV file {output_csv_path}: {e}")

    print("\nAnalysis complete.")
    print(f"Successfully analyzed files: {processed_count}")
    print(f"Files with errors or no data: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze coverage from multiple test result CSV files in a folder."
    )
    parser.add_argument("input_folder", help="Path to the folder containing result CSV files.")
    parser.add_argument("output_file", help="Path for the output summary CSV file.")
    args = parser.parse_args()

    analyze_folder(args.input_folder, args.output_file)