#!/usr/bin/env python3
import csv
import argparse
import os
from collections import defaultdict
import glob # To find CSV files easily

def analyze_variants_coverage(csv_filepath):
    """
    Calculates coverage percentages from a single test results CSV file,
    broken down by the 'variant' column. [Identical to previous version]
    """
    # (Code for analyze_variants_coverage from the previous response - kept for brevity)
    # ... [omitted - identical to previous script's analyze_variants_coverage function] ...
    if not os.path.isfile(csv_filepath): return None
    coverage_cols = ['found_in_top_1','found_in_top_10','found_in_top_20','found_in_top_100']
    col_map = {'found_in_top_1': 'top1','found_in_top_10': 'top10','found_in_top_20': 'top20','found_in_top_100': 'top100'}
    variant_stats = defaultdict(lambda: {'counts': defaultdict(int), 'total': 0})
    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            first_line = csvfile.readline();
            if not first_line: return None
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)
            header = reader.fieldnames;
            if not header: return None
            required_cols = ['variant'] + coverage_cols
            if not all(col in header for col in required_cols):
                missing = [col for col in required_cols if col not in header]
                print(f"Warning: CSV '{os.path.basename(csv_filepath)}' missing {missing}. Skipping.")
                return None
            for row in reader:
                if not any(row.values()): continue
                variant = row.get('variant', '').strip();
                if not variant: continue
                variant_stats[variant]['total'] += 1
                for col in coverage_cols:
                    if row.get(col, '').strip().lower() == 'true':
                        variant_stats[variant]['counts'][col_map[col]] += 1
    except Exception as e:
        print(f"Error reading {os.path.basename(csv_filepath)}: {e}"); return None
    if not variant_stats: return {}
    final_results = {}
    for variant, stats in variant_stats.items():
        total_variant_rows = stats['total']
        variant_coverage = {}
        for csv_col, key in col_map.items():
            count = stats['counts'][key]
            percentage = (count / total_variant_rows) * 100 if total_variant_rows > 0 else 0.0
            variant_coverage[key] = {'count': count,'total': total_variant_rows,'percentage': percentage}
        final_results[variant] = variant_coverage
    return final_results
# --- End of omitted analyze_variants_coverage function ---


def analyze_folder_compare_variants(input_folder, output_csv_path):
    """
    Analyzes all CSV files in a folder for coverage per variant and writes
    a single summary CSV file with variants compared side-by-side per input file.

    Args:
        input_folder (str): Path to the folder containing result CSV files.
        output_csv_path (str): Path to write the summary CSV file.
    """
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in folder: {input_folder}"); return

    print(f"Found {len(csv_files)} CSV files in {input_folder}. Analyzing to compare variants...")

    # --- Define the header dynamically based on coverage levels and target variants ---
    target_variants = ['f', 'c', '30s'] # Variants to include in columns
    coverage_levels = ['top1', 'top10', 'top20', 'top100']
    output_fieldnames = ['input_filename']
    for variant in target_variants:
        output_fieldnames.append(f'{variant}_total_cases')
        for level in coverage_levels:
            output_fieldnames.append(f'{variant}_{level}_count')
            output_fieldnames.append(f'{variant}_{level}_percentage')
    # --- End Header Definition ---


    all_results_data = [] # List to hold one row dictionary per input CSV
    processed_count = 0
    error_count = 0

    for csv_path in sorted(csv_files):
        filename = os.path.basename(csv_path)
        print(f"--- Analyzing file: {filename} ---")
        variant_coverage_results = analyze_variants_coverage(csv_path)

        if variant_coverage_results is None:
            print(f"Failed to process {filename} due to errors.")
            error_count += 1; continue
        if not variant_coverage_results:
            print(f"No variant data found in {filename}. Skipping.")
            continue

        processed_count += 1
        # --- Create ONE row for the current input file ---
        combined_row_data = {'input_filename': filename}

        # Populate the row with data for each target variant
        for variant in target_variants:
            results = variant_coverage_results.get(variant) # Get results for this specific variant

            if results: # If data exists for this variant in the file
                total_cases_variant = results.get('top1', {}).get('total', 0)
                combined_row_data[f'{variant}_total_cases'] = total_cases_variant
                for level in coverage_levels: # key is 'top1', 'top10', etc.
                    stats = results.get(level, {'count': 0, 'percentage': 0.0}) # Default if level missing
                    combined_row_data[f'{variant}_{level}_count'] = stats['count']
                    combined_row_data[f'{variant}_{level}_percentage'] = f"{stats['percentage']:.2f}"
            else: # If this variant wasn't found in the input file
                combined_row_data[f'{variant}_total_cases'] = 0
                for level in coverage_levels:
                    combined_row_data[f'{variant}_{level}_count'] = 0
                    combined_row_data[f'{variant}_{level}_percentage'] = "0.00" # Or "" for blank

        all_results_data.append(combined_row_data)
        # --- End of processing for one input file ---

    if not all_results_data:
        print("No data collected from any CSV files. Output file will not be created.")
        return

    print(f"\n--- Writing variant comparison summary to: {output_csv_path} ---")
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()
            # Sort by filename before writing
            all_results_data.sort(key=lambda x: x['input_filename'])
            writer.writerows(all_results_data)
        print("Summary file written successfully.")
    except Exception as e:
        print(f"Error writing summary CSV file {output_csv_path}: {e}")

    print("\nAnalysis complete.")
    print(f"Successfully analyzed files: {processed_count}")
    print(f"Files skipped due to errors or no data: {error_count + (len(csv_files) - processed_count - error_count)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze coverage comparing variants (f, c, 30s) from multiple test result CSV files."
    )
    parser.add_argument("input_folder", help="Path to the folder containing result CSV files.")
    parser.add_argument("output_file", help="Path for the output summary CSV file.")
    args = parser.parse_args()

    analyze_folder_compare_variants(args.input_folder, args.output_file)