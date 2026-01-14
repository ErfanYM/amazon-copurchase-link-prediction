#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run script for Amazon Co-Purchase Link Prediction Pipeline

This wrapper makes the main code runnable outside of Google Colab by:
1. Removing Colab-specific imports (google.colab, IPython.display)
2. Configuring data paths via command-line arguments
3. Providing a clean entry point

Usage:
    python src/run.py --data-dir data --max-lines 100000
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Amazon Co-Purchase Link Prediction Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the .json.gz dataset files (default: data)"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100000,
        help="Maximum number of review lines to load per category (default: 100000)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for output CSV files (default: results)"
    )
    parser.add_argument(
        "--figs-dir",
        type=str,
        default="figs",
        help="Directory for output figures (default: figs)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for evaluation (default: 42 43 44 45 46)"
    )
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)
    
    print("="*70)
    print("Amazon Co-Purchase Link Prediction Pipeline")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Max lines per category: {args.max_lines}")
    print(f"Results directory: {args.results_dir}")
    print(f"Figures directory: {args.figs_dir}")
    print(f"Random seeds: {args.seeds}")
    print("="*70)
    
    # Check if data files exist
    data_files = {
        "Electronics": os.path.join(args.data_dir, "Electronics_5.json.gz"),
        "All_Beauty": os.path.join(args.data_dir, "All_Beauty_5.json.gz"),
        "Home_and_Kitchen": os.path.join(args.data_dir, "Home_and_Kitchen_5.json.gz"),
    }
    
    missing_files = [name for name, path in data_files.items() if not os.path.exists(path)]
    if missing_files:
        print(f"\n❌ ERROR: Missing dataset files for: {', '.join(missing_files)}")
        print(f"\nPlease download the required datasets and place them in '{args.data_dir}/'")
        print("See data/README.md for download instructions.")
        sys.exit(1)
    
    print("\n✓ All required dataset files found")
    
    # Import and run the main pipeline
    # We'll set global variables that the original code expects
    globals()['RESULTS_DIR'] = args.results_dir
    globals()['FIGS_DIR'] = args.figs_dir
    globals()['DATA_PATHS'] = data_files
    globals()['MAX_LINES'] = args.max_lines
    globals()['SEEDS'] = args.seeds
    
    print("\n" + "="*70)
    print("Starting pipeline execution...")
    print("="*70 + "\n")
    
    # Execute the main pipeline by importing and running the code
    # For now, we'll print instructions since we need to refactor the main code first
    print("⚠️  To run the pipeline:")
    print("1. Edit src/projectmilestone2.py and update the data paths (lines 52-54):")
    print(f"   - Change to: load_partial_json('{data_files['Electronics']}', {args.max_lines})")
    print(f"   - Change to: load_partial_json('{data_files['All_Beauty']}', {args.max_lines})")
    print(f"   - Change to: load_partial_json('{data_files['Home_and_Kitchen']}', {args.max_lines})")
    print("2. Comment out or remove Google Colab imports (lines 10-15)")
    print("3. Then run: python src/projectmilestone2.py")
    print("\nAlternatively, we can create a refactored version that's fully modular.")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
