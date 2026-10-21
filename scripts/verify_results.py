"""Recompute the headline numbers from the files in this repository.

Reads only the progress ledger (paper_list_counts_52.csv) and the metadata
file (data/circuits_metadata.json). It does not download or process any PDFs.

Usage:
    python scripts/verify_results.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS_FILE = os.path.join(ROOT, "paper_list_counts_52.csv")
METADATA_FILE = os.path.join(ROOT, "data", "circuits_metadata.json")


def read_counts():
    """Return a list with one entry per paper: the number of images, or None if not processed."""
    counts = []
    with open(COUNTS_FILE, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            value = row[1].strip()
            counts.append(int(value) if value.isdigit() else None)
    return counts


def main():
    counts = read_counts()
    processed = [c for c in counts if c is not None]
    total_images = sum(processed)
    papers_with_images = sum(1 for c in processed if c > 0)

    with open(METADATA_FILE, encoding="utf-8-sig") as f:
        metadata = json.load(f)
    metadata_papers = {entry["arxiv_id"] for entry in metadata.values()}

    print("Progress ledger (paper_list_counts_52.csv)")
    print(f"  papers processed:           {len(processed)}")
    print(f"  images saved (sum):         {total_images}")
    print(f"  papers with >= 1 image:     {papers_with_images}")
    print(f"  share of papers with image: {papers_with_images / len(processed):.1%}")
    print(f"  images per paper:           {total_images / len(processed):.3f}")
    print()
    print("Metadata (data/circuits_metadata.json)")
    print(f"  image entries:              {len(metadata)}")
    print(f"  distinct arXiv papers:      {len(metadata_papers)}")
    print()
    matches = total_images == len(metadata) and papers_with_images == len(metadata_papers)
    print("Ledger and metadata agree:", "yes" if matches else "NO, check the files")


if __name__ == "__main__":
    main()
