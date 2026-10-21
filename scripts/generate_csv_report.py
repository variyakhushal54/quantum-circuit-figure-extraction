# This script generates a CSV report of the image extraction process.
# It matches the requirements for Exam ID 52.

import os   # For file and directory operations
import csv  # For creating the CSV report

# --- CONFIGURATION (Exam ID 52) ---
PAPER_LIST_FILE = "paper_list_52.txt"  
IMAGE_FOLDER = "images_52"          
OUTPUT_CSV = "paper_list_counts_52.csv"

def main():
    """
    Main function to generate the report.
    Reads the paper list, counts extracted images, and writes to CSV.
    """
    # 1. READ THE PAPER LIST
    if not os.path.exists(PAPER_LIST_FILE):
        print(f"Error: Could not find {PAPER_LIST_FILE}")
        return

    clean_ids = []
    with open(PAPER_LIST_FILE, "r") as f:
        for line in f:
            clean_id = line.strip().replace("arXiv:", "").replace("arxiv:", "")
            if clean_id:
                clean_ids.append(clean_id)
    
    print(f"Loaded {len(clean_ids)} papers from list.")

    # 2. COUNT IMAGES IN FOLDER
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Could not find folder {IMAGE_FOLDER}")
        return

    image_counts = {}
    
    files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]
    print(f"Found {len(files)} images in folder.")
    
    for filename in files:
        # Expected format: 2402.15026_p3_fig1.png
        try:
            paper_id = filename.split("_")[0]
            
            if paper_id in image_counts:
                image_counts[paper_id] += 1
            else:
                image_counts[paper_id] = 1
        except:
            print(f"Skipping filename: {filename}")

    # 3. DETERMINE PROGRESS
    last_scanned_index = -1
    for i, pid in enumerate(clean_ids):
        if pid in image_counts:
            last_scanned_index = i

    print(f"Stopped around paper #{last_scanned_index + 1}.")

    # 4. GENERATE CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Paper ID", "Images Found"]) # Header

        for i, pid in enumerate(clean_ids):
            if pid in image_counts:
                # Images found -> Write count
                writer.writerow([pid, image_counts[pid]])
                
            elif i <= last_scanned_index:
                # Scanned but no images -> Write 0
                writer.writerow([pid, 0])
                
            else:
                # Not reached -> Write blank
                writer.writerow([pid, ""])

    print("="*50)
    print(f"SUCCESS! Created '{OUTPUT_CSV}'")
    print("="*50)

if __name__ == "__main__":
    main()
