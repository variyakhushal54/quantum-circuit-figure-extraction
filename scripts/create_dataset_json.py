# This script processes extracted images to generate the final dataset JSON.
# It performs OCR to detect gates and merges metadata found during extraction.

import os           # Tool for files and folders
import json         # Tool for handling JSON data format
import pytesseract  # Tool for reading text from images (OCR)
from PIL import Image # Tool for opening and manipulating images

# --- CONFIGURATION ---
EXAM_ID = "52"
IMAGE_FOLDER = f"images_{EXAM_ID}"       # Folder with images
METADATA_FILE = os.path.join("data", "circuits_metadata.json") # Metadata from extraction step
OUTPUT_JSON = os.path.join("data", "quantum_dataset.json")      # Final output file

# Set TESSERACT_CMD to the tesseract executable if it is not on your PATH
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

def get_gates_from_image(image_path):
    """
    Detect quantum gates in an image using OCR.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    list of str
        A list of unique quantum gates detected in the image.
    """
    known_gates = [
        "H", "X", "Y", "Z", "I", "S", "T", "CNOT", 
        "Rx", "Ry", "Rz", "U", "SWAP", "Measure", 
        "QFT", "Oracle", "Diffuser"
    ]
    
    found_gates = []
    
    try:
        img = Image.open(image_path)
        # --psm 6 assumes a single uniform block of text, good for schematic labels
        text = pytesseract.image_to_string(img, config='--psm 6')
        
        words = text.replace("|", " ").replace("-", " ").split()
        
        for word in words:
            # Clean punctuation
            clean_word = word.strip("[](){}<>.,")
            
            if clean_word in known_gates:
                found_gates.append(clean_word)
                
            # Handle variations
            if clean_word == "M":
                found_gates.append("Measure")

    except Exception as e:
        print(f"OCR Error on {image_path}: {e}")
        
    return list(set(found_gates)) 

def guess_problem_type(gates_list):
    """
    Heuristically determine the quantum algorithm based on gates.

    Parameters
    ----------
    gates_list : list of str
        List of detected gates.

    Returns
    -------
    str
        The name of the likely quantum algorithm or problem.
    """
    gates_set = set(gates_list)
    
    if "QFT" in gates_set:
        return "Quantum Fourier Transform"
    if "Oracle" in gates_set and "H" in gates_set:
        return "Grover's Search"
    if "Diffuser" in gates_set:
        return "Grover's Search"
    if "Rx" in gates_set or "Ry" in gates_set:
        return "Variational Quantum Algorithm (VQA)"
    
    return "General Quantum Circuit"

def main():
    """
    Main execution function.
    Combines image analysis (OCR) with extracted metadata to produce the dataset.
    """
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")
        return

    # Load metadata found during extraction (captions, positions)
    extracted_metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            extracted_metadata = json.load(f)
    else:
        print("Warning: No metadata file found. Captions will be empty.")

    dataset = {}
    
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]
    print(f"Processing {len(images)} images for dataset generation...")

    for img_file in images:
        full_path = os.path.join(IMAGE_FOLDER, img_file)
        
        # 1. Parse Metadata from Filename
        # Format: {arxiv_id}_p{page}_fig{fig}.png
        try:
            parts = img_file.replace(".png", "").split("_")
            arxiv_id = parts[0]
            page_num = int(parts[1].replace("p", "")) 
            fig_num = int(parts[2].replace("fig", ""))
        except:
            print(f"Skipping bad filename: {img_file}")
            continue

        # 2. Extract Gates (OCR)
        gates_found = get_gates_from_image(full_path)
        
        # 3. Guess Problem Type
        problem_type = guess_problem_type(gates_found)

        # 4. Convert Metadata to required types
        descriptions = []
        text_positions = []
        
        if img_file in extracted_metadata:
            meta = extracted_metadata[img_file]
            descriptions = meta.get("descriptions", [])
            # Convert text positions (tuples) to list of lists/tuples as required
            # The requirement asks for "tuple (beginning, end)". 
            # Our extraction provides (page, start_block, end_block).
            # We will use that as the definition of position in this context.
            text_positions = meta.get("text_positions", [])

        # 5. Build Entry
        # 5. Build Entry
        entry = {
            "arxiv number": str(arxiv_id),
            "page number": int(page_num),
            "figure number": int(fig_num),
            "quantum gates": gates_found,
            "quantum problem": str(problem_type),
            "descriptions": descriptions,
            "text positions": text_positions 
        }
        
        dataset[img_file] = entry
        
        if len(dataset) % 10 == 0:
            print(f"Processed {len(dataset)}/{len(images)}...")

    # 6. Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=4)

    print("\n" + "="*50)
    print(f"SUCCESS! Dataset saved to '{OUTPUT_JSON}'")
    print(f"Total entries: {len(dataset)}")
    print("="*50)

if __name__ == "__main__":
    main()
