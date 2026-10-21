import os

# CONFIG
GROUND_TRUTH_VALID = "dataset_verified/valid"
GROUND_TRUTH_INVALID = "dataset_verified/invalid"

PREDICTED_VALID = "images_52"
TRASH_FOLDERS = ["trash_images", "trash_images_nlp"]

def params():
    print("--- EVALUATION CONFIG ---")
    print(f"Ground Truth Valid: {GROUND_TRUTH_VALID}")
    print(f"Ground Truth Invalid: {GROUND_TRUTH_INVALID}")
    print(f"Predicted Valid: {PREDICTED_VALID}")
    print(f"Predicted Trash: {TRASH_FOLDERS}")
    print("-------------------------")

def get_files_in_tree(folder):
    """Recursively get all primitive file names (basename) in a folder try tree."""
    file_set = set()
    if not os.path.exists(folder):
        return file_set
        
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".png"):
                file_set.add(f)
    return file_set

def main():
    params()
    
    # 1. Load Ground Truth Labels
    gt_valid = get_files_in_tree(GROUND_TRUTH_VALID)
    gt_invalid = get_files_in_tree(GROUND_TRUTH_INVALID)
    
    print(f"Ground Truth Loaded: {len(gt_valid)} Valid, {len(gt_invalid)} Invalid images.")
    
    # 2. Load Predictions
    pred_valid = get_files_in_tree(PREDICTED_VALID)
    
    pred_trash = set()
    for tf in TRASH_FOLDERS:
        pred_trash.update(get_files_in_tree(tf))
    
    # 3. Calculate Metrics
    tp = 0 # True Positive (Valid in Valid)
    tn = 0 # True Negative (Invalid in Trash)
    fp = 0 # False Positive (Invalid in Valid) - BAD (Garbage in dataset)
    fn = 0 # False Negative (Valid in Trash) - BAD (Lost data)
    
    # Analyze Valid Ground Truth
    for f in gt_valid:
        if f in pred_valid:
            tp += 1
        elif f in pred_trash:
            fn += 1
            # print(f"[FALSE NEGATIVE] {f} (Valid Image -> Trash)")
        else:
            # File missing? Maybe lost or not in the processed batch?
            pass

    # Analyze Invalid Ground Truth
    for f in gt_invalid:
        if f in pred_trash:
            tn += 1
        elif f in pred_valid:
            fp += 1
            # print(f"[FALSE POSITIVE] {f} (Invalid Image -> Accepted)")
        else:
            pass

    # REPORT
    print("\n" + "="*40)
    print("       FILTER VERIFICATION REPORT")
    print("="*40)
    
    # RECALL (Preservation)
    recall = tp / len(gt_valid) if len(gt_valid) > 0 else 0
    print(f"\n1. RECALL (How many valid images did we keep?)")
    print(f"   - Kept (TP):      {tp}")
    print(f"   - Trashed (FN):   {fn}")
    print(f"   -> Recall:        {recall:.1%}")
    
    # SPECIFICITY (Rejection)
    specificity = tn / len(gt_invalid) if len(gt_invalid) > 0 else 0
    print(f"\n2. SPECIFICITY (How many bad images did we trash?)")
    print(f"   - Trashed (TN):   {tn}")
    print(f"   - Kept (FP):      {fp}")
    print(f"   -> Rejection Rate:{specificity:.1%}")

    # PRECISION (Purity)
    # Valid Ratio in the output folder (based only on labeled data)
    total_predicted_valid_labeled = tp + fp
    precision = tp / total_predicted_valid_labeled if total_predicted_valid_labeled > 0 else 0
    
    print(f"\n3. PRECISION (How pure is the output folder?)")
    print(f"   - Valid (TP):     {tp}")
    print(f"   - Garbage (FP):   {fp}")
    print(f"   -> VALID IMAGE RATIO: {precision:.1%}")
    
    print("\n" + "="*40)

if __name__ == "__main__":
    main()
