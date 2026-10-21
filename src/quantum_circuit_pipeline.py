#!/usr/bin/env python3
"""
Quantum Circuit Extraction Pipeline - Unified System
Extracts and validates quantum circuit diagrams from scientific PDFs

STAGES:
1. Caption-Based Extraction - Find figures with circuit keywords
2. Pre-Filter - Remove code, graphs, heatmaps  
3. Structure Validation - Check for wires, gates, CNOTs
4. Save Results - Organize valid circuits

Usage:
    python quantum_circuit_pipeline.py
"""

import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import csv
import re
from PIL import Image

# Import adaptive filter
try:
    from adaptive_filter import AdaptiveCircuitFilter
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    print("Warning: adaptive_filter.py not found, using manual validation")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output
import sys
# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

# Set CWD to Project Root (parent of src)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if os.path.basename(SRC_DIR) == "src":
    os.chdir(PROJECT_ROOT)
    print(f"Working Directory set to: {PROJECT_ROOT}")

try:
    from arxiv_downloader import ArxivDownloader
except ImportError:
    try:
        from src.arxiv_downloader import ArxivDownloader
    except ImportError:
        pass

# Exam ID
EXAM_ID = "52"

# Input/Output
PAPER_LIST_FILE = os.path.join("extra", f"paper_list_{EXAM_ID}.txt")
OUTPUT_DIR = f"images_{EXAM_ID}"
METADATA_FILE = os.path.join("data", "circuits_metadata.json")
COUNTS_FILE = f"paper_list_counts_{EXAM_ID}.csv"

# Target
TARGET_CIRCUITS = 250

# Rejection folders
REJECT_DIR = os.path.join("data", "rejected")
REJECT_PREFILTER = os.path.join(REJECT_DIR, "stage1_prefilter")
REJECT_STRUCTURE = os.path.join(REJECT_DIR, "stage2_structure")

# Circuit keywords (caption must contain at least one)
CIRCUIT_KEYWORDS = [
    "quantum circuit", "circuit diagram", "gate sequence",
    "quantum gate", "ansatz", "circuit depth", "circuit implementation",
    "teleportation", "bell state", "qubit", "cnot", "hadamard",
    "toffoli", "controlled", "swap gate", "vqe", "qaoa",
    "lattice", "surface code", "array"
]

# Strong indicators (if caption has 2+, very likely a circuit)
STRONG_CIRCUIT_KEYWORDS = [
    "quantum circuit", "circuit diagram", "gate sequence", "circuit implementation",
    "ansatz", "cnot", "hadamard", "toffoli", "swap gate", "controlled gate",
    "circuit depth", "quantum algorithm circuit", "bell circuit"
]

# Specific gate names (high confidence)
GATE_NAMES = [
    "cnot", "hadamard", "pauli", "toffoli", "fredkin", "swap",
    "phase gate", "t gate", "s gate", "rotation gate", "ry", "rx", "rz",
    "controlled-not", "controlled-z", "cz", "cx",
    "h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "u3", "u2", "u1", "id", "measure", "reset"
]

# Quantum Problems / Algorithms
QUANTUM_PROBLEMS = [
    "shor", "grover", "vqe", "qaoa", "teleportation", "deutsch-jozsa", 
    "bernstein-vazirani", "simon", "phase estimation", "amplitude amplification",
    "quantum fourier transform", "qft", "variational", "eigensolver", "maxcut",
    "knapsack", "traveling salesman", "factorization", "search algorithm",
    "error correction", "surface code", "toric code", "color code",
    "quantum simulation", "ising model", "heisenberg model", "hubbard model"
]

# Exclude keywords (reject if caption contains these)
EXCLUDE_KEYWORDS = [
    "plot", "graph", " vs ", "versus", "efficiency", "fidelity",
    "performance", "error rate", "scaling", "histogram", "benchmark",
    "comparison", "chart", "waveform", "time series", "spectrum"
]

# Code detection keywords
CODE_KEYWORDS = [
    'for', 'if', 'then', 'else', 'while', 'do', 'function',
    'def', 'class', 'import', 'return', 'algorithm', 'procedure'
]

def score_caption_confidence(caption):
    """
    Score caption for circuit confidence (0-100)
    Higher score = more likely to be actual circuit
    """
    caption_lower = caption.lower()
    
    # Count keyword matches
    circuit_matches = sum(1 for kw in CIRCUIT_KEYWORDS if kw in caption_lower)
    strong_matches = sum(1 for kw in STRONG_CIRCUIT_KEYWORDS if kw in caption_lower)
    gate_matches = sum(1 for kw in GATE_NAMES if kw in caption_lower)
    exclude_matches = sum(1 for kw in EXCLUDE_KEYWORDS if kw in caption_lower)
    
    # Base score
    score = 0
    
    # Strong indicators
    if strong_matches >= 2:
        score += 60  # Very high confidence
    elif strong_matches >= 1:
        score += 40
    elif circuit_matches >= 2:
        score += 30
    elif circuit_matches >= 1:
        score += 20
    
    # Gate names bonus
    score += min(gate_matches * 15, 40)  # Up to +40 for gates
    
    # Penalties
    if exclude_matches > 0:
        score -= 50
    
    # Caption length factor (circuits usually have concise captions)
    if len(caption) > 300:
        score -= 10  # Very long captions are often not circuits
    
    return max(0, min(100, score)), {
        "circuit_kw": circuit_matches,
        "strong_kw": strong_matches,
        "gate_kw": gate_matches,
        "exclude_kw": exclude_matches
    }

# ============================================================================
# LOAD NOISE KEYWORDS (from metadata)
# ============================================================================

NOISE_KEYWORDS = []

def load_noise_keywords():
    """Load noise keywords from metadata file"""
    global NOISE_KEYWORDS
    metadata_file = os.path.join("data", "keyword_metadata.json")
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                NOISE_KEYWORDS = metadata.get("noise_keywords", [])
                print(f"✓ Loaded {len(NOISE_KEYWORDS)} noise keywords for filtering")
        except Exception as e:
            print(f"Warning: Could not load {metadata_file}: {e}")
    else:
        print(f"Note: {metadata_file} not found. Run analyze_keywords.py first.")

# ============================================================================
# STAGE 1: CAPTION-BASED EXTRACTION
# ============================================================================

def is_caption_relevant(text):
    """Check if caption mentions circuits and doesn't have exclude keywords"""
    text_lower = text.lower()
    
    has_circuit = any(kw in text_lower for kw in CIRCUIT_KEYWORDS)
    if not has_circuit:
        return False
    
    has_exclude = any(kw in text_lower for kw in EXCLUDE_KEYWORDS)
    if has_exclude:
        return False
    
    # NEW: Check for noise keywords
    # If caption contains ONLY noise keywords (no specific circuit terms), reject
    if NOISE_KEYWORDS:
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Filter out very common words
        content_words = [w for w in words if len(w) > 2]
        
        if len(content_words) > 0:
            noise_count = sum(1 for w in content_words if w in NOISE_KEYWORDS)
            noise_ratio = noise_count / len(content_words)
            
            # If >80% of words are noise keywords, caption is too generic
            if noise_ratio > 0.8:
                return False
    
    return True

def get_image_above_caption(page, caption_rect):
    """Find and extract the image/diagram above a caption"""
    page_width = page.rect.width
    
    search_top = max(0, caption_rect.y0 - 800)
    search_bottom = caption_rect.y0
    
    col_x0 = max(0, caption_rect.x0 - 50)
    col_x1 = min(page_width, caption_rect.x1 + 50)
    
    element_rects = []
    
    for drawing in page.get_drawings():
        r = drawing['rect']
        if (search_top < r.y0 and r.y1 < search_bottom + 10 and
            col_x0 <= r.x0 and r.x1 <= col_x1):
            element_rects.append(r)
    
    for img_info in page.get_images():
        xref = img_info[0]
        try:
            # Wrap in try-except to handle PyMuPDF errors with specific images
            image_rects = page.get_image_rects(xref)
        except Exception as e:
            # print(f"Warning: Could not get rects for image xref {xref} on page {page.number}: {e}")
            continue
            
        for r in image_rects:
            if (search_top < r.y0 and r.y1 < search_bottom + 10 and
                col_x0 <= r.x0 and r.x1 <= col_x1):
                element_rects.append(r)
    
    if element_rects:
        x0 = min(r.x0 for r in element_rects)
        y0 = min(r.y0 for r in element_rects)
        x1 = max(r.x1 for r in element_rects)
        y1 = max(r.y1 for r in element_rects)
        
        bbox = fitz.Rect(x0, y0, x1, y1)
        if bbox.height > 30 and bbox.width > 30:
            return bbox
    
    return None

def trim_whitespace(img_bgr):
    """Remove whitespace borders from image"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    if x_min >= x_max:
        return None
    
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img_bgr.shape[1], x_max + padding)
    y_max = min(img_bgr.shape[0], y_max + padding)
    
    return img_bgr[y_min:y_max, x_min:x_max]

# ============================================================================
# STAGE 2: PRE-FILTER (Fast rejection of obvious non-circuits)
# ============================================================================

def prefilter_code(img):
    """Detect if image is code/pseudocode"""
    try:
        import pytesseract
        from PIL import Image as PILImage
        
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil_img).lower()
        
        code_score = sum(1 for kw in CODE_KEYWORDS if kw in text)
        text_length = len(text.strip())
        
        # Very strict: only reject obvious code
        
        # If it has circuit keywords, it's NOT code
        if "quantum" in text or "circuit" in text or "qubit" in text:
            return False, "pass"
            
        if code_score >= 8:
             return True, f"code (score: {code_score})"
             
        if text_length > 1000 and code_score >= 2:
             return True, f"code (len: {text_length}, score: {code_score})"
             
        return False, "pass"
    except:
        return False, "pass"

def prefilter_heatmap(img):
    """Detect if image is a heatmap/colormap"""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0
        
        # High saturation = colorful heatmap
        if avg_saturation > 0.55:
            return True, f"heatmap (sat: {avg_saturation:.2f})"
        return False, "pass"
    except:
        return False, "pass"

def prefilter_graph(img):
    """Detect if image is a plot/graph/chart - ENHANCED"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=int(min(w, h) * 0.25), maxLineGap=15)
        
        h_lines, v_lines = 0, 0
        long_h_lines, long_v_lines = 0, 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if abs(y1 - y2) < 8:  # Horizontal
                    h_lines += 1
                    if length > w * 0.5:  # Long horizontal (axis)
                        long_h_lines += 1
                        
                if abs(x1 - x2) < 8:  # Vertical
                    v_lines += 1
                    if length > h * 0.5:  # Long vertical (axis)
                        long_v_lines += 1
        
        # === 1. STRONG GRID DETECTION ===
        has_strong_grid = (h_lines >= 4 and v_lines >= 4)
        
        # === 2. AXES DETECTION (plots have prominent L-shape axes) ===
        has_axes = (long_h_lines >= 1 and long_v_lines >= 1)
        
        # === 3. CURVE DENSITY (plots have smooth curves, circuits have discrete boxes) ===
        edge_density = np.sum(edges > 0) / edges.size
        has_smooth_curves = (edge_density < 0.10 and edge_density > 0.02)
        
        # === 4. DETECT WAVEFORMS/TIME-SERIES ===
        # Look for horizontal regions with oscillating patterns
        mid_row = gray[h//2, :]
        # Check for variation (waveforms oscillate)
        if len(mid_row) > 50:
            std_dev = np.std(mid_row)
            mean_val = np.mean(mid_row)
            variation = std_dev / max(mean_val, 1)
            has_waveform = (variation > 0.15 and h_lines >= 2)
        else:
            has_waveform = False
        
        # === 5. MULTI-PANEL DETECTION ===
        # Plots often have multiple subplots side-by-side
        # Look for vertical dividers
        vertical_edge_projection = np.sum(edges, axis=0)
        peaks = []
        threshold = np.mean(vertical_edge_projection) * 2
        for i in range(10, len(vertical_edge_projection) - 10):
            if vertical_edge_projection[i] > threshold:
                peaks.append(i)
        
        # Multiple peaks suggest multi-panel
        has_panels = len(peaks) > 3
        
        # === 6. CHECK FOR TICK MARKS ===
        # Plots have small tick marks along axes
        # Count small horizontal/vertical lines near edges
        edge_margin = 30
        left_edge = edges[:, :edge_margin]
        bottom_edge = edges[-edge_margin:, :]
        
        left_ticks = np.sum(left_edge) / max(left_edge.size, 1)
        bottom_ticks = np.sum(bottom_edge) / max(bottom_edge.size, 1)
        has_ticks = (left_ticks > 0.05 or bottom_ticks > 0.05)
        
        # === DECISION LOGIC ===
        # Reject if ANY strong indicator of plot/graph
        if has_strong_grid:
            # CHECK FOR LATTICE NODES (Quantum dots/qubits)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            node_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 20 < area < 500:  # Small nodes
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.6:  # Circle-like
                            node_count += 1
            
            # If we see many nodes in a grid, it's likely a lattice circuit
            if node_count >= 6:
                 return False, f"pass (lattice grid with {node_count} nodes)"
            
            # CHECK FOR DENSE PARALLEL WIRES (Quantum Circuit Characteristic)
            # If h_lines are dominant (qubit lines) and evenly spaced, it's a circuit
            if h_lines >= 6:
                 # Check spacing variance
                 y_coords = sorted([line[0][1] for line in lines if abs(line[0][1] - line[0][3]) < 8])
                 if len(y_coords) > 2:
                     spacings = []
                     for i in range(len(y_coords) - 1):
                         diff = y_coords[i+1] - y_coords[i]
                         if diff > 5: # Ignore close duplicates
                            spacings.append(diff)
                     if len(spacings) > 3:
                         # Coefficient of variation
                         cv = np.std(spacings) / np.mean(spacings)
                         
                         # FIX: Only accept parallel wires if vertical lines are NOT dominant (avoiding plot grids)
                         # If it's a grid (v_lines ~ h_lines), it's a plot. Circuits have more horizontals.
                         is_grid_like = v_lines > h_lines * 0.6
                         
                         if cv < 0.6 and not is_grid_like: # Relatively uniform spacing AND not a full grid
                             return False, f"pass (parallel wires: {len(y_coords)})"
                         
                         # If it IS grid-like, we need stronger evidence (e.g. very low CV = perfect spacing, or many gates)
                         # But let's reject by default if it's grid-like and passed the node check 


            return True, f"plot (grid: {h_lines}x{v_lines})"
        
        # PRIORITIZE PLOT FEATURES
        if has_axes and has_ticks:
            return True, f"plot (axes+ticks)"

        if has_panels and h_lines > 2:
             return True, f"multi-panel plot"

        if has_axes and has_smooth_curves:
            return True, f"plot (axes+curves)"
        
        if has_waveform:
            return True, f"waveform/time-series"
        
        if has_axes and has_ticks:
            return True, f"plot (axes+ticks)"
        
        if has_panels and h_lines >= 3:
            return True, f"multi-panel plot"
        
        # Looser check: moderate lines + smooth curves
        if h_lines >= 3 and v_lines >= 2 and has_smooth_curves:
            return True, f"plot (lines+curves)"
        
        return False, "pass"
        
    except Exception as e:
        return False, "pass"

# ============================================================================
# STAGE 3: CIRCUIT STRUCTURE VALIDATION
# ============================================================================

def validate_circuit_structure(img):
    """
    Perform a strict manual validation of the image structure.
    Checks for:
    1. Parallel Horizontal Lines (Wires)
    2. Box-like contours on wires (Gates)
    3. Vertical lines connecting wires (CNOT/Control)
    4. Pixel Density (rejects photos/heatmaps)
    5. Filled Gates (rejects markers/cells)
    
    :param img: OpenCV image (BGR)
    :return: (is_valid, reason, details_dict)
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        score = 0
        reason = ""
        penalty_reason = ""
        
        # Detect lines (Canny Edges -> Hough Transform)
        edges = cv2.Canny(gray, 50, 150)
        
        # === PIXEL DENSITY CHECK ===
        # Rejects heatmaps, photos, and dense plots.
        # Logic: Circuit diagrams are sparse (mostly white background, few edges).
        total_pixels = h * w
        white_pixels = np.sum(gray > 220)
        bg_ratio = white_pixels / total_pixels
        
        edge_pixels = np.sum(edges > 0)
        edge_ratio = edge_pixels / total_pixels
        
        # Reject if background is too dark (<60% white) OR too busy (>15% edges)
        is_dense_image = (bg_ratio < 0.60) or (edge_ratio > 0.15)
        
        # Detect Lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=int(w * 0.3), maxLineGap=20)
        
        horizontal_wires = []
        vertical_connectors = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Horizontal wires (qubit lines)
                if abs(y1 - y2) < 10 and length > w * 0.25:
                    horizontal_wires.append(y1)
                
                # Vertical connectors (CNOT/Controls)
                if abs(x1 - x2) < 10 and length > 30:
                    vertical_connectors.append(x1)
        
        # Count PARALLEL wires (essential for mult-qubit circuits)
        horizontal_wires.sort()
        parallel_count = 0
        for i in range(len(horizontal_wires) - 1):
            spacing = abs(horizontal_wires[i+1] - horizontal_wires[i])
            if 20 < spacing < 150: # Reasonable spacing
                parallel_count += 1
        
        has_wires = len(horizontal_wires) >= 2 and parallel_count >= 1
        
        # === GATE DETECTION ===
        # Find contours that look like gates (boxes/circles)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gate_count = 0
        gate_rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (too small) and frames (too big)
            if 100 < area < 5000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = cw / float(ch) if ch > 0 else 0
                # Gates are roughly square or rectangular
                if 0.5 < aspect_ratio < 2.0 and 10 < cw < 100:
                    gate_count += 1
                    gate_rects.append((x,y,cw,ch)) # Store rect
        
        # CHECK GATE-WIRE INTERSECTION
        # Real gates sit ON the wires. Plot legends float elsewhere.
        gates_on_wires = 0
        if gate_rects and horizontal_wires:
            for gx, gy, gw, gh in gate_rects:
                center_y = gy + gh // 2
                # Check if any wire is closely aligned with the gate center
                if any(abs(wy - center_y) < gh * 0.6 for wy in horizontal_wires):
                    gates_on_wires += 1
        
        # === FILLED GATE CHECK ===
        # Defense against Heatmaps and Scatter Plots.
        # Logic: Quantum gates are white with text. Heatmap cells/markers are solid color.
        filled_gates = 0
        if gate_rects:
             for gx, gy, gw, gh in gate_rects:
                 # Look at the pixels inside the box
                 roi = gray[gy+2:gy+gh-2, gx+2:gx+gw-2]
                 if roi.size > 0:
                     avg_val = np.mean(roi)
                     if avg_val < 210: # If it's dark/gray, it's "filled"
                         filled_gates += 1
        
        # Reject if >50% of the "gates" are actually solid blocks
        filled_ratio = filled_gates / max(1, gate_count)
        if filled_ratio > 0.5 and gate_count > 2:
             score -= 3
             penalty_reason += f" filled_gates({filled_ratio:.2f})"
        
        # Verify gates are on wires (reject legend boxes)
        on_wire_ratio = gates_on_wires / max(1, gate_count)
        if gate_count > 0 and on_wire_ratio < 0.2 and len(horizontal_wires) > 4:
            # It has wires, but boxes are floating -> Likely a grid plot
            has_gates = False 
        else:
            has_gates = gate_count >= 2
        
        has_cnot = len(vertical_connectors) >= 1
        
        # --- SCORING ---
        if has_wires: score += 3
        if has_gates: score += 2
        if has_cnot: score += 1
        
        # --- OCR CHECK ---
        # Look for plot-specific keywords (Time, Error, vs)
        try:
            import pytesseract
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_text = pytesseract.image_to_string(pil_img).lower()
            
            plot_words = ["time", "error", "probability", "fidelity", "efficiency", "vs", "versus", "count"]
            has_plot_text = any(w in img_text.split() for w in plot_words)
            
            # If it says "Error vs Time", it's a plot, unless it has HUGE structure
            if has_plot_text and gate_count < 5:
                score -= 3
                reason = "plot_text_detected"
        except:
            pass  # OCR failure is ignored
            
        # Apply penalties
        if is_dense_image:
             score -= 4
             penalty_reason += f" dense_image (bg:{bg_ratio:.2f}, edge:{edge_ratio:.2f})"
        else:
             penalty_reason += f" (bg:{bg_ratio:.2f}, edge:{edge_ratio:.2f})"
        
        # --- FINAL DECISION ---
        is_valid = score >= 4
        
        if not has_wires:
            reason = "no_parallel_wires" + penalty_reason
        elif not has_gates:
            reason = "no_gate_symbols" + penalty_reason
        elif is_dense_image:
             reason = "dense_image" + penalty_reason
        elif score < 4:
             reason = f"low_score (score: {score}/6)" + penalty_reason
        else:
            reason = f"valid (score: {score}/6)" + penalty_reason
        
        return is_valid, reason, {
            "wires": len(horizontal_wires),
            "gates": gate_count,
            "cnot": len(vertical_connectors),
            "score": score
        }
        
    except Exception as e:
        return False, f"error: {str(e)}", {}


def extract_quantum_info(text):
    """Extract gates and problems from text"""
    found_gates = set()
    found_problems = set()
    
    text_lower = text.lower()
    
    for gate in GATE_NAMES:
        # Use word boundaries to avoid partial matches (e.g., 's' in 'this')
        if f" {gate} " in text_lower or text_lower.startswith(f"{gate} ") or text_lower.endswith(f" {gate}"):
            found_gates.add(gate)
            
    for prob in QUANTUM_PROBLEMS:
        if prob in text_lower:
            found_problems.add(prob)
            
    # Heuristics for problems not in list
    if "algorithm" in text_lower:
        # Try to find word before algorithm
        words = text_lower.split()
        if "algorithm" in words:
            idx = words.index("algorithm")
            if idx > 0:
                # Add the two-word phrase if it's not too generic
                prev_word = words[idx-1]
                if prev_word not in ["a", "an", "the", "this", "that", "new", "novel", "efficient", "quantum"]:
                    found_problems.add(f"{prev_word} algorithm")

    return list(found_gates), list(found_problems)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_pdf(pdf_data, arxiv_id, metadata_store, stats):
    """
    Core function to process a single PDF file (in-memory).
    
    Steps:
    1. Parse PDF using PyMuPDF (fitz)
    2. Search for captions starting with "Figure" or "Fig."
    3. Extract the image ABOVE the caption
    4. Convert to OpenCV format and trim whitespace
    5. Run Cascade Filtering (Code -> Heatmap -> Graph)
    6. Run Adaptive/Manual Validation (Structure check)
    7. Save to disk if valid and update metadata
    
    :param pdf_data: Raw bytes of the PDF file
    :param arxiv_id: The ID string (e.g. '2501.00123')
    :param metadata_store: Dictionary to store results
    :param stats: Dictionary to track counts (found, rejected, saved)
    """
    try:
        # Open PDF from memory (no temp files)
        doc = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception as e:
        print(f"  ✗ Error opening PDF stream: {e}")
        stats["failed"] += 1
        return 0
    
    # Regex to find Figure captions (e.g., "Figure 1", "Fig. 2")
    caption_pattern = re.compile(r"^Fig(ure)?\.?\s*(\d+)", re.IGNORECASE)
    saved_count = 0
    
    # Iterate through every page
    for page_num, page in enumerate(doc):
        # precise text block extraction
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1]) # Sort by Y-coordinate (top to bottom)
        
        for i, block in enumerate(blocks):
            text = block[4].strip().replace("\n", " ")
            
            # Check if this block looks like a caption
            match = caption_pattern.match(text)
            if not match:
                continue
            
            # Merge multi-line captions
            full_caption = text
            for k in range(1, 3):
                if i + k >= len(blocks):
                    break
                next_text = blocks[i + k][4].strip()
                if caption_pattern.match(next_text) or len(next_text) < 3:
                    break
                gap = blocks[i + k][1] - block[3]
                if gap > 20:
                    break
                full_caption += " " + next_text.replace("\n", " ")
            
            # Semantic Check: Does the caption sound like a circuit?
            # (e.g., "Quantum circuit for...", "Schematic of...")
            if not is_caption_relevant(full_caption):
                continue
            
            stats["captions_found"] += 1
            
            fig_num = match.group(2)
            caption_rect = fitz.Rect(block[:4])
            
            # Extract the Image Region
            # (Heuristic: The image is usually immediately above the caption)
            bbox = get_image_above_caption(page, caption_rect)
            
            # Skip if no image found or it's too small (icon/noise)
            if bbox is None or bbox.height < 40 or bbox.width < 50:
                continue
            
            try:
                # Render image
                clip_rect = fitz.Rect(bbox.x0 - 5, bbox.y0 - 5, bbox.x1 + 5, bbox.y1 + 5)
                clip_rect = clip_rect & page.rect
                pix = page.get_pixmap(clip=clip_rect, dpi=200)
                
                # Convert to OpenCV
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                if pix.n == 4:
                    img = cv2.cvtColor(img_data.reshape((pix.height, pix.width, 4)), cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(img_data.reshape((pix.height, pix.width, 3)), cv2.COLOR_RGB2BGR)
                
                # Trim
                trimmed = trim_whitespace(img)
                if trimmed is None or trimmed.shape[0] < 50 or trimmed.shape[1] < 50:
                    continue
                
                stats["extracted"] += 1
                filename = f"{arxiv_id}_p{page_num}_fig{fig_num}.png"
                
                # === CAPTION CONFIDENCE SCORING (NEW!) ===
                caption_confidence, caption_details = score_caption_confidence(full_caption)
                
                # === STAGE 2: PRE-FILTER (Fast Rejection) ===
                # If the AI caption model is uncertain (<60%), we run strict visual checks.
                # This filters out code screenshots, heatmaps, and plots.
                if caption_confidence < 60:
                    is_code, code_reason = prefilter_code(trimmed)
                    if is_code:
                        cv2.imwrite(os.path.join(REJECT_PREFILTER, filename), trimmed)
                        stats["rejected_code"] += 1
                        print(f"    ❌ {filename}: {code_reason}")
                        continue
                    
                    is_heatmap, heat_reason = prefilter_heatmap(trimmed)
                    if is_heatmap:
                        cv2.imwrite(os.path.join(REJECT_PREFILTER, filename), trimmed)
                        stats["rejected_heatmap"] += 1
                        print(f"    ❌ {filename}: {heat_reason}")
                        continue
                    
                    is_graph, graph_reason = prefilter_graph(trimmed)
                    if is_graph:
                        cv2.imwrite(os.path.join(REJECT_PREFILTER, filename), trimmed)
                        stats["rejected_graph"] += 1
                        print(f"    ❌ {filename}: {graph_reason}")
                        continue
                
                # === EXTRACT CONTEXT TEXT ===
                # Grab text above and below the image to help with metadata/understanding
                context_text = {
                    "above": "",
                    "caption": full_caption,
                    "below": ""
                }
                
                descriptions = []
                text_positions = []
                
                # Add caption as first description
                descriptions.append(full_caption)
                # Position relative to page text blocks
                text_positions.append((page_num, i)) 
                
                # Get text above image
                for b_idx, block in enumerate(blocks):
                    block_y = block[1]
                    if bbox.y0 - 200 < block_y < bbox.y0:
                        text_part = block[4].strip()
                        if text_part:
                            context_text["above"] += " " + text_part
                            descriptions.append(text_part)
                            text_positions.append((page_num, b_idx))
                
                # Get text below caption
                caption_bottom = bbox.y1 + 100
                for b_idx, block in enumerate(blocks):
                    block_y = block[1]
                    if bbox.y1 < block_y < caption_bottom:
                         text_part = block[4].strip()
                         if text_part:
                            context_text["below"] += " " + text_part
                            descriptions.append(text_part)
                            text_positions.append((page_num, b_idx))

                # === STAGE 3: ADAPTIVE VALIDATION (LEARNED FROM VALID CIRCUITS) ===
                
                # Use adaptive filter if available
                if 'adaptive_filter' in globals():
                    result = adaptive_filter.is_circuit(trimmed, full_caption)
                    
                    if not result["is_valid"]:
                        cv2.imwrite(os.path.join(REJECT_STRUCTURE, filename), trimmed)
                        stats["rejected_structure"] += 1
                        print(f"    ❌ {filename}: {result['reason']}")
                        continue
                    
                    # SECONDARY CHECK: Run manual validation to catch plots that fool the adaptive filter (e.g. grids)
                    manual_valid, manual_reason, manual_details = validate_circuit_structure(trimmed)
                    
                    # If manual check fails heavily (score < 4), reject unless adaptive confidence is EXTREMELY high (>95)
                    # This catches plots (manual score often < 0 due to penalties) while allowing clear circuits.
                    
                    if not manual_valid and result["visual_score"] < 92:
                         cv2.imwrite(os.path.join(REJECT_STRUCTURE, filename), trimmed)
                         stats["rejected_structure"] += 1
                         print(f"    ❌ {filename}: rejected by manual filter ({manual_reason}) despite V:{result['visual_score']:.0f}")
                         continue

                    # Save with learned scores
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(output_path, trimmed)
                    
                    # Deduplicate and Clean Descriptions
                    final_descriptions = []
                    final_positions = []
                    seen_texts = set()
                    
                    # Normalize full_caption for duplicate check
                    seen_texts.add(full_caption.strip())
                    final_descriptions.append(full_caption)
                    final_positions.append(text_positions[0]) # Caption position
                    
                    for desc, pos in zip(descriptions[1:], text_positions[1:]):  # skip the caption
                        desc_clean = desc.strip()
                        if not desc_clean: continue
                        
                        # Check for duplicates
                        if desc_clean in seen_texts:
                            continue
                        
                        # Check for other figure captions
                        # Regex to find "Figure X" or "Fig. X" at the START of the text
                        fig_match = re.search(r"^(?:Figure|Fig\.?)\s+(\d+)", desc_clean, re.IGNORECASE)
                        if fig_match:
                            found_fig = int(fig_match.group(1))
                            if found_fig != int(fig_num):
                                # This is a caption for a DIFFERENT figure
                                continue
                        
                        seen_texts.add(desc_clean)
                        final_descriptions.append(desc_clean)
                        final_positions.append(pos)
                    
                    metadata_store[filename] = {
                        "caption": full_caption,
                        "page": page_num + 1,  # 1-based page number
                        "arxiv_id": arxiv_id,
                        "descriptions": final_descriptions,
                        "text_positions": final_positions, # Stores (page_num, block_idx) raw, formatted below
                        "validation": {
                            "method": "adaptive_learned",
                            "visual_score": result["visual_score"],
                            "caption_score": result["caption_score"],
                            "confidence": (result["visual_score"] + result["caption_score"]) / 2
                        }
                    }
                    
                    # Extract rich metadata
                    all_text = full_caption + " " + " ".join(final_descriptions)
                    gates, problems = extract_quantum_info(all_text)
                    
                    metadata_store[filename]["quantum_gates"] = gates if gates else ["unknown"]
                    metadata_store[filename]["quantum_problem"] = ", ".join(problems) if problems else "unknown"
                    metadata_store[filename]["figure_number"] = int(fig_num)
                    
                    # Store text positions as (block_index, length)
                    formatted_positions = [(p[1], p[1]+1) for p in final_positions]
                    metadata_store[filename]["text_positions"] = formatted_positions
                    
                    print(f"    saved {filename}: V:{result['visual_score']:.0f} C:{result['caption_score']:.0f}")
                    saved_count += 1
                    stats["saved"] += 1
                    continue
                
                # Context extracted above
                pass
                
                # ASK TEXT QUESTIONS
                full_text = (context_text["above"] + " " + context_text["caption"] + " " + context_text["below"]).lower()
                
                text_score = 0
                text_evidence = []
                
                # Q1: Specific quantum gates?
                gates = ["cnot", "hadamard", "pauli", "toffoli", "swap", "phase", "rx", "ry", "rz", "cz", "cx"]
                gate_mentions = sum(1 for g in gates if g in full_text)
                if gate_mentions >= 2:
                    text_score += 30
                    text_evidence.append(f"{gate_mentions} gates")
                elif gate_mentions >= 1:
                    text_score += 15
                    text_evidence.append(f"{gate_mentions} gate")
                
                # Q2: Circuit structure terms?
                structure_terms = ["circuit depth", "gate sequence", "qubit line", "wire", "layer", "circuit diagram"]
                struct_count = sum(1 for t in structure_terms if t in full_text)
                if struct_count >= 2:
                    text_score += 25
                    text_evidence.append(f"{struct_count} struct")
                elif struct_count >= 1:
                    text_score += 10
                
                # Q3: Implementation/ansatz?
                impl_terms = ["ansatz", "circuit implementation", "quantum algorithm", "vqe", "qaoa", "bell circuit"]
                impl_count = sum(1 for t in impl_terms if t in full_text)
                if impl_count >= 1:
                    text_score += 20
                    text_evidence.append(f"impl:{impl_count}")
                
                # Q4: Avoid plot language?
                plot_terms = ["plot", "graph", "versus", "waveform", "time series", "histogram", "curve"]
                plot_count = sum(1 for t in plot_terms if t in full_text)
                if plot_count > 0:
                    text_score -= 30
                    text_evidence.append(f"plot:-{plot_count}")
                
                # Q5: Qubit count specified?
                qubit_pattern = r'\b\d+\s*qubit'
                if re.search(qubit_pattern, full_text):
                    text_score += 15
                    text_evidence.append("qubits")
                
                text_score = max(0, min(100, text_score))
                
                # ASK VISUAL QUESTIONS
                visual_score = 0
                visual_evidence = []
                
                vis_gray = cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY)
                vis_h, vis_w = vis_gray.shape
                vis_edges = cv2.Canny(vis_gray, 50, 150)
                vis_lines = cv2.HoughLinesP(vis_edges, 1, np.pi/180, threshold=50,
                                           minLineLength=int(vis_w * 0.2), maxLineGap=20)
                
                # Q1: Parallel horizontal wires?
                h_wires = []
                if vis_lines is not None:
                    for line in vis_lines:
                        x1, y1, x2, y2 = line[0]
                        if abs(y1 - y2) < 10 and abs(x2 - x1) > vis_w * 0.25:
                            h_wires.append(y1)
                
                h_wires.sort()
                parallel = 0
                for i in range(len(h_wires) - 1):
                    if 20 < abs(h_wires[i+1] - h_wires[i]) < 150:
                        parallel += 1
                
                if parallel >= 2:
                    visual_score += 35
                    visual_evidence.append(f"{parallel+1}wires")
                elif parallel >= 1:
                    visual_score += 20
                    visual_evidence.append(f"{parallel+1}wires")
                
                # Q2: Gate symbols?
                vis_contours, _ = cv2.findContours(vis_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                gate_like = 0
                for cnt in vis_contours:
                    area = cv2.contourArea(cnt)
                    if 100 < area < 5000:
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        aspect = cw / float(ch) if ch > 0 else 0
                        if 0.5 < aspect < 2.0 and 10 < cw < 100:
                            gate_like += 1
                
                if gate_like >= 3:
                    visual_score += 30
                    visual_evidence.append(f"{gate_like}gates")
                elif gate_like >= 2:
                    visual_score += 15
                
                # Q3: Discrete structure?
                edge_density = np.sum(vis_edges > 0) / vis_edges.size
                if 0.02 < edge_density < 0.20:
                    visual_score += 20
                    visual_evidence.append("discrete")
                elif edge_density < 0.02 or edge_density > 0.30:
                    visual_score -= 20
                    visual_evidence.append(f"density:{edge_density:.2f}")
                
                # Q4: Avoid grid/axes?
                v_lines = 0
                if vis_lines is not None:
                    for line in vis_lines:
                        x1, y1, x2, y2 = line[0]
                        if abs(x1 - x2) < 10 and abs(y2 - y1) > vis_h * 0.4:
                            v_lines += 1
                
                if len(h_wires) > 5 and v_lines > 5:
                    visual_score -= 40
                    visual_evidence.append(f"grid:{len(h_wires)}x{v_lines}")
                
                visual_score = max(0, min(100, visual_score))
                
                # === CONSENSUS DECISION (RELAXED) ===
                # Accept if:
                #   1. HIGH confidence in either modality: text≥50 OR visual≥60
                #   2. MODERATE confidence in both: text≥25 AND visual≥30
                
                high_text = text_score >= 50
                high_visual = visual_score >= 60
                moderate_text = text_score >= 25
                moderate_visual = visual_score >= 30
                
                consensus = (high_text or high_visual) or (moderate_text and moderate_visual)
                
                if not consensus:
                    cv2.imwrite(os.path.join(REJECT_STRUCTURE, filename), trimmed)
                    stats["rejected_structure"] += 1
                    reason = f"T:{text_score} V:{visual_score}"
                    if not moderate_text:
                        reason += " T✗"
                    if not moderate_visual:
                        reason += " V✗"
                    print(f"    ❌ {filename}: {reason}")
                    continue
                
                # === SAVE VALID CIRCUIT ===
                output_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(output_path, trimmed)
                
                metadata_store[filename] = {
                    "caption": full_caption,
                    "page": page_num + 1,  # 1-based, same as the other branch
                    "arxiv_id": arxiv_id,
                    "descriptions": descriptions,
                    "text_positions": text_positions,
                    "validation": {
                        "text_score": text_score,
                        "visual_score": visual_score,
                        "text_evidence": text_evidence,
                        "visual_evidence": visual_evidence,
                        "confidence": (text_score + visual_score) / 2
                    }
                }
                
                conf = (text_score + visual_score) / 2
                print(f"    saved {filename}: T:{text_score} V:{visual_score} [{conf:.0f}]")
                saved_count += 1
                stats["saved"] += 1
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
    
    doc.close()
    return saved_count

def save_counts_csv(counts_data):
    """Save paper counts to CSV"""
    with open(COUNTS_FILE, 'w') as f:
        for pid, count in counts_data.items():
            val = str(count) if count is not None else ""
            f.write(f"{pid},{val}\n")

def main():
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REJECT_PREFILTER, exist_ok=True)
    os.makedirs(REJECT_STRUCTURE, exist_ok=True)
    
    # Init downloader
    downloader = ArxivDownloader(download_dir=os.path.join("data", "temp_pdfs"), delay=(10, 20))
    
    # Load noise keywords
    load_noise_keywords()
    
    # Load adaptive filter
    global adaptive_filter
    if ADAPTIVE_AVAILABLE:
        try:
            adaptive_filter = AdaptiveCircuitFilter()
        except Exception:
            adaptive_filter = None
    else:
        adaptive_filter = None
        
    # Load Paper List
    if not os.path.exists(PAPER_LIST_FILE):
        print(f"Error: Paper list {PAPER_LIST_FILE} not found!")
        return
        
    with open(PAPER_LIST_FILE, 'r') as f:
        paper_ids = [line.strip().replace("arXiv:", "") for line in f if line.strip()]
        
    print(f"Loaded {len(paper_ids)} papers from {PAPER_LIST_FILE}")
    
    # Load existing counts
    counts_data = {}
    if os.path.exists(COUNTS_FILE):
        try:
            # Load counts from CSV with BOM handling
            with open(COUNTS_FILE, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 1:
                        pid = parts[0]
                        val = parts[1] if len(parts) > 1 and parts[1] else None
                        counts_data[pid] = int(val) if val is not None else None
        except:
            pass
            
    # Initialize counts
    for pid in paper_ids:
        if pid not in counts_data:
            counts_data[pid] = None
            
    # Load metadata with BOM handling
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8-sig') as f:
            metadata_store = json.load(f)
    else:
        metadata_store = {}
        
    # Check progress
    existing_images = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    total_saved = len(existing_images)
    print(f"Already collected: {total_saved} images")
    
    # === MAIN LOOP ===
    for idx, pid in enumerate(paper_ids):
        if total_saved >= TARGET_CIRCUITS:
            print(f"\n🎯 TARGET REACHED: {total_saved} circuits!")
            break
            
        if counts_data.get(pid) is not None:
             continue
            
        print(f"\nProcessing [{idx+1}/{len(paper_ids)}] arXiv:{pid} ...")
        
        # Download bytes (in-memory)
        pdf_bytes = downloader.fetch_pdf_bytes(pid)
        if not pdf_bytes:
            counts_data[pid] = 0
            save_counts_csv(counts_data)
            continue
            
        try:
            stats = {"saved": 0, "captions_found": 0, "extracted": 0, "rejected_code": 0, "rejected_heatmap": 0, "rejected_graph": 0, "rejected_structure": 0, "failed": 0}
            
            saved_from_paper = process_pdf(pdf_bytes, pid, metadata_store, stats)
            
            counts_data[pid] = saved_from_paper
            total_saved += saved_from_paper
            print(f"  -> Found {saved_from_paper} valid circuits. Total: {total_saved}")
            
            save_counts_csv(counts_data)
            
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata_store, f, indent=2)
                
        except Exception as e:
            print(f"  ✗ Error processing {pid}: {e}")
            counts_data[pid] = 0
            save_counts_csv(counts_data)
            
        # Optional: cleanup PDF
        # downloader.cleanup(pdf_path) 

    print("\nDONE.")

if __name__ == "__main__":
    main()
