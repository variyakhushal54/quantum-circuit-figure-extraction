#!/usr/bin/env python3
"""
Learn from Valid Circuits Only (One-Class Learning)
Extracts visual features + caption patterns from circuits_final/
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import re

def extract_visual_features(image_path):
    """
    Extract visual features from a circuit image
    
    Returns dict of numerical features
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    features = {}
    
    # === 1. STRUCTURE FEATURES ===
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=int(w * 0.2), maxLineGap=20)
    
    # Count horizontal/vertical lines
    h_lines, v_lines = 0, 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:
                h_lines += 1
            if abs(x1 - x2) < 10:
                v_lines += 1
    
    features['h_line_count'] = h_lines
    features['v_line_count'] = v_lines
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # === 2. SHAPE FEATURES ===
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    box_like = 0
    circle_like = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.7:
                    circle_like += 1
                else:
                    box_like += 1
    
    features['box_count'] = box_like
    features['circle_count'] = circle_like
    
    # === 3. COLOR FEATURES ===
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features['avg_saturation'] = np.mean(hsv[:, :, 1]) / 255.0
    features['avg_brightness'] = np.mean(hsv[:, :, 2]) / 255.0
    
    # === 4. ASPECT RATIO ===
    features['aspect_ratio'] = w / h if h > 0 else 1.0
    
    return features

def extract_caption_patterns(metadata_path):
    """
    Extract common phrases and patterns from validated captions
    """
    if not os.path.exists(metadata_path):
        return {}
    
    with open(metadata_path, 'r', encoding='utf-8-sig') as f:
        metadata = json.load(f)
    
    # Collect all captions
    captions = []
    for item in metadata.values():
        if 'caption' in item and item['caption']:
            captions.append(item['caption'])
    
    # Extract n-grams
    bigrams = []
    trigrams = []
    words_all = []
    
    for caption in captions:
        caption_lower = caption.lower()
        # Remove special chars
        words = re.findall(r'\b\w+\b', caption_lower)
        words_all.extend(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i+1]}")
        
        # Trigrams
        for i in range(len(words) - 2):
            trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    # Count frequencies
    word_freq = Counter(words_all)
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)
    
    return {
        "top_words": word_freq.most_common(50),
        "top_bigrams": bigram_freq.most_common(30),
        "top_trigrams": trigram_freq.most_common(20),
        "total_captions": len(captions)
    }

def learn_from_valid_circuits(circuits_dir=os.path.join("data", "circuits_final"), metadata_file=os.path.join("data", "circuits_metadata.json")):
    """
    Learn visual features and caption patterns from validated circuits
    """
    print("="*70)
    print(" LEARNING FROM VALID CIRCUITS ONLY")
    print("="*70)
    
    if not os.path.exists(circuits_dir):
        print(f"Error: {circuits_dir}/ not found!")
        return None
    
    # Get all circuit images
    circuit_files = [f for f in os.listdir(circuits_dir) if f.endswith('.png')]
    print(f"\nFound {len(circuit_files)} validated circuits")
    
    if len(circuit_files) < 5:
        print("Warning: Need at least 5 circuits for meaningful learning")
    
    # Extract visual features
    print("\nExtracting visual features...")
    all_features = []
    feature_names = None
    
    for img_file in circuit_files:
        img_path = os.path.join(circuits_dir, img_file)
        features = extract_visual_features(img_path)
        
        if features:
            if feature_names is None:
                feature_names = list(features.keys())
            all_features.append(list(features.values()))
    
    print(f"✓ Extracted features from {len(all_features)} images")
    
    # Compute statistics
    features_array = np.array(all_features)
    feature_stats = {}
    
    for i, name in enumerate(feature_names):
        values = features_array[:, i]
        feature_stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
    
    print("\nFeature Statistics:")
    print("-" * 50)
    for name, stats in feature_stats.items():
        print(f"  {name:20s}: μ={stats['mean']:.3f} σ={stats['std']:.3f}")
    
    # Extract caption patterns
    print("\nExtracting caption patterns...")
    caption_patterns = extract_caption_patterns(metadata_file)
    
    if caption_patterns:
        print(f"✓ Analyzed {caption_patterns['total_captions']} captions")
        print("\nTop Caption Phrases:")
        print("-" * 50)
        for phrase, count in caption_patterns['top_bigrams'][:10]:
            print(f"  '{phrase}': {count} times")
    
    # Save learned knowledge
    knowledge = {
        "visual_features": {
            "feature_names": feature_names,
            "statistics": feature_stats,
            "num_samples": len(all_features)
        },
        "caption_patterns": caption_patterns
    }
    
    output_file = os.path.join("data", "learned_circuit_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(knowledge, f, indent=2)
    
    print(f"\n✓ Saved learned patterns to: {output_file}")
    
    return knowledge

if __name__ == "__main__":
    learn_from_valid_circuits()
