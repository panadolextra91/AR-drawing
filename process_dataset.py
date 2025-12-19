"""
Process HaGRID dataset through MediaPipe Hands to extract landmarks.
Maps HaGRID gestures to our 4 classes: pointing, pinch, open_palm, idle
"""

import os
import json
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from pathlib import Path

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture mapping from HaGRID to our classes
# HaGRID has 18 gestures - we'll map relevant ones to our 4 classes
GESTURE_MAPPING = {
    # Pointing gestures
    "point_up": "pointing",
    "one": "pointing",  # Number 1 gesture (index finger up)
    
    # Pinch gestures
    "ok": "pinch",  # OK sign (thumb + index)
    "two": "pinch",  # Sometimes used as pinch
    
    # Open palm gestures
    "open_palm": "open_palm",
    "five": "open_palm",  # All fingers up
    "stop": "open_palm",  # Open palm stop gesture
    
    # Idle/other gestures
    "fist": "idle",
    "three": "idle",
    "four": "idle",
    "call": "idle",
    "mute": "idle",
    "peace": "idle",
    "like": "idle",
    "dislike": "idle",
    "rock": "idle",
    "stop_inverted": "idle",
    "no_gesture": "idle",
}

def extract_landmarks(image_path, hands_model):
    """Extract MediaPipe hand landmarks from an image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands_model.process(image_rgb)
    
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        # Use first detected hand
        landmarks = results.multi_hand_landmarks[0]
        
        # Extract normalized coordinates
        landmark_data = []
        features = []
        for landmark in landmarks.landmark:
            landmark_data.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })
            features.extend([landmark.x, landmark.y, landmark.z])
        
        return {
            "landmarks": landmark_data,
            "features": features  # 63 features (21 landmarks × 3 coords)
        }
    
    return None

def process_haGRID_dataset(dataset_root, output_file="gesture-data-haGRID.json", max_samples_per_class=500):
    """
    Process HaGRID dataset.
    
    Args:
        dataset_root: Path to HaGRID dataset root directory
        output_file: Output JSON file path
        max_samples_per_class: Maximum samples to extract per gesture class
    """
    dataset_path = Path(dataset_root)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_root}' does not exist!")
        print("\nPlease download HaGRID dataset from:")
        print("https://github.com/hukenovs/hagrid")
        return
    
    # Initialize MediaPipe Hands
    print("Initializing MediaPipe Hands...")
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    collected_data = []
    class_counts = {"pointing": 0, "pinch": 0, "open_palm": 0, "idle": 0}
    
    # HaGRID structure: dataset_root/gesture_name/images/
    gesture_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(gesture_dirs)} gesture directories")
    print("Processing images...\n")
    
    for gesture_dir in tqdm(gesture_dirs, desc="Processing gestures"):
        gesture_name = gesture_dir.name
        
        # Map HaGRID gesture to our class
        our_class = GESTURE_MAPPING.get(gesture_name, "idle")
        
        # Skip if we already have enough samples for this class
        if class_counts[our_class] >= max_samples_per_class:
            continue
        
        # Find images in this gesture directory
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(gesture_dir.glob(f"**/*{ext}")))
            image_files.extend(list(gesture_dir.glob(f"**/*{ext.upper()}")))
        
        # Process images
        for img_path in tqdm(image_files, desc=f"  {gesture_name}", leave=False):
            if class_counts[our_class] >= max_samples_per_class:
                break
            
            landmark_data = extract_landmarks(img_path, hands)
            
            if landmark_data:
                collected_data.append({
                    "label": our_class,
                    "landmarks": landmark_data["landmarks"],
                    "features": landmark_data["features"],
                    "source": "haGRID",
                    "original_gesture": gesture_name,
                    "image_path": str(img_path)
                })
                class_counts[our_class] += 1
    
    # Close MediaPipe
    hands.close()
    
    # Save results
    print(f"\n\nCollected {len(collected_data)} samples:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    with open(output_file, 'w') as f:
        json.dump(collected_data, f, indent=2)
    
    print(f"\n✅ Saved to {output_file}")
    print(f"\nYou can now use this file to train your model!")

def process_custom_dataset(image_dir, label, output_file="gesture-data-custom.json"):
    """
    Process a custom directory of images with a single label.
    Useful for processing your own collected images.
    
    Args:
        image_dir: Directory containing images
        label: Label for all images (e.g., "pointing", "pinch", etc.)
        output_file: Output JSON file path
    """
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"Error: Directory '{image_dir}' does not exist!")
        return
    
    # Initialize MediaPipe Hands
    print("Initializing MediaPipe Hands...")
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    collected_data = []
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_path.glob(f"*{ext}")))
        image_files.extend(list(image_path.glob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images")
    print("Processing...\n")
    
    for img_path in tqdm(image_files):
        landmark_data = extract_landmarks(img_path, hands)
        
        if landmark_data:
            collected_data.append({
                "label": label,
                "landmarks": landmark_data["landmarks"],
                "features": landmark_data["features"],
                "source": "custom",
                "image_path": str(img_path)
            })
    
    hands.close()
    
    print(f"\n✅ Collected {len(collected_data)} samples")
    
    with open(output_file, 'w') as f:
        json.dump(collected_data, f, indent=2)
    
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process hand gesture datasets through MediaPipe")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to HaGRID dataset root directory",
        default=None
    )
    parser.add_argument(
        "--custom",
        type=str,
        help="Path to custom image directory (use with --label)",
        default=None
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["pointing", "pinch", "open_palm", "idle"],
        help="Label for custom dataset images",
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gesture-data-processed.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples per class for HaGRID (default: 500)"
    )
    
    args = parser.parse_args()
    
    if args.custom:
        if not args.label:
            print("Error: --label is required when using --custom")
            exit(1)
        process_custom_dataset(args.custom, args.label, args.output)
    elif args.dataset:
        process_haGRID_dataset(args.dataset, args.output, args.max_samples)
    else:
        print("Please specify either --dataset (for HaGRID) or --custom (for custom images)")
        print("\nExamples:")
        print("  # Process HaGRID dataset:")
        print("  python process_dataset.py --dataset /path/to/hagrid --output gesture-data.json")
        print("\n  # Process custom images:")
        print("  python process_dataset.py --custom /path/to/images --label pointing --output pointing-data.json")

