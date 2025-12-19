"""
Train a TensorFlow/Keras model for hand gesture classification.
Converts MediaPipe hand landmarks to gesture classes: pointing, pinch, open_palm, idle
"""

import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(json_file):
    """Load gesture data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Extract features and labels
    X = []
    y = []
    
    for sample in data:
        # Use the features array (63 features: 21 landmarks × 3 coords)
        if 'features' in sample:
            X.append(sample['features'])
        else:
            # Fallback: extract from landmarks
            features = []
            for landmark in sample['landmarks']:
                features.extend([landmark['x'], landmark['y'], landmark['z']])
            X.append(features)
        
        y.append(sample['label'])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
    
    return X, y

def create_model(input_dim, num_classes):
    """Create a simple neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(json_file, output_dir='model', epochs=100, batch_size=32):
    """
    Train the gesture classification model.
    
    Args:
        json_file: Path to JSON file with gesture data
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Load data
    X, y = load_data(json_file)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create model
    model = create_model(X.shape[1], len(label_encoder.classes_))
    
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Keras model (for Python use)
    keras_model_path = os.path.join(output_dir, 'keras_model.keras')
    model.save(keras_model_path)
    print(f"\n✅ Saved Keras model to {keras_model_path}")
    
    # Save label encoder mapping
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"✅ Saved label mapping to {output_dir}/label_mapping.json")
    
    # Convert to TensorFlow.js format
    try:
        import tensorflowjs as tfjs
        tfjs_dir = os.path.join(output_dir, 'tfjs_model')
        tfjs.converters.save_keras_model(model, tfjs_dir)
        print(f"✅ Saved TensorFlow.js model to {tfjs_dir}/")
        print(f"\nTo use in browser, copy the '{tfjs_dir}' folder to your web app.")
    except ImportError:
        print("\n⚠️  tensorflowjs not installed. Install it to export TensorFlow.js model:")
        print("   pip install tensorflowjs")
        print("\nOr convert manually later:")
        print("   tensorflowjs_converter --input_format=keras keras_model tfjs_model")
    
    return model, label_encoder, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train gesture classification model")
    parser.add_argument(
        "--data",
        type=str,
        default="gesture-data-1766151397649.json",
        help="Path to JSON file with gesture data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found!")
        exit(1)
    
    train_model(args.data, args.output, args.epochs, args.batch_size)
