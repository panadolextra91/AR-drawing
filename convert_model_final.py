"""
Convert Keras model to TensorFlow.js format.
Uses TensorFlow's internal conversion utilities.
"""

import tensorflow as tf
import os
import json
import numpy as np

def convert_model():
    """Convert Keras model to TensorFlow.js Layers format."""
    
    model_path = "model/keras_model.keras"
    output_dir = "model/tfjs_model"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return False
    
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Use model.to_json() for proper Keras JSON format
    print("Extracting model architecture...")
    model_json_str = model.to_json()
    model_topology_raw = json.loads(model_json_str)
    
    # Clean up topology - remove fields that might cause issues
    model_topology = {
        "class_name": model_topology_raw.get("class_name", "Sequential"),
        "config": model_topology_raw.get("config", {}),
        "keras_version": model_topology_raw.get("keras_version", tf.keras.__version__),
        "backend": "tensorflow"
    }
    
    # Fix InputLayer configuration - TensorFlow.js needs inputShape instead of batch_shape
    if "config" in model_topology and "layers" in model_topology["config"]:
        for layer in model_topology["config"]["layers"]:
            if layer.get("class_name") == "InputLayer":
                layer_config = layer.get("config", {})
                # Convert batch_shape to inputShape
                if "batch_shape" in layer_config:
                    batch_shape = layer_config["batch_shape"]
                    if batch_shape and len(batch_shape) > 1:
                        # Remove batch dimension, keep the rest as inputShape
                        layer_config["inputShape"] = list(batch_shape[1:])
                    del layer_config["batch_shape"]
    
    # Remove build_config and compile_config if present (they can cause issues)
    if "build_config" in model_topology:
        del model_topology["build_config"]
    if "compile_config" in model_topology:
        del model_topology["compile_config"]
    
    # Get all weights
    all_weights = model.get_weights()
    print(f"Found {len(all_weights)} weight arrays")
    
    # Create output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create weights manifest - TensorFlow.js format
    # TensorFlow.js expects weightsManifest to be an array with "paths" and "weights" fields
    weight_paths = []
    weight_info = []
    
    weight_index = 0
    layer_weight_index = 0
    
    # Map weights to layer names
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            for i, weight in enumerate(layer_weights):
                weight_shape = list(weight.shape)
                weight_name = f"group{weight_index}-shard1of1.bin"
                weight_paths.append(weight_name)
                
                # Create proper weight name based on layer
                if i == 0:
                    weight_name_tfjs = f"{layer.name}/kernel"
                else:
                    weight_name_tfjs = f"{layer.name}/bias"
                
                weight_info.append({
                    "name": weight_name_tfjs,
                    "shape": weight_shape,
                    "dtype": "float32"
                })
                weight_index += 1
    
    # Group weights into manifest entries (TensorFlow.js format)
    weights_manifest = [{
        "paths": weight_paths,
        "weights": weight_info
    }]
    
    # Create model.json in TensorFlow.js Layers format
    model_json = {
        "modelTopology": model_topology,
        "weightsManifest": weights_manifest,
        "format": "layers-model",
        "generatedBy": "tensorflowjs",
        "convertedBy": "manual-conversion"
    }
    
    # Save model.json
    model_json_path = os.path.join(output_dir, "model.json")
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f, indent=2)
    print(f"✅ Created {model_json_path}")
    
    # Save weights as binary files (little-endian float32)
    print("Saving weights...")
    weight_index = 0
    for weight in all_weights:
        weight_name = f"group{weight_index}-shard1of1.bin"
        weight_path = os.path.join(output_dir, weight_name)
        
        # Convert to float32 and save as binary (little-endian)
        weight_flat = weight.flatten().astype(np.float32)
        # Ensure little-endian byte order
        weight_flat.byteswap(False).tofile(weight_path)
        
        weight_index += 1
    
    print(f"✅ Saved {weight_index} weight files")
    print(f"\n✅ Conversion complete! Model saved to {output_dir}/")
    
    return True

if __name__ == "__main__":
    success = convert_model()
    if success:
        print("\n🎉 Model conversion successful!")
        print("Refresh your browser to load the model!")
    else:
        print("\n❌ Model conversion failed!")

