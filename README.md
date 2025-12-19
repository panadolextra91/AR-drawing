# AR Hand Drawing

A web-based augmented reality drawing application that uses **machine learning** and hand gesture recognition to draw directly on your webcam feed. Control drawing with simple hand gestures - no mouse or touchscreen needed!

## 🎨 Features

- **Real-time hand tracking** using MediaPipe Hands
- **ML-powered gesture recognition** - trained neural network for accurate gesture classification
- **Gesture-based drawing** - draw by pointing with your index finger
- **Gesture-based erasing** - erase by opening your palm (all 5 fingers up)
- **Pen up/down toggle** - pinch gesture to start/stop drawing for multi-line artwork
- **Smooth drawing** - vector-based detection and trajectory smoothing for stable strokes
- **Fullscreen webcam view** - minimal UI, just you and your canvas

## 🛠️ Technologies Used

- **HTML5** - Structure and layout
- **CSS3** - Styling and fullscreen layout
- **Vanilla JavaScript** - Core application logic
- **MediaPipe Hands** - Real-time hand landmark detection (loaded via CDN)
- **TensorFlow.js** - Machine learning model for gesture classification
- **WebRTC (getUserMedia API)** - Webcam access
- **Canvas API** - Drawing overlay on video feed
- **Python/TensorFlow** - Model training (for developers)

## 📋 Prerequisites

- A modern web browser (Chrome, Edge, or Firefox recommended)
- A webcam connected to your computer
- A local web server (for running locally)

## 🚀 Quick Start

### For End Users (Using Pre-trained Model)

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AR-drawing.git
   cd AR-drawing
   ```

2. **Download the trained model** (if not included):
   - The model files should be in `model/tfjs_model/` directory
   - If missing, you'll need to train the model (see "For Developers" section below)

3. **Start a local web server**:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Or Python 2
   python -m SimpleHTTPServer 8000
   
   # Or Node.js
   npx http-server -p 8000
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

5. **Allow camera permissions** when prompted

6. **Start drawing!** See "Gesture Controls" section below.

### For Developers (Training Your Own Model)

If you want to train your own gesture recognition model:

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect training data**:
   - Open `data-collector.html` in your browser
   - Follow the instructions to collect gesture samples
   - Export the data as JSON

3. **Train the model**:
   ```bash
   python train_model.py --data your-data.json --output model --epochs 100
   ```

4. **Convert to TensorFlow.js format**:
   ```bash
   python convert_model_final.py
   ```

5. **Test the app** - the model will be automatically loaded!

## 🎮 Gesture Controls

### Drawing Mode
- **Pointing gesture**: Raise only your **index finger** (other fingers down)
- **Pinch to toggle pen**: Touch your thumb and index finger tips together briefly
  - First pinch: **Pen down** - start drawing
  - Second pinch: **Pen up** - stop drawing (move hand without drawing)
  - Third pinch: **Pen down** again - resume drawing
- While in pointing gesture with pen down, move your hand to draw white strokes

### Erasing Mode
- **Open palm**: Raise **all 5 fingers** (thumb, index, middle, ring, pinky)
- Move your hand to erase existing drawings with a large eraser brush

### Idle Mode
- Any other hand gesture or no hand detected
- No drawing or erasing occurs

## 📁 Project Structure

```
AR-drawing/
├── index.html              # Main application HTML
├── style.css               # Styling and layout
├── main.js                 # Core application logic with ML model integration
├── data-collector.html     # Tool for collecting training data
├── data-collector.js       # Data collection logic
├── data-collector.css      # Data collector styling
├── train_model.py          # Python script to train the gesture model
├── convert_model_final.py  # Convert Keras model to TensorFlow.js format
├── process_dataset.py      # Process existing datasets (e.g., HaGRID)
├── requirements.txt        # Python dependencies
├── model/                  # Trained model files (gitignored)
│   ├── keras_model.keras   # Keras model (for training)
│   ├── label_mapping.json  # Gesture label mapping
│   └── tfjs_model/        # TensorFlow.js model files
│       ├── model.json      # Model architecture
│       └── *.bin           # Model weights
└── README.md               # This file
```

## 🔧 How It Works

1. **Webcam Access**: The app requests camera permissions and displays the live video feed
2. **Hand Detection**: MediaPipe Hands processes each video frame to detect hand landmarks (21 points per hand)
3. **Feature Extraction**: Extracts 63 features (21 landmarks × 3 coordinates) from hand landmarks
4. **ML Gesture Classification**: 
   - Trained neural network (4-layer dense network) classifies gestures
   - Outputs probabilities for: pointing, pinch, open_palm, idle
   - Falls back to rule-based detection if model fails to load
5. **Drawing Engine**: When drawing mode is active, the index fingertip coordinates are:
   - Mapped from normalized MediaPipe coordinates to canvas pixels
   - Smoothed using a low-pass filter to reduce jitter
   - Used to draw continuous white strokes on a transparent canvas overlay
6. **Canvas Overlay**: The drawing canvas is positioned absolutely over the video, creating the AR effect

## 🌐 Deploy to GitHub Pages

1. **Create a new repository** on GitHub

2. **Push your code** (excluding model files - they're gitignored):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/AR-drawing.git
   git push -u origin main
   ```

3. **Go to your repository Settings → Pages**

4. **Under Source**, select:
   - Branch: `main`
   - Folder: `/ (root)`

5. **Click Save**

6. **Your site will be available at**:
   ```
   https://YOUR_USERNAME.github.io/AR-drawing/
   ```

**Note**: Model files are gitignored. You'll need to either:
- Include them manually for deployment, or
- Train the model and convert it after deployment

## 📝 Notes

- The app works best in well-lit environments
- Keep your hand clearly visible to the camera for best tracking
- The video feed is mirrored horizontally for more natural interaction
- Drawing strokes are white with a 10px brush width
- Eraser brush is 40px wide for easy erasing
- The ML model requires ~60% confidence threshold (adjustable in `main.js`)

## 🐛 Troubleshooting

- **Camera not working**: Make sure you've granted camera permissions in your browser
- **No hand detection**: Ensure your hand is well-lit and clearly visible to the camera
- **Drawing is jittery**: Try to keep your hand steady, or adjust the smoothing factor in `main.js`
- **Gestures not recognized**: Make sure your gestures are clear and held steady for a moment
- **Model not loading**: Check browser console for errors. The app will fall back to rule-based detection
- **"ML model not available" warning**: This is normal if model files are missing. The app will use rule-based detection

## 🎓 Training Your Own Model

### Data Collection

#### Option 1: Use Built-in Data Collector

1. Open `data-collector.html` in your browser
2. Press and hold number keys to collect samples:
   - `1` (hold) - Auto-collect Pointing gesture
   - `2` (hold) - Auto-collect Pinch gesture
   - `3` (hold) - Auto-collect Open Palm
   - `4` (hold) - Auto-collect Idle gesture
3. Hold each gesture for 3-5 seconds to collect ~30-50 samples automatically
4. Repeat 2-3 times per gesture with slight variations
5. Press `E` to export data as JSON
6. Aim for ~50-100 samples per gesture (200-400 total)

**Tips for better data:**
- Vary hand distance from camera (closer/farther)
- Try different angles (slightly left/right)
- Collect in different lighting conditions
- Keep gestures clear and steady while holding the key

#### Option 2: Process Existing Datasets

You can process existing hand gesture datasets (like HaGRID) through MediaPipe Hands:

**HaGRID Dataset**

HaGRID is a large hand gesture dataset with 554K images and 18 gesture classes.

1. **Download HaGRID**:
   ```bash
   # Clone the repository (contains download scripts)
   git clone https://github.com/hukenovs/hagrid.git
   cd hagrid
   # Follow their download instructions
   ```
   Or visit: https://github.com/hukenovs/hagrid

2. **Process the dataset**:
   ```bash
   python process_dataset.py --dataset /path/to/hagrid --output gesture-data-haGRID.json --max-samples 500
   ```

   **Parameters:**
   - `--dataset`: Path to HaGRID root directory
   - `--output`: Output JSON file name
   - `--max-samples`: Max samples per class (default: 500)

   **What it does:**
   - Maps HaGRID gestures to our 4 classes:
     - **pointing**: `point_up`, `one`
     - **pinch**: `ok`, `two`
     - **open_palm**: `open_palm`, `five`, `stop`
     - **idle**: All others (`fist`, `three`, `four`, etc.)
   - Processes each image through MediaPipe Hands
   - Extracts 21 landmarks (63 features) per hand
   - Saves to JSON in the same format as the data collector tool

**Custom Dataset Processing**

If you have your own images organized by gesture:

```bash
# Process pointing gesture images
python process_dataset.py --custom /path/to/pointing_images --label pointing --output pointing-data.json

# Process pinch gesture images
python process_dataset.py --custom /path/to/pinch_images --label pinch --output pinch-data.json
```

**Combining Multiple Datasets**

You can combine data from multiple sources:

```python
import json

# Load multiple JSON files
with open('gesture-data-haGRID.json') as f:
    data1 = json.load(f)

with open('gesture-data-collected.json') as f:  # From web tool
    data2 = json.load(f)

# Combine
combined = data1 + data2

# Save
with open('gesture-data-combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
```

### Training

Once you have your training data (JSON file):

```bash
python train_model.py --data your-data.json --output model --epochs 100
```

**Parameters:**
- `--data`: Path to JSON file with gesture data
- `--output`: Output directory for trained model (default: `model`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)

The script will:
- Load and validate your data
- Split into training/test sets (80/20)
- Train a neural network (4-layer dense network)
- Save the Keras model and label mapping
- Display training accuracy and test accuracy

### Conversion to TensorFlow.js

After training, convert the model to TensorFlow.js format:

```bash
python convert_model_final.py
```

This will:
- Load the trained Keras model
- Convert to TensorFlow.js Layers format
- Save to `model/tfjs_model/` directory
- Create `model.json` and weight files (`.bin`)

The converted model will be automatically loaded by the app when you refresh the browser.

### Dataset Processing Tips

1. **Use GPU** (if available): MediaPipe can use GPU acceleration
2. **Process in batches**: For very large datasets, modify the script to process in chunks
3. **Filter low-confidence detections**: Only keep samples where MediaPipe confidence is high
4. **Balance classes**: Use `--max-samples` to ensure balanced dataset

### Troubleshooting Dataset Processing

- **"No hand detected in many images"**: Normal for some datasets - MediaPipe filters out low-confidence detections. You'll still get plenty of samples.
- **"Out of memory"**: Process smaller batches or reduce `--max-samples` per class
- **"Dataset path not found"**: Check the path is correct. HaGRID structure should be: `dataset_root/gesture_name/images/`

## 📄 License

This project is open source and available for personal and educational use.

## 🙏 Acknowledgments

- MediaPipe Hands for hand tracking
- TensorFlow.js for browser-based ML
- HaGRID dataset (optional, for training)
