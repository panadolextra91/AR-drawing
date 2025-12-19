# AR Hand Drawing

A web-based augmented reality drawing application that uses hand gesture recognition to draw directly on your webcam feed. Control drawing with simple hand gestures - no mouse or touchscreen needed!

## 🎨 Features

- **Real-time hand tracking** using MediaPipe Hands
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
- **WebRTC (getUserMedia API)** - Webcam access
- **Canvas API** - Drawing overlay on video feed

## 📋 Prerequisites

- A modern web browser (Chrome, Edge, or Firefox recommended)
- A webcam connected to your computer
- A local web server (for running locally)

## 🚀 How to Run Locally

### Option 1: Python HTTP Server

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd AR-drawing
   ```
3. Start a local web server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Or Python 2
   python -m SimpleHTTPServer 8000
   ```
4. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
5. Allow camera permissions when prompted

### Option 2: Node.js HTTP Server

If you have Node.js installed:

```bash
npx http-server -p 8000
```

Then open `http://localhost:8000` in your browser.

### Option 3: VS Code Live Server

If you use VS Code, install the "Live Server" extension and click "Go Live" in the status bar.

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
├── index.html      # Main HTML structure
├── style.css       # Styling and layout
├── main.js         # Core application logic, gesture detection, and drawing
└── README.md       # This file
```

## 🌐 Deploy to GitHub Pages

1. Create a new repository on GitHub
2. Push your code to the repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/AR-drawing.git
   git push -u origin main
   ```
3. Go to your repository **Settings → Pages**
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/ (root)`
5. Click **Save**
6. Your site will be available at:
   ```
   https://YOUR_USERNAME.github.io/AR-drawing/
   ```

## 🔧 How It Works

1. **Webcam Access**: The app requests camera permissions and displays the live video feed
2. **Hand Detection**: MediaPipe Hands processes each video frame to detect hand landmarks (21 points per hand)
3. **Gesture Recognition**: Vector-based algorithms analyze finger positions to classify gestures:
   - Finger extension vectors determine if fingers are up or down
   - Distance calculations detect pinch gestures
4. **Drawing Engine**: When drawing mode is active, the index fingertip coordinates are:
   - Mapped from normalized MediaPipe coordinates to canvas pixels
   - Smoothed using a low-pass filter to reduce jitter
   - Used to draw continuous white strokes on a transparent canvas overlay
5. **Canvas Overlay**: The drawing canvas is positioned absolutely over the video, creating the AR effect

## 📝 Notes

- The app works best in well-lit environments
- Keep your hand clearly visible to the camera for best tracking
- The video feed is mirrored horizontally for more natural interaction
- Drawing strokes are white with a 10px brush width
- Eraser brush is 40px wide for easy erasing

## 🐛 Troubleshooting

- **Camera not working**: Make sure you've granted camera permissions in your browser
- **No hand detection**: Ensure your hand is well-lit and clearly visible to the camera
- **Drawing is jittery**: Try to keep your hand steady, or adjust the smoothing factor in `main.js`
- **Gestures not recognized**: Make sure your gestures are clear and held steady for a moment

## 📄 License

This project is open source and available for personal and educational use.

