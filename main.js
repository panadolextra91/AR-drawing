const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("draw-canvas");
const modeLabelEl = document.getElementById("mode-label");
const fpsLabelEl = document.getElementById("fps-label");
const errorOverlayEl = document.getElementById("error-overlay");

const canvasCtx = canvasEl.getContext("2d");

// Application state
const appState = {
  mode: "idle", // 'idle' | 'drawing' | 'erasing'
  lastPoint: null,
  lastFrameTime: performance.now(),
  isDrawing: false, // pen down/up toggle controlled by gesture
  stableGestureName: null,
  stableGestureFrames: 0,
  smoothedPoint: null,
};

// MediaPipe Hands state
let hands = null;
let handsResults = null;
let handsProcessing = false;

// TensorFlow.js model state
let gestureModel = null;
let labelMapping = null;
let modelLoaded = false;

async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    videoEl.srcObject = stream;

    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    resizeCanvasToVideo();
    window.addEventListener("resize", resizeCanvasToVideo);

    initHands();
  } catch (err) {
    console.error("Failed to initialize webcam:", err);
    showError(
      "Could not access the webcam. Please allow camera permissions and refresh the page."
    );
  }
}

async function loadGestureModel() {
  try {
    // Load label mapping
    const mappingResponse = await fetch("./model/label_mapping.json");
    labelMapping = await mappingResponse.json();
    console.log("Label mapping loaded:", labelMapping);

    // Load TensorFlow.js model (Layers format)
    const modelPath = "./model/tfjs_model/model.json";
    try {
      gestureModel = await tf.loadLayersModel(modelPath);
      console.log(`✅ Loaded ML model successfully from ${modelPath}`);
      modelLoadedSuccessfully = true;
    } catch (e) {
      console.error(`Failed to load model from ${modelPath}:`, e);
      throw e;
    }
    
    if (!modelLoadedSuccessfully) {
      throw new Error("Could not load model from any path");
    }
    
    modelLoaded = true;
    console.log("Gesture model loaded successfully!");
  } catch (err) {
    console.error("Failed to load gesture model:", err);
    console.warn("⚠️  ML model not available - using rule-based gesture detection");
    console.warn("This is normal if the model hasn't been converted yet.");
    console.warn("The app will work fine with rule-based detection!");
    modelLoaded = false;
  }
}

function initHands() {
  if (typeof Hands === "undefined") {
    showError(
      "MediaPipe Hands library failed to load. Check your internet connection and refresh."
    );
    return;
  }

  hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });

  hands.onResults((results) => {
    handsResults = results;
  });

  // Load gesture model
  loadGestureModel();

  requestAnimationFrame(frameLoop);
}

function resizeCanvasToVideo() {
  // Match the canvas internal resolution to the video frame size
  if (videoEl.videoWidth && videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }
}

function updateFps(now) {
  const dt = now - appState.lastFrameTime;
  appState.lastFrameTime = now;
  const fps = 1000 / dt;
  fpsLabelEl.textContent = `FPS: ${fps.toFixed(0)}`;
}

function setMode(mode) {
  if (appState.mode === mode) return;
  appState.mode = mode;
  modeLabelEl.textContent = `Mode: ${mode}`;
}

function showError(message) {
  errorOverlayEl.textContent = message;
  errorOverlayEl.hidden = false;
}

function fingerIsUp(landmarks, tipIndex, pipIndex) {
  // y is inverted in image coordinates (top = 0), so "up" means smaller y
  const tip = landmarks[tipIndex];
  const pip = landmarks[pipIndex];
  return tip.y < pip.y - 0.02; // small margin to reduce noise
}

function fingerExtension(landmarks, tipIndex, mcpIndex) {
  const tip = landmarks[tipIndex];
  const mcp = landmarks[mcpIndex];
  const dx = tip.x - mcp.x;
  const dy = tip.y - mcp.y;
  const dz = (tip.z || 0) - (mcp.z || 0);
  return Math.hypot(dx, dy, dz);
}

function distance2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

// Extract features from landmarks (63 features: 21 landmarks × 3 coords)
function extractFeatures(landmarks) {
  const features = [];
  for (const landmark of landmarks) {
    features.push(landmark.x, landmark.y, landmark.z || 0);
  }
  return features;
}

// Classify gesture using trained model or fallback to rule-based
async function classifyGesture(landmarks) {
  // Use trained model if available
  if (modelLoaded && gestureModel && labelMapping) {
    try {
      // Extract features
      const features = extractFeatures(landmarks);
      
      // Convert to tensor
      const inputTensor = tf.tensor2d([features]);
      
      // Predict
      const prediction = gestureModel.predict(inputTensor);
      const probabilities = await prediction.data();
      
      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();
      
      // Get predicted class index
      const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedIndex];
      
      // Map index to gesture name
      const gestureName = labelMapping[predictedIndex];
      
      // Only use prediction if confidence is high enough
      if (confidence > 0.6) {
        // Map model labels to our gesture names
        if (gestureName === "pointing") return "drawPose";
        if (gestureName === "pinch") return "pinch";
        if (gestureName === "open_palm") return "erasePose";
        if (gestureName === "idle") return "none";
      }
    } catch (err) {
      console.error("Model prediction error:", err);
      // Fall through to rule-based
    }
  }
  
  // Fallback to rule-based detection
  return classifyGestureRuleBased(landmarks);
}

// Rule-based gesture classification (fallback)
function classifyGestureRuleBased(landmarks) {
  // Use both vertical position and extension vector length
  const thumbUp = fingerExtension(landmarks, 4, 2) > 0.14;
  const indexUp =
    fingerIsUp(landmarks, 8, 6) &&
    fingerExtension(landmarks, 8, 5) > 0.18;
  const middleUp =
    fingerIsUp(landmarks, 12, 10) &&
    fingerExtension(landmarks, 12, 9) > 0.18;
  const ringUp =
    fingerIsUp(landmarks, 16, 14) &&
    fingerExtension(landmarks, 16, 13) > 0.18;
  const pinkyUp =
    fingerIsUp(landmarks, 20, 18) &&
    fingerExtension(landmarks, 20, 17) > 0.18;

  const indexDown = !indexUp;
  const middleDown = !middleUp;
  const ringDown = !ringUp;
  const pinkyDown = !pinkyUp;

  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinchDistance = distance2D(thumbTip, indexTip);

  // Pinch gesture: thumb and index close together
  if (pinchDistance < 0.05) {
    return "pinch";
  }

  // Drawing pose: index up, others down
  if (indexUp && middleDown && ringDown && pinkyDown) {
    return "drawPose";
  }

  // Erasing pose: open palm (all five fingers up)
  if (thumbUp && indexUp && middleUp && ringUp && pinkyUp) {
    return "erasePose";
  }

  return "none";
}

async function interpretHandsResults(results) {
  if (
    !results ||
    !results.multiHandLandmarks ||
    results.multiHandLandmarks.length === 0
  ) {
    appState.stableGestureName = null;
    appState.stableGestureFrames = 0;
    return { mode: "idle", point: null };
  }

  const landmarks = results.multiHandLandmarks[0];
  const gestureName = await classifyGesture(landmarks);

  // Update gesture stability (simple hysteresis)
  if (gestureName === appState.stableGestureName) {
    appState.stableGestureFrames += 1;
  } else {
    appState.stableGestureName = gestureName;
    appState.stableGestureFrames = 1;
  }

  // If pinch is stable for a few frames, toggle pen up/down
  if (
    appState.stableGestureName === "pinch" &&
    appState.stableGestureFrames === 5
  ) {
    appState.isDrawing = !appState.isDrawing;
  }

  let mode = "idle";

  if (gestureName === "erasePose") {
    // Erase while fist is held
    mode = "erasing";
  } else if (gestureName === "drawPose" && appState.isDrawing) {
    // Only draw when pen is toggled "down"
    mode = "drawing";
  } else {
    mode = "idle";
  }

  // Use index fingertip (landmark 8) as drawing point
  const tip = landmarks[8];

  // MediaPipe gives normalized coords; map to canvas.
  // The video is visually mirrored via CSS (scaleX(-1)), so mirror x here too.
  const x = canvasEl.width - tip.x * canvasEl.width;
  const y = tip.y * canvasEl.height;

  // Smooth fingertip using a simple low-pass filter on the vector
  let smoothed = appState.smoothedPoint;
  const alpha = 0.35; // higher = more responsive, lower = smoother
  if (!smoothed) {
    smoothed = { x, y };
  } else {
    smoothed = {
      x: smoothed.x + alpha * (x - smoothed.x),
      y: smoothed.y + alpha * (y - smoothed.y),
    };
  }
  appState.smoothedPoint = smoothed;

  const point =
    mode === "drawing" || mode === "erasing" ? smoothed : null;

  return { mode, point };
}

function drawWithGesture(point) {
  if (!point) {
    appState.lastPoint = null;
    return;
  }

  const { x, y } = point;

  if (!appState.lastPoint) {
    appState.lastPoint = { x, y };
    return;
  }

  canvasCtx.save();
  if (appState.mode === "erasing") {
    canvasCtx.globalCompositeOperation = "destination-out";
    canvasCtx.lineWidth = 40;
  } else {
    canvasCtx.globalCompositeOperation = "source-over";
    canvasCtx.strokeStyle = "#ffffff";
    canvasCtx.lineWidth = 10;
  }

  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";

  canvasCtx.beginPath();
  canvasCtx.moveTo(appState.lastPoint.x, appState.lastPoint.y);
  canvasCtx.lineTo(x, y);
  canvasCtx.stroke();
  canvasCtx.restore();

  appState.lastPoint = { x, y };
}

async function frameLoop(now) {
  updateFps(now);

  // Send current frame to MediaPipe Hands (avoid overlapping calls).
  if (hands && !handsProcessing) {
    handsProcessing = true;
    await hands.send({ image: videoEl });
    handsProcessing = false;
  }

  const { mode, point } = await interpretHandsResults(handsResults);
  setMode(mode);

  if (mode === "drawing" || mode === "erasing") {
    drawWithGesture(point);
  } else {
    appState.lastPoint = null;
  }

  requestAnimationFrame(frameLoop);
}

// Entry
window.addEventListener("load", () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("Your browser does not support webcam access (getUserMedia).");
    return;
  }

  initWebcam();
});


