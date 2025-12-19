const videoEl = document.getElementById("webcam");
const overlayCanvasEl = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvasEl.getContext("2d");

const statusTextEl = document.getElementById("status-text");
const currentLabelEl = document.getElementById("current-label");
const currentLabelNameEl = document.getElementById("current-label-name");

const countEls = {
  pointing: document.getElementById("count-pointing"),
  pinch: document.getElementById("count-pinch"),
  open_palm: document.getElementById("count-open_palm"),
  idle: document.getElementById("count-idle"),
  total: document.getElementById("count-total"),
};

const errorOverlayEl = document.getElementById("error-overlay");

// Data collection state
const collectedData = [];
let isRecording = true;
let currentLabel = null;
let autoSampleInterval = null;
const SAMPLE_RATE = 10; // samples per second (10 Hz)

// Gesture label mapping
const LABEL_MAP = {
  1: "pointing",
  2: "pinch",
  3: "open_palm",
  4: "idle",
};

// MediaPipe Hands state
let hands = null;
let handsResults = null;
let handsProcessing = false;

// Initialize webcam
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
        resizeCanvas();
        window.addEventListener("resize", resizeCanvas);
        resolve();
      };
    });

    initHands();
  } catch (err) {
    console.error("Failed to initialize webcam:", err);
    showError(
      "Could not access the webcam. Please allow camera permissions and refresh the page."
    );
  }
}

function resizeCanvas() {
  // Set canvas to match container size (not video dimensions)
  const container = videoEl.parentElement;
  overlayCanvasEl.width = container.clientWidth;
  overlayCanvasEl.height = container.clientHeight;
}

// Calculate the displayed video rectangle (accounting for object-fit: contain)
function getDisplayedVideoRect() {
  const container = videoEl.parentElement;
  const containerAspect = container.clientWidth / container.clientHeight;
  const videoAspect = videoEl.videoWidth / videoEl.videoHeight;
  
  let displayWidth, displayHeight, offsetX, offsetY;
  
  if (containerAspect > videoAspect) {
    // Container is wider - letterboxing
    displayHeight = container.clientHeight;
    displayWidth = displayHeight * videoAspect;
    offsetX = (container.clientWidth - displayWidth) / 2;
    offsetY = 0;
  } else {
    // Container is taller - pillarboxing
    displayWidth = container.clientWidth;
    displayHeight = displayWidth / videoAspect;
    offsetX = 0;
    offsetY = (container.clientHeight - displayHeight) / 2;
  }
  
  return { displayWidth, displayHeight, offsetX, offsetY };
}

// Initialize MediaPipe Hands
function initHands() {
  hands = new Hands({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    },
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  hands.onResults(onHandsResults);

  frameLoop();
}

// Process hand detection results
function onHandsResults(results) {
  handsResults = results;
  handsProcessing = false;

  // Draw landmarks on overlay canvas
  drawLandmarks(results);

  // Auto-sampling is handled by interval, not here
}

// Draw hand landmarks for visualization
function drawLandmarks(results) {
  overlayCtx.clearRect(0, 0, overlayCanvasEl.width, overlayCanvasEl.height);

  if (results.multiHandLandmarks.length === 0) return;

  const landmarks = results.multiHandLandmarks[0];
  const connections = HAND_CONNECTIONS;
  
  // Get the actual displayed video rectangle
  const videoRect = getDisplayedVideoRect();

  // Draw connections
  overlayCtx.strokeStyle = "#00ff88";
  overlayCtx.lineWidth = 2;
  connections.forEach(([start, end]) => {
    const startPoint = landmarks[start];
    const endPoint = landmarks[end];
    overlayCtx.beginPath();
    overlayCtx.moveTo(
      videoRect.offsetX + startPoint.x * videoRect.displayWidth,
      videoRect.offsetY + startPoint.y * videoRect.displayHeight
    );
    overlayCtx.lineTo(
      videoRect.offsetX + endPoint.x * videoRect.displayWidth,
      videoRect.offsetY + endPoint.y * videoRect.displayHeight
    );
    overlayCtx.stroke();
  });

  // Draw landmarks
  landmarks.forEach((landmark) => {
    overlayCtx.fillStyle = "#00ff88";
    overlayCtx.beginPath();
    overlayCtx.arc(
      videoRect.offsetX + landmark.x * videoRect.displayWidth,
      videoRect.offsetY + landmark.y * videoRect.displayHeight,
      3,
      0,
      2 * Math.PI
    );
    overlayCtx.fill();
  });
}

// Collect a single sample
function collectSample(landmarks, label) {
  // Extract features: normalized coordinates (x, y, z) for all 21 landmarks
  const features = [];
  landmarks.forEach((landmark) => {
    features.push(landmark.x, landmark.y, landmark.z);
  });

  collectedData.push({
    label: label,
    landmarks: landmarks.map((l) => ({ x: l.x, y: l.y, z: l.z })),
    features: features, // 63 features (21 landmarks × 3 coords)
    timestamp: Date.now(),
  });

  updateStats();
  updateStatus(`Collected ${label} sample #${getCountForLabel(label)}`);
}

// Update statistics display
function updateStats() {
  const counts = {
    pointing: 0,
    pinch: 0,
    open_palm: 0,
    idle: 0,
  };

  collectedData.forEach((sample) => {
    counts[sample.label]++;
  });

  countEls.pointing.textContent = counts.pointing;
  countEls.pinch.textContent = counts.pinch;
  countEls.open_palm.textContent = counts.open_palm;
  countEls.idle.textContent = counts.idle;
  countEls.total.textContent = collectedData.length;
}

function getCountForLabel(label) {
  return collectedData.filter((s) => s.label === label).length;
}

// Start auto-sampling for a label
function startAutoSampling(label) {
  stopAutoSampling(); // Stop any existing sampling
  
  currentLabel = label;
  currentLabelNameEl.textContent = label;
  currentLabelEl.classList.remove("hidden");
  updateStatus(`Auto-collecting ${label} - Hold gesture steady!`);
  
  // Auto-sample at SAMPLE_RATE Hz
  autoSampleInterval = setInterval(() => {
    if (handsResults && handsResults.multiHandLandmarks.length > 0 && isRecording) {
      collectSample(handsResults.multiHandLandmarks[0], label);
    }
  }, 1000 / SAMPLE_RATE);
}

// Stop auto-sampling
function stopAutoSampling() {
  if (autoSampleInterval) {
    clearInterval(autoSampleInterval);
    autoSampleInterval = null;
  }
  currentLabel = null;
  currentLabelEl.classList.add("hidden");
}

// Keyboard controls
document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    isRecording = !isRecording;
    updateStatus(isRecording ? "Recording..." : "Paused");
    if (!isRecording) {
      stopAutoSampling();
    }
  } else if (e.key >= "1" && e.key <= "4") {
    e.preventDefault();
    const label = LABEL_MAP[e.key];
    startAutoSampling(label);
  } else if (e.key === "e" || e.key === "E") {
    exportData();
  } else if (e.key === "c" || e.key === "C") {
    if (confirm("Clear all collected data?")) {
      clearData();
    }
  }
});

// Stop sampling when key is released
document.addEventListener("keyup", (e) => {
  if (e.key >= "1" && e.key <= "4") {
    stopAutoSampling();
    updateStatus(`Stopped collecting. Press 1-4 to collect another gesture.`);
  }
});

// Export collected data as JSON
function exportData() {
  if (collectedData.length === 0) {
    alert("No data to export! Collect some samples first.");
    return;
  }

  const dataStr = JSON.stringify(collectedData, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `gesture-data-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(url);

  updateStatus(`Exported ${collectedData.length} samples!`);
}

// Clear all collected data
function clearData() {
  stopAutoSampling();
  collectedData.length = 0;
  updateStats();
  updateStatus("Data cleared. Ready to collect.");
}

function updateStatus(message) {
  statusTextEl.textContent = message;
}

function showError(message) {
  errorOverlayEl.textContent = "";
  const p = document.createElement("p");
  p.textContent = message;
  errorOverlayEl.appendChild(p);
  errorOverlayEl.hidden = false;
}

// Main frame loop
function frameLoop() {
  if (!handsProcessing && videoEl.readyState === videoEl.HAVE_ENOUGH_DATA) {
    handsProcessing = true;
    hands.send({ image: videoEl });
  }
  // Redraw landmarks on each frame (in case window resized)
  if (handsResults) {
    drawLandmarks(handsResults);
  }
  requestAnimationFrame(frameLoop);
}

// MediaPipe Hand Connections (for visualization)
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
  [17, 5],
];

// Initialize on load
window.addEventListener("load", () => {
  initWebcam();
});

