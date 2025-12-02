# Eye Tracking Project

A real-time eye tracking application using OpenCV and MediaPipe to detect face landmarks, calculate Eye Aspect Ratio (EAR), and track blink patterns with duration measurement.

## Project Description

This project implements a real-time eye tracking system that:

- Detects facial landmarks using MediaPipe Face Mesh
- Calculates Eye Aspect Ratio (EAR) for both eyes
- Detects and counts blinks in real-time
- Measures blink duration (average, min, max)
- Displays eye meshes and metrics on video feed
- Provides console output with detailed status information

The system uses threshold-based classification where EAR values below 0.20 indicate closed eyes. Temporal smoothing requires 2 consecutive frames below threshold to reduce false positives.

## Installation Instructions

### Prerequisites

- Python 3.8 or higher (Python 3.11-3.12 recommended for MediaPipe compatibility)
- Webcam/camera access
- pip package manager

### Setup Steps

1. **Clone or download this repository**

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install opencv-python numpy mediapipe
   ```
   
   Or if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

### Basic Usage

Run the eye tracker with default settings:
```bash
python eye_tracker.py
```

### What to Expect

- A window opens showing your webcam feed
- Eye meshes are drawn over your eyes
- Real-time metrics displayed on screen:
  - Frame number
  - Left EAR value (cyan)
  - Right EAR value (magenta)
  - Average EAR (yellow)
  - Blink status (red for blinking, green for open)
  - Total blink count
  - Average blink duration (when available)
  - EAR threshold value

### Controls

- **Press 'q'** - Quit the application

### Console Output

The application provides detailed console output:
- Startup information (camera resolution, FPS, settings)
- Periodic status updates (every 30 frames)
- Blink detection alerts with duration
- Shutdown statistics (total frames, blinks, duration stats)

### Example Console Output

```
============================================================
🎯 Eye Tracker Started Successfully
============================================================
📹 Camera Resolution: 640x480 @ 30 FPS
⚙️  EAR Threshold: 0.2
📊 Blink Frame Threshold: 2 frames
============================================================

[Frame 30] 👁️ OPEN | L_EAR: 0.312 | R_EAR: 0.298 | Avg: 0.305 | Blinks: 0
[Frame 60] 👁️ CLOSED | L_EAR: 0.185 | R_EAR: 0.172 | Avg: 0.179 | Blinks: 1 | Duration: 45.2ms
[Frame 90] 👁️ OPEN | L_EAR: 0.308 | R_EAR: 0.301 | Avg: 0.305 | Blinks: 1 | Avg Duration: 45.2ms
```

## Known Limitations

1. **MediaPipe Compatibility**: MediaPipe may not be available for Python 3.13+. Use Python 3.11 or 3.12 for best compatibility.

2. **Single Face Detection**: The system is configured to detect only one face at a time (`max_num_faces=1`).

3. **False Positives**: When looking up or when eyes are partially occluded (e.g., hand covering eyes), the system may detect false blinks or show "Face Not Fully Visible" messages. This is mitigated with temporal smoothing but may still occur.

4. **Lighting Sensitivity**: Performance depends on good lighting conditions. Poor lighting may affect face detection accuracy.

5. **Camera Requirements**: Requires a working webcam/camera. The system defaults to camera index 0.

6. **Real-time Performance**: Processing speed depends on your hardware. Lower-end systems may experience reduced frame rates.

7. **EAR Threshold**: The default threshold (0.20) may need adjustment for different users or lighting conditions. This is not currently configurable without modifying the code.

8. **No Calibration**: The system doesn't perform user-specific calibration. EAR thresholds are fixed.

## Code Documentation

### Main Components

**`EyeTracker` class** (`eye_tracker.py`):
- `__init__()`: Initializes MediaPipe Face Mesh, sets thresholds and tracking variables
- `calculate_ear()`: Computes Eye Aspect Ratio from 6 eye landmark points
- `get_eye_landmarks()`: Extracts specific eye landmarks from MediaPipe results
- `process_frame()`: Main processing function that detects face, calculates EAR, and detects blinks
- `run()`: Main loop that captures frames, processes them, and displays results

### Key Features

**Eye Aspect Ratio (EAR) Calculation**:
- Formula: `EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)`
- Uses 6 landmark points per eye (outer, top, top-mid, inner, bottom-mid, bottom)
- EAR decreases as eyes close (approaches 0)

**Blink Detection**:
- Temporal smoothing: Requires 2 consecutive frames below threshold
- State tracking: Only counts transitions from open → closed (avoids duplicate counting)
- Duration measurement: Tracks start/end times with millisecond precision

**Visual Feedback**:
- Eye meshes drawn using MediaPipe's `FACEMESH_LEFT_EYE` and `FACEMESH_RIGHT_EYE`
- Color-coded metrics (cyan, magenta, yellow, red, green)
- Real-time blink duration display

### Test Files

The `tests/` folder contains test scripts for different phases:
- `test_phase1_setup.py`: Environment setup verification
- `test_phase2_face_detection.py`: Face detection testing
- `test_phase3_eye_landmark_extraction.py`: Eye landmark extraction
- `test_phase4_ear_calculation.py`: EAR calculation verification
- `test_phase5_eye_state_classification.py`: Eye state classification
- `test_phase6_polish_edge_cases.py`: Edge case handling
