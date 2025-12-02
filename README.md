# Eye Tracking Project

A comprehensive real-time eye tracking application using OpenCV and MediaPipe to detect and analyze eye movements, blink patterns, gaze direction, and drowsiness.

## Features

### Level 1: Enhanced Features
- ✅ Real-time eye tracking using webcam or video files
- ✅ Eye Aspect Ratio (EAR) calculation for blink detection
- ✅ **Blink count tracking** over time (left, right, and both eyes)
- ✅ **Blink frequency calculation** (blinks per minute)
- ✅ **Blink duration measurement** (average, min, max)
- ✅ **Independent eye tracking** - detect winking (one eye closed, other open)
- ✅ Face mesh landmark detection using MediaPipe
- ✅ Visual feedback with annotated video frames

### Level 2: Advanced Functionality
- ✅ **CSV export** - Save session data (timestamp, EAR, state, gaze direction)
- ✅ **Configurable threshold** - Adjust EAR threshold via keyboard (+/- keys)
- ✅ **Temporal smoothing** - Reduce jitter with moving average filter
- ✅ **Video file input** - Process pre-recorded videos in addition to webcam

### Level 3: Research Extensions
- ✅ **Drowsiness detection** - Alert when eyes remain closed for extended periods
- ✅ **Gaze direction estimation** - Detect horizontal (left/center/right) and vertical (up/center/down) gaze
- ✅ **Algorithm comparison** - Compare EAR vs. Pupil Distance Ratio methods
- ✅ **Technical report generation** - Automated analysis and statistics report

## Phase 1: Environment Setup

### Prerequisites

- Python 3.8 or higher
- Webcam/camera access

### Installation Steps

1. **Set up Python virtual environment**

   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `opencv-python`: Computer vision library for webcam access and image processing
   - `mediapipe`: Google's framework for face mesh detection
   - `numpy`: Numerical computing library
   - `scipy`: Scientific computing library
   - `pandas`: Data analysis library for CSV export

3. **Run Phase 1 tests**

   Run the Phase 1 test script to verify everything is set up correctly:
   ```bash
   python tests/test_phase1_setup.py
   ```

   This will:
   - Verify MediaPipe installation and display version
   - Test webcam access and frame capture
   - Optionally display webcam feed (press 'q' to quit)

### Deliverable

A working webcam display using OpenCV that confirms:
- Virtual environment is set up correctly
- All dependencies are installed
- Webcam can be accessed and frames can be captured
- MediaPipe is properly installed and importable

## Phase 2: Face Detection

### Tasks

1. **Initialize MediaPipe Face Mesh**
   - Set up Face Mesh with appropriate parameters
   - Configure detection and tracking confidence thresholds

2. **Process webcam frames and detect face**
   - Capture frames from webcam
   - Convert BGR to RGB for MediaPipe
   - Process frames through Face Mesh

3. **Extract and visualize all facial landmarks**
   - Extract 468 facial landmarks from MediaPipe
   - Draw face mesh contours and tesselation
   - Visualize landmarks in real-time

4. **Handle "no face detected" gracefully**
   - Check if face is detected before processing
   - Display appropriate message when no face is found
   - Continue processing without errors

### Testing Phase 2

Run the Phase 2 test script:
```bash
python tests/test_phase2_face_detection.py
```

This will:
- Test Face Mesh initialization
- Test face detection on a single frame
- Optionally run real-time face landmark detection

### Deliverable

Real-time face landmark detection that:
- Successfully initializes MediaPipe Face Mesh
- Detects faces in webcam frames
- Visualizes all 468 facial landmarks
- Handles "no face detected" cases gracefully
- Runs smoothly in real-time

## Project Structure

```
eye_tracking_project/
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
├── eye_tracker.py               # Main eye tracking application (enhanced)
└── tests/
    ├── test_phase1_setup.py     # Phase 1 environment setup tests
    └── test_phase2_face_detection.py  # Phase 2 face detection tests
```

## Feature Details

### Level 1: Enhanced Features

**Blink Tracking:**
- Tracks blinks independently for left and right eyes
- Counts total blinks, individual eye blinks, and both-eye blinks
- Calculates real-time blink frequency (blinks per minute)
- Measures blink duration in milliseconds

**Wink Detection:**
- Detects when one eye is closed while the other remains open
- Distinguishes between left winks and right winks
- Useful for gesture recognition applications

### Level 2: Advanced Functionality

**CSV Export:**
- Exports comprehensive session data including:
  - Timestamp for each frame
  - EAR values (left, right, average)
  - Eye states (open/closed)
  - Gaze direction estimates
  - Blink counts and frequency
  - Drowsiness detection status

**Configurable Threshold:**
- Adjust EAR threshold in real-time using keyboard
- Allows fine-tuning for different lighting conditions or users
- Changes take effect immediately

**Temporal Smoothing:**
- Uses moving average filter to reduce jitter
- Configurable window size (default: 5 frames)
- Improves stability of blink detection

**Video File Support:**
- Process pre-recorded video files
- Useful for offline analysis and testing
- Maintains all features (blink tracking, CSV export, etc.)

### Level 3: Research Extensions

**Drowsiness Detection:**
- Monitors extended eye closure periods
- Configurable threshold (default: 2 seconds)
- Alerts when drowsiness is detected
- Tracks drowsiness events in session data

**Gaze Direction Estimation:**
- Estimates horizontal gaze (left/center/right)
- Estimates vertical gaze (up/center/down)
- Based on eye position relative to facial landmarks
- Useful for attention and focus analysis

**Algorithm Comparison:**
- Implements both EAR (Eye Aspect Ratio) and Pupil Distance Ratio methods
- Compares performance in technical report
- Provides insights into algorithm effectiveness

**Technical Report:**
- Automated generation of comprehensive analysis
- Includes statistics, comparisons, and recommendations
- Useful for research and documentation purposes

## Usage

### Running the Eye Tracker

#### Basic Usage (Webcam)
```bash
# Activate virtual environment
source venv/bin/activate

# Run the main eye tracker with default settings
python eye_tracker.py
```

#### Advanced Usage Options
```bash
# Process a video file instead of webcam
python eye_tracker.py --video path/to/video.mp4

# Customize EAR threshold (default: 0.25)
python eye_tracker.py --threshold 0.30

# Adjust temporal smoothing window (default: 5)
python eye_tracker.py --smoothing 10

# Set drowsiness detection threshold in seconds (default: 2.0)
python eye_tracker.py --drowsiness 3.0

# Combine options
python eye_tracker.py --video test.mp4 --threshold 0.28 --smoothing 7
```

### Keyboard Controls

While the eye tracker is running:
- **'q'** - Quit the application
- **'+' or '='** - Increase EAR threshold by 0.01
- **'-'** - Decrease EAR threshold by 0.01
- **'s'** - Save current session data to CSV file
- **'r'** - Generate technical report

### Output Files

The application automatically generates:
1. **CSV Session Data** (`eye_tracking_session_YYYYMMDD_HHMMSS.csv`):
   - Timestamp for each frame
   - Left/Right/Average EAR values
   - Eye states (open/closed)
   - Gaze direction
   - Blink counts and frequency
   - Drowsiness detection status

2. **Technical Report** (`technical_report_YYYYMMDD_HHMMSS.txt`):
   - Session statistics (duration, FPS, frame count)
   - Blink statistics (count, frequency, duration)
   - EAR statistics (mean, min, max, std dev)
   - Algorithm comparison (EAR vs. Pupil Distance Ratio)
   - Drowsiness event analysis
   - Gaze direction distribution
   - Recommendations and findings

Files are saved automatically when you quit the application, or manually when pressing 's' (CSV) or 'r' (report).

### Running Tests

```bash
# Phase 1: Environment setup
python tests/test_phase1_setup.py

# Phase 2: Face detection
python tests/test_phase2_face_detection.py
```

**Note:** MediaPipe may not be available for Python 3.13+. If you encounter import errors, consider using Python 3.11 or 3.12.

## License

(To be added)

