# Classroom Tracking System

An AI-powered application that monitors student movement within designated zones using computer vision. This system helps educators track student presence and movement patterns in classroom settings.

## Features

- **Real-time Person Detection**: Uses YOLO object detection to identify people in video streams
- **Zone Monitoring**: Creates a personalized zone for each detected student
- **Movement Tracking**: Highlights when students move outside their designated zones
- **Multiple Input Sources**: Works with webcams or uploaded video files
- **User-friendly Interface**: Built with Streamlit for easy setup and operation

## How It Works

1. The system detects people using YOLOv8 object detection
2. After 2 seconds of detection, it creates a fixed zone around each person
3. The system tracks if people stay within their zones
4. Visual feedback is provided:
   - Green boxes indicate people staying within their zones
   - Red boxes indicate people who have moved outside their zones

## Installation

```bash
# Clone the repository
git clone https://github.com/amitg404/Classroom_Tracking_System_YoloV8n.git
cd Classroom_Tracking_System_YoloV8n

# Install dependencies
pip install -r requirements.txt
## Usage

```bash
# Run the application
streamlit run app.py
```

Once running, you can:
1. Select between camera input or video file upload
2. Choose a camera if using camera input
3. Upload a video file if using file input
4. Click "Start Tracking" to begin monitoring

## Requirements

The main dependencies are:
- streamlit
- opencv-python
- ultralytics (YOLOv8)
- numpy

See `requirements.txt` for the complete list of dependencies.

## Project Structure

```
classroom-tracking/
│
├── app.py                # Main application file
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── yolov8n.pt            # Pre-trained YOLO model
└── test_video            # Sample video for testing
```

## Future Improvements

- Add attendance tracking functionality
- Implement student identification
- Add statistics and reporting features
- Support for multiple camera views
- Export tracking data for analysis


## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [Streamlit](https://streamlit.io/) for the web application framework
