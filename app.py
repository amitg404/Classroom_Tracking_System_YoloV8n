import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import time

class ClassroomTracker:
    def __init__(self, input_source=0):
        """
        Initialize the classroom tracking system
        
        Args:
        - input_source (int/str): Video source (camera index or file path)
        """
        # Load pre-trained YOLO model
        self.model = YOLO('yolov8n.pt')  # Using nano version for performance
        
        # Video capture setup
        self.input_source = input_source
        self.cap = cv2.VideoCapture(self.input_source)
        
        # Track student dynamic zones and movement
        self.student_tracking = {}
    
    def detect_and_track(self, stframe):
        """
        Detect and track people with fixed zones
        
        Args:
        - stframe: Streamlit frame to display video
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                st.warning("Unable to read frame. Video might have ended.")
                break
            
            # Detect and track people using YOLO
            results = self.model.track(frame, persist=True, classes=[0])  # Track only persons
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Check if it's a person (class 0 in COCO dataset)
                    if int(box.cls[0]) == 0:
                        # Get track ID
                        track_id = int(box.id[0]) if box.id is not None else None
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        bbox = (x1, y1, x2, y2)
                        
                        # Manage tracking for this person
                        self.update_person_tracking(frame, bbox, track_id)
            
            # Convert frame from BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame in Streamlit
            stframe.image(frame_rgb, channels="RGB")
            
            # Small delay to control frame rate
            time.sleep(0.03)
    
    def update_person_tracking(self, frame, bbox, track_id):
        """
        Update tracking for a specific person
        """
        x1, y1, x2, y2 = bbox

        # If this person is not yet tracked
        if track_id not in self.student_tracking:
            # Store the time when the person is first detected
            self.student_tracking[track_id] = {
                'current_bbox': bbox,
                'creation_time': time.time(),
                'fixed_zone': None  # Initially, no fixed zone
            }
            # Draw green box for new person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return

        # Retrieve existing tracking info
        student_info = self.student_tracking[track_id]
        current_time = time.time()

        # Check if 2 seconds have passed since the person was first detected
        if student_info['fixed_zone'] is None and (current_time - student_info['creation_time'] >= 2):
            # Create initial Box2 (fixed zone) after 2 seconds
            box2 = self.create_fixed_zone(bbox)
            student_info['fixed_zone'] = box2  # Set the fixed zone
            # Draw Box2 (fixed zone)
            self.draw_zone(frame, box2, (0, 255, 0))

        # Check if Box2 is set
        if student_info['fixed_zone'] is not None:
            box2 = student_info['fixed_zone']
            # Check if current Box1 is within Box2
            is_in_zone = self.is_box_in_zone(bbox, box2)

            # Determine box color and zone drawing
            color = (0, 255, 0) if is_in_zone else (0, 0, 255)

            # Draw Box1 with the determined color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw Box2 (fixed zone) with the same color as Box1's current state
            self.draw_zone(frame, box2, color)

        # Update tracking information
        student_info['current_bbox'] = bbox
    
    def create_fixed_zone(self, bbox):
        x1, y1, x2, y2 = bbox
        # Calculate width and height of the original bounding box
        width = x2 - x1
        height = y2 - y1
        # Calculate margin as 5% of the width and height
        margin_x = int(width * 0.05)  # 5% of width as margin
        margin_y = int(height * 0.05)  # 5% of height as margin
        # Expand the bbox by the margins to create Box2
        new_x1 = x1 - margin_x  # Expand left
        new_y1 = y1 - margin_y  # Expand top
        new_x2 = x2 + margin_x  # Expand right
        new_y2 = y2 + margin_y  # Expand bottom
        return ((new_x1, new_y1), (new_x2, new_y2))

    def is_box_in_zone(self, bbox, zone):
        """
        Check if the bounding box is within a given zone
        """
        (start_x, start_y), (end_x, end_y) = zone
        x1, y1, x2, y2 = bbox
        return (start_x <= x1 <= end_x) and (start_y <= y1 <= end_y) and (start_x <= x2 <= end_x) and (start_y <= y2 <= end_y)

    def draw_zone(self, frame, zone, color):
        """
        Draw a rectangular zone on the frame
        """
        (start_x, start_y), (end_x, end_y) = zone
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

def get_available_cameras():
    """
    Detect and return a list of available camera indices
    """
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    # Set up Streamlit page
    st.title("Classroom Tracking System")
    
    # Input source selection
    input_type = st.radio("Select input source:", ["Camera", "Video File"])
    
    # Input source configuration
    if input_type == "Camera":
        # Get available cameras
        available_cameras = get_available_cameras()
        
        if not available_cameras:
            st.error("No cameras found!")
            return
        
        # Camera selection dropdown
        camera_choice = st.selectbox(
            "Select Camera:", 
            options=available_cameras, 
            format_func=lambda x: f"Camera {x}"
        )
        input_source = camera_choice
    else:
        input_source = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        if input_source is not None:
            # Save uploaded file temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(input_source.getvalue())
            input_source = "temp_video.mp4"
        else:
            st.warning("Please upload a video file.")
            return
    
    # Create a placeholder for the video stream
    stframe = st.empty()
    
    # Start tracking button
    if st.button("Start Tracking"):
        try:
            # Create and run tracker
            tracker = ClassroomTracker(input_source)
            tracker.detect_and_track(stframe)
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Release resources
            tracker.cap.release()

if __name__ == "__main__":
    main()