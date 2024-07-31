import cv2
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Draw YOLOv8 Results on Video")
    parser.add_argument("-v", "--video", type=str, help="Path to the input video file")
    parser.add_argument("-j", "--json", type=str, help="Path to the JSON file with detection results")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Load the JSON file
with open(args.json, "r") as f:
    frame_data = json.load(f)

# Open the video file
cap = cv2.VideoCapture(args.video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Get the corresponding data for the current frame
    if frame_idx < len(frame_data):
        objects = frame_data[frame_idx]["objects"]
        
        # Draw the bounding boxes and labels on the frame
        for obj in objects:
            x1 = int(obj["box"]["x1"] * frame_width)
            y1 = int(obj["box"]["y1"] * frame_height)
            x2 = int(obj["box"]["x2"] * frame_width)
            y2 = int(obj["box"]["y2"] * frame_height)
            label = obj["name"]
            tracking_id = obj["track_id"]

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw the label
            cv2.putText(frame, f"{label}_{tracking_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame with the drawn annotations
    cv2.imshow("YOLOv8 Tracking", frame)
    
    # Increment the frame index
    frame_idx += 1

    # Set the delay according to the video's FPS
    delay = int(1000 / fps)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
