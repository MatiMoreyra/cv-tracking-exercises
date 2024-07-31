from typing import List
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results as YOLOResults
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Tracking on Video")
    parser.add_argument("-v","--video", type=str, help="Path to the input video file")
    parser.add_argument("-m","--model", type=str, help="Path to the YOLOv8 model weights")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Open the video file
cap = cv2.VideoCapture(args.video)

model = YOLO(args.model)

frame_number = 0
frame_data = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        h, w = frame.shape[:2]

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results: List[YOLOResults] = model.track(frame, persist=True, iou=0.5)

        objects = json.loads(results[0].tojson())

        # Normalize the bounding box coordinates to the range [0, 1]
        for object in objects:
            object["box"]["x1"] = (object["box"]["x1"]) / w
            object["box"]["y1"] = (object["box"]["y1"]) / h
            object["box"]["x2"] = (object["box"]["x2"]) / w
            object["box"]["y2"] = (object["box"]["y2"]) / h

        frame_dict = {
            "frame_number": frame_number,
            "objects": objects
        }

        frame_data.append(frame_dict)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Increment the frame number
        frame_number += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Dump the frame data to a file
with open(args.video + ".json", "w") as f:
    json.dump(frame_data, f, indent=4)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
