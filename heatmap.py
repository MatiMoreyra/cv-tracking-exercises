import cv2
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Propagated Heatmap from YOLOv8 Detections with Reduced Resolution")
    parser.add_argument("-v", "--video", type=str, help="Path to the input video file")
    parser.add_argument("-j", "--json", type=str, help="Path to the JSON file with detection results")
    return parser.parse_args()

OUTPUT_SCALE = 0.5

# Parse command-line arguments
args = parse_args()

# Load the JSON file
with open(args.json, "r") as f:
    data = json.load(f)

# Open the video file
cap = cv2.VideoCapture(args.video)
w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reduce resolution by a factor of 0.1. This will speed up the processing and give a better overview of the heatmap.
scale = 0.1
w_res = int(w_orig * scale)
h_res = int(h_orig * scale)

# Constants
DIST_BIAS = 100
SAMPLE_RATE = 2

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Resize the frame to reduced resolution
    frame_res = cv2.resize(frame, (w_res, h_res))
    
    # Create an empty map of heat values at reduced resolution
    heat_values = np.zeros((h_res, w_res), dtype=np.float32)
    
    # Get the corresponding data for the current frame
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    if idx % SAMPLE_RATE != 0:
        continue

    if idx < len(data):
        objs = data[idx]["objects"]
        
        # Update the heat values based on detected objects
        for obj in objs:
            if obj["name"] == "person":
                # Scale the bounding box to image resolution
                x1 = int(obj["box"]["x1"] * w_res)
                y1 = int(obj["box"]["y1"] * h_res)
                x2 = int(obj["box"]["x2"] * w_res)
                y2 = int(obj["box"]["y2"] * h_res)

                # Calculate the centroid of the bounding box in scaled coordinates
                cx_res = (x1 + x2) // 2
                cy_res = (y1 + y2) // 2

                # Propagate values using inverse square of the distance
                for i in range(h_res):
                    for j in range(w_res):
                        dist_sq = DIST_BIAS + ((cx_res - j)**2 + (cy_res - i)**2)
                        if dist_sq > 0:
                            heat_values[i, j] += 1 / dist_sq

    # Normalize and convert the heat values map to a heatmap
    heat_values = cv2.normalize(heat_values, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heat_values.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the original frame
    overlay = cv2.addWeighted(frame_res, 0.6, heatmap, 0.4, 0)
    
    overlay = cv2.resize(overlay, (int(w_orig * OUTPUT_SCALE), int(h_orig * OUTPUT_SCALE)))

    # Display the overlay frame
    cv2.imshow("Propagated Heatmap", overlay)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
