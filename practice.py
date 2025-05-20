import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

# Paths and setup
HOME = os.getcwd()
SOURCE_VIDEO_PATH = "Cctv try.mp4"
TARGET_VIDEO_PATH = f"{HOME}/Q result.mp4"

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Class selection
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Annotators
box_annotator = sv.BoxAnnotator(thickness=4)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Define queue box coordinates (adjust these based on the image)
QUEUE_BOX = np.array([
    [400, 410],  # Top-left
    [1415, 410],  # Top-right
    [1415, 680],  # Bottom-right
    [400, 680]   # Bottom-left
], dtype=np.int32)

# Create a polygon zone for queue detection
queue_zone = sv.PolygonZone(polygon=QUEUE_BOX)
queue_zone_annotator = sv.PolygonZoneAnnotator(
    zone=queue_zone,
    color=sv.Color(0, 255, 0),
    thickness=2,
    text_thickness=0,
    text_scale=0
)

# BYTETracker
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# Vehicle data tracking
vehicle_timestamps = {}
vehicles_in_queue = set()

# Time series tracking lists
avg_wait_times = []
queue_lengths = []
frame_indices = []

# Suggestion system
def suggest_solution(avg_wait_time):
    if avg_wait_time > 60:
        return "Consider adding additional lanes or improving signal timings."
    elif avg_wait_time > 30:
        return "Optimize vehicle flow by adjusting staff operations."
    else:
        return "Traffic flow is efficient."

# Get frame rate
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# === Frame processing callback ===
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global vehicles_in_queue, vehicle_timestamps

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)

    # Update queue zone with current detections
    in_zone_mask = queue_zone.trigger(detections=detections)
    
    # Track vehicles in queue
    current_queue_vehicles = set()
    for i, (tracker_id, in_zone) in enumerate(zip(detections.tracker_id, in_zone_mask)):
        if tracker_id is None:
            continue
            
        if in_zone:
            current_queue_vehicles.add(tracker_id)
            if tracker_id not in vehicle_timestamps:
                vehicle_timestamps[tracker_id] = {'entry': index}
        elif tracker_id in vehicles_in_queue:
            if 'exit' not in vehicle_timestamps[tracker_id]:
                vehicle_timestamps[tracker_id]['exit'] = index

    vehicles_in_queue = current_queue_vehicles
    queue_length = len(vehicles_in_queue)

    # Calculate average wait time
    current_wait_times = []
    for v_id in vehicles_in_queue:
        if v_id in vehicle_timestamps:
            wait_time = (index - vehicle_timestamps[v_id]['entry']) / frame_rate
            current_wait_times.append(wait_time)
    
    avg_wait_time = sum(current_wait_times) / len(current_wait_times) if current_wait_times else 0

    # Update tracking lists
    avg_wait_times.append(avg_wait_time)
    queue_lengths.append(queue_length)
    frame_indices.append(index / frame_rate)

    suggestion = suggest_solution(avg_wait_time)

    # Annotate frame
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = queue_zone_annotator.annotate(scene=annotated_frame)

    # Add information panel
    panel_x, panel_y = 40, 40
    panel_width, panel_height = 650, 380
    panel_color = (0, 0, 0)
    panel_opacity = 0.4

    overlay = annotated_frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        panel_color,
        thickness=-1
    )
    cv2.addWeighted(overlay, panel_opacity, annotated_frame, 1 - panel_opacity, 0, annotated_frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    font_thickness = 2

    wait_color = (255, 255, 255)
    if avg_wait_time > 60:
        wait_color = (0, 0, 255)
    elif avg_wait_time > 30:
        wait_color = (0, 165, 255)

    overlay_items = [
        (f"Queue Length: {queue_length} vehicles", (255, 255, 255)),
        (f"Avg Wait Time: {avg_wait_time:.2f} sec", wait_color),
        (f"Solution: {suggestion}", (180, 255, 100))
    ]

    for i, (text, color) in enumerate(overlay_items):
        y_offset = panel_y + 60 + i * 80
        cv2.putText(
            annotated_frame,
            text,
            (panel_x + 30, y_offset),
            font,
            font_scale,
            color,
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return annotated_frame

# === Manual Video Processing ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, frame_rate, (width, height))

frame_gen = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
for index, frame in tqdm(enumerate(frame_gen), total=total_frames, desc="Processing Video"):
    annotated_frame = callback(frame, index)
    writer.write(annotated_frame)

writer.release()
print("✅ Video processing complete.")

# === Plotting ===
plt.figure()
plt.plot(frame_indices, avg_wait_times, label="Avg Wait Time (sec)", color='blue')
plt.xlabel("Time (sec)")
plt.ylabel("Avg Wait Time")
plt.title("Average Wait Time Over Time")
plt.grid(True)
plt.legend()
plt.savefig("avg_wait_time_plot.png")

plt.figure()
plt.plot(frame_indices, queue_lengths, label="Queue Length", color='orange')
plt.xlabel("Time (sec)")
plt.ylabel("Queue Length")
plt.title("Queue Length Over Time")
plt.grid(True)
plt.legend()
plt.savefig("queue_length_plot.png")

print("📊 Plots saved as images.")
