import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
import collections

# GPU Configuration
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: CUDA not available, using CPU. For faster processing, install CUDA and compatible PyTorch")

# Paths and setup
HOME = os.getcwd()
SOURCE_VIDEO_PATH = "Cctv.mp4"
TARGET_VIDEO_PATH = f"{HOME}/Q result.mp4"

# Load YOLOv8 model with GPU support
model = YOLO("yolov8x.pt")
model.to(DEVICE)

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

# New tracking variables
cumulative_vehicles = 0
historical_wait_times = []  # Changed from dict to list
vehicles_exit_times = {}

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

# Queue Management Optimization
class QueueManager:
    MAX_CALLBACKS = 10000

    def __init__(self):
        self._condition = threading.Condition()
        self._waiters = set()  # Using set for O(1) lookup/removal
        self._done_callbacks = collections.deque()
        self._state = 'PENDING'

    def _remove_waiter(self, vehicle_id):
        try:
            self._waiters.remove(vehicle_id)
        except KeyError:
            pass

    def _invoke_callbacks(self, batch_size=10):
        while True:
            with self._condition:
                if not self._done_callbacks:
                    break
                current_batch = []
                for _ in range(min(batch_size, len(self._done_callbacks))):
                    if self._done_callbacks:
                        current_batch.append(self._done_callbacks.popleft())
            # Process current batch
            for callback in current_batch:
                callback()

    def add_vehicle_callback(self, fn):
        with self._condition:
            if len(self._done_callbacks) >= self.MAX_CALLBACKS:
                raise RuntimeError('Queue callback limit exceeded')
            self._done_callbacks.append(fn)

# Initialize queue manager
queue_manager = QueueManager()

# === Frame processing callback ===
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global vehicles_in_queue, vehicle_timestamps, cumulative_vehicles
    
    def process_vehicle_exit(tracker_id, index):
        def _exit_callback():
            global cumulative_vehicles
            wait_time = (index - vehicle_timestamps[tracker_id]['entry']) / frame_rate
            historical_wait_times.append(wait_time)
            vehicles_exit_times[tracker_id] = index
            cumulative_vehicles += 1
        return _exit_callback

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)

    # Update queue zone with batch processing
    in_zone_mask = queue_zone.trigger(detections=detections)
    current_queue_vehicles = set()

    for i, (tracker_id, in_zone) in enumerate(zip(detections.tracker_id, in_zone_mask)):
        if tracker_id is None:
            continue
            
        if in_zone:
            current_queue_vehicles.add(tracker_id)
            if tracker_id not in vehicle_timestamps:
                vehicle_timestamps[tracker_id] = {'entry': index}
                queue_manager._waiters.add(tracker_id)
        elif tracker_id in vehicles_in_queue:
            if 'exit' not in vehicle_timestamps[tracker_id]:
                vehicle_timestamps[tracker_id]['exit'] = index
                queue_manager._remove_waiter(tracker_id)
                queue_manager.add_vehicle_callback(process_vehicle_exit(tracker_id, index))

    # Process callbacks in batches
    queue_manager._invoke_callbacks()
    
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
        (f"Total Vehicles: {cumulative_vehicles}", (255, 255, 100)),
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

def process_batch(frames):
    """Process a batch of frames using GPU"""
    if torch.cuda.is_available():
        with torch.amp.autocast('cuda'):  # Fixed deprecated autocast
            results = model(frames, verbose=False)
    else:
        results = model(frames, verbose=False)
    return results

# === Manual Video Processing with GPU ===
BATCH_SIZE = 4  # Adjust based on GPU memory
frames_batch = []
indices_batch = []

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, frame_rate, (width, height))

with ThreadPoolExecutor(max_workers=4) as executor:
    for frame_idx in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_batch.append(frame)
        indices_batch.append(frame_idx)
        
        if len(frames_batch) == BATCH_SIZE or frame_idx == total_frames - 1:
            # Process batch with GPU
            results = process_batch(frames_batch)
            
            # Process results in parallel
            futures = []
            for frame, result, idx in zip(frames_batch, results, indices_batch):
                future = executor.submit(callback, frame, idx)
                futures.append(future)
            
            # Write processed frames
            for future in futures:
                writer.write(future.result())
            
            frames_batch = []
            indices_batch = []
            torch.cuda.empty_cache()

cap.release()
writer.release()

# === Plotting ===
plt.style.use('default')  # Use default style instead of seaborn

try:
    plt.figure(figsize=(15, 10))

    # Average Wait Time Plot
    plt.subplot(2, 2, 1)
    plt.plot(frame_indices, avg_wait_times, label="Average Wait Time", 
             color='#2E86C1', linewidth=2)
    plt.fill_between(frame_indices, avg_wait_times, alpha=0.2, color='#2E86C1')
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Wait Time (seconds)", fontsize=10)
    plt.title("Average Wait Time Over Time", fontsize=12, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Queue Length Plot
    plt.subplot(2, 2, 2)
    plt.plot(frame_indices, queue_lengths, label="Queue Length",
             color='#E67E22', linewidth=2)
    plt.fill_between(frame_indices, queue_lengths, alpha=0.2, color='#E67E22')
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Number of Vehicles", fontsize=10)
    plt.title("Queue Length Over Time", fontsize=12, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Cumulative Vehicles Plot
    plt.subplot(2, 2, 3)
    if vehicles_exit_times:  # Only plot if we have data
        cumulative_times = [vehicles_exit_times[v_id]/frame_rate for v_id in sorted(vehicles_exit_times.keys())]
        cumulative_counts = range(1, len(cumulative_times) + 1)
        plt.plot(cumulative_times, cumulative_counts, label="Total Vehicles",
                 color='#27AE60', linewidth=2)
        plt.fill_between(cumulative_times, cumulative_counts, alpha=0.2, color='#27AE60')
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Total Vehicles", fontsize=10)
    plt.title("Cumulative Vehicles Over Time", fontsize=12, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Wait Time Distribution Plot
    plt.subplot(2, 2, 4)
    if historical_wait_times:  # Only plot if we have data
        plt.hist(historical_wait_times, bins=30, color='#8E44AD', alpha=0.7,
                 edgecolor='black', linewidth=1)
    plt.xlabel("Wait Time (seconds)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.title("Wait Time Distribution", fontsize=12, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Improve overall layout
    plt.tight_layout(pad=3.0)
    plt.suptitle("Queue Analysis Dashboard", fontsize=14, y=1.02)

    # Save with higher DPI for better quality
    plt.savefig("Queue_analysis.png", dpi=300, bbox_inches='tight')
    print("📊 Analysis saved as Queue_analysis.png")

except Exception as e:
    print(f"Error generating plots: {str(e)}")

if historical_wait_times:
    print(f"Average Historical Wait Time: {np.mean(historical_wait_times):.2f} seconds")
print(f"Total Vehicles Processed: {cumulative_vehicles}")
