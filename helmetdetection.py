import cv2
import math
import cvzone
from ultralytics import YOLO
import numpy as np
from collections import Counter
import os
import sys
import time
from tqdm import tqdm

print('-'*60)
print('CHECKING FILE(S)')
print('='*60)
print(f'FILE NAME: {__file__}')
print(f'EXISTS: {os.path.exists(__file__)}')
print(f'FILE DIRECTORY: {os.getcwd()}')
print(f'PYTHON.EXE: {sys.executable}')
print(f'SCRIPT DIRECTORY: {os.path.dirname(os.path.abspath(__file__))}')
print('-'*60)

file_name = input("Enter the name of the file to check: ")
video_path = "media/"+file_name+".mp4"

# Validate input file
if not os.path.exists(video_path):
    print(f"Error: Video file {video_path} not found!")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    sys.exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = "output/"+file_name+"_output.mp4"
os.makedirs("output", exist_ok=True)

# Use more efficient codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Load models with optimized settings
print("Loading AI models...")
vehicle_model = YOLO("weight/yolov8m.pt")
helmet_model = YOLO("weight/helmetdetectormodel.pt")

# Optimize models for inference
vehicle_model.fuse()  # Fuse conv and bn layers for faster inference
helmet_model.fuse()

# Set model to evaluation mode for better performance
vehicle_model.eval()
helmet_model.eval() 

classNames = ['Helmet', 'NoHelmet']

BICYCLE_CLASS = 1
MOTORCYCLE_CLASS = 3

def get_fixed_subrectangle(x, y, w, h):
    sub_x = int(x + (2/4) * w)
    sub_y = int(y + (1/3) * h)
    sub_w = int(w / 4)
    sub_h = int(h / 3)
    return sub_x, sub_y, sub_w, sub_h

def detect_helmet_color(img, x1, y1, x2, y2):
    """Optimized helmet color detection with reduced computations."""
    try:
        # Calculate dimensions once
        w, h = x2 - x1, y2 - y1
        
        # Get optimized sub-rectangle
        sub_x = x1 + w // 2
        sub_y = y1 + h // 4
        sub_w = w // 4
        sub_h = h // 3
        
        # Clamp to image bounds
        sub_x1 = max(0, sub_x)
        sub_y1 = max(0, sub_y)
        sub_x2 = min(img.shape[1], sub_x + sub_w)
        sub_y2 = min(img.shape[0], sub_y + sub_h)
        
        # Extract sub-region directly in BGR
        sub_region = img[sub_y1:sub_y2, sub_x1:sub_x2]
        
        if sub_region.size == 0:
            sub_region = img[y1:y2, x1:x2]
        
        # Compute mean color directly in BGR space for speed
        mean_color_bgr = np.mean(sub_region.reshape(-1, 3), axis=0).astype(int)
        
        # Convert BGR to RGB for color naming
        mean_color_rgb = mean_color_bgr[::-1]  # Simple reverse for BGR to RGB
        
        color_name = rgb_to_color_name(mean_color_rgb)
        return color_name, mean_color_rgb
        
    except:
        return "Unknown", [128, 128, 128]

def rgb_to_color_name(rgb):
    """
    Convert RGB values to approximate color names.
    Input: RGB array [r, g, b] with values 0-255
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    
    # Ensure values are in valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    # Define color ranges (order matters - most specific first)
    # White and light colors
    if r > 200 and g > 200 and b > 200:
        return "White"
    # Black and very dark colors
    elif r < 60 and g < 60 and b < 60:
        return "Black"
    # Red colors (using more sensitive thresholds)
    elif r > 90 and g < 70 and b < 70:
        return "Red"
    # Blue colors
    elif r < 100 and g < 100 and b > 150:
        return "Blue"
    # Green colors
    elif r < 100 and g > 150 and b < 100:
        return "Green"
    # Yellow colors
    elif r > 150 and g > 150 and b < 100:
        return "Yellow"
    # Orange colors
    elif r > 150 and g > 100 and b < 100:
        return "Orange"
    # Purple/Magenta colors
    elif r > 150 and g < 150 and b > 150:
        return "Purple"
    # Cyan colors
    elif r < 150 and g > 150 and b > 150:
        return "Cyan"
    # Brown colors
    elif r > 120 and g > 80 and b < 80 and r > g and g > b:
        return "Brown"
    # Pink colors
    elif r > 180 and g > 150 and b > 150:
        return "Pink"
    # Gray colors (similar RGB values)
    elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30 and 60 <= r <= 200:
        return "Gray"
    # Silver (lighter gray)
    elif abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20 and r > 150:
        return "Silver"
    # Dark colors that don't fit black
    elif max(r, g, b) < 100:
        return "Dark"
    else:
        return "Mixed"

polygon = np.array([[-1,11], [1,411], [1,973], [970,982], [1134,584], [687,498], [712,0], [256,0], [199,12], [179,25]], np.int32)
polygon = polygon.reshape((-1, 1, 2))


print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_path}")
print(f"Video properties: {width}x{height} @ {fps}fps")
print(f"Total frames to process: {total_frames}")

# Initialize counters
frame_count = 0
total_helmets_detected = 0
total_without_helmets_detected = 0
total_motorcycles_detected = 0
total_bicycles_detected = 0
helmet_colors_detected = Counter()

# Pre-convert polygon points for faster testing
polygon_contour = polygon.reshape(-1, 2)

# Initialize progress bar
progress_bar = tqdm(total=total_frames, desc="Processing video", 
                   unit="frames", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.1f}%")

start_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # Process vehicles with optimized settings
    vehicle_results = vehicle_model(img, stream=True, conf=0.5, verbose=False)
    
    for r in vehicle_results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Quick center calculation and polygon test
                cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1
                
                if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                    if cls == MOTORCYCLE_CLASS:
                        total_motorcycles_detected += 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cvzone.putTextRect(img, f'Motorcycle {conf:.2f}', 
                                         (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
    
    # Process helmets with optimized settings
    helmet_results = helmet_model(img, stream=True, conf=0.3, verbose=False)
    
    for r in helmet_results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                    if cls == 0:  # With helmet
                        total_helmets_detected += 1
                        
                        # Optimized color detection (only every 5th frame for performance)
                        if frame_count % 5 == 0:
                            color_name, color_rgb = detect_helmet_color(img, x1, y1, x2, y2)
                            helmet_colors_detected[color_name] += 1
                        else:
                            color_name = "Detected"
                            color_rgb = [128, 128, 128]
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f'Helmet {conf:.2f}'
                        if color_name != "Detected":
                            text += f' - {color_name}'
                        cvzone.putTextRect(img, text, (max(0, x1), max(35, y1)), 
                                         scale=0.8, thickness=1)
                        
                        # Add color indicator
                        if color_name != "Detected":
                            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                            cv2.rectangle(img, (x1, y2-20), (x1+60, y2), color_bgr, -1)
                            cv2.rectangle(img, (x1, y2-20), (x1+60, y2), (0, 0, 0), 1)
                        
                    elif cls == 1:  # Without helmet
                        total_without_helmets_detected += 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(img, f'No Helmet {conf:.2f}', 
                                         (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
    
    # Draw detection zone
    cv2.polylines(img, [polygon], isClosed=True, color=(0,255,255), thickness=2)
    
    # Write frame
    out.write(img)
    
    # Update progress bar
    progress_bar.update(1)
    
    # Update progress bar description with current stats
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_current = frame_count / elapsed if elapsed > 0 else 0
        progress_bar.set_description(f"Processing (Helmets: {total_helmets_detected}, No Helmet: {total_without_helmets_detected}, FPS: {fps_current:.1f})")

# Close progress bar
progress_bar.close()

# Calculate final performance metrics
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n🎉 Video processing complete!")
print(f"📁 Output saved to: {output_path}")
print(f"⏱️  Processing time: {total_time:.2f} seconds")
print(f"⚡ Average FPS: {avg_fps:.2f}")
print(f"📊 Total frames processed: {frame_count}")

print(f"\n📊 Detection Summary:")
print(f"   🏍️  Motorcycles detected: {total_motorcycles_detected}")
print(f"   🪖  With Helmets detected: {total_helmets_detected}")
print(f"   🚨  Without Helmets detected: {total_without_helmets_detected}")

# Calculate compliance rate
total_riders = total_helmets_detected + total_without_helmets_detected
if total_riders > 0:
    compliance_rate = (total_helmets_detected / total_riders) * 100
    print(f"   📈 Helmet compliance rate: {compliance_rate:.1f}%")

print(f"\n🎨 Helmet Color Distribution:")
if helmet_colors_detected:
    for color, count in helmet_colors_detected.most_common():
        percentage = (count / sum(helmet_colors_detected.values())) * 100
        print(f"   {color}: {count} detections ({percentage:.1f}%)")
else:
    print("   No helmet colors detected")

print(f"\n🎯 Detection Legend:")
print(f"   🔵 Blue = Motorcycles")
print(f"   🟢 Green = With Helmet")
print(f"   🔴 Red = Without Helmet")