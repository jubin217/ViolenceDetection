import cv2
import numpy as np
import keras
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from termcolor import colored
import threading
from collections import deque
import time

# ------------------- Load models -------------------
print("[INFO] Loading VGG16 base model...")
base_model = VGG16(weights='imagenet', include_top=True)
transfer_layer = base_model.get_layer('fc2')
feature_extractor = Model(inputs=base_model.input, outputs=transfer_layer.output)

print("[INFO] Loading Violence Detection model...")
violence_model = keras.models.load_model('model/vlstm_92.h5')

# ------------------- Adaptive Parameters -------------------
img_size = 224
sequence_length = 20
frame_skip = 5
buffer_size = sequence_length

# ------------------- Global variables -------------------
frame_buffer = deque(maxlen=buffer_size)
latest_prediction = "CALIBRATING"
prediction_color = (0, 255, 255)
confidence = 0.0
processing = False

# Adaptive detection
calibration_complete = False
normal_confidence_values = []
violent_detections = 0
total_detections = 0

def get_transfer_values_from_frames(frames):
    """Match the exact preprocessing from your working infer.py"""
    batch = np.array(frames, dtype=np.float16)
    batch = (batch / 255.).astype(np.float16)
    transfer_values = feature_extractor.predict(batch, verbose=0)
    return transfer_values

def process_frames():
    global latest_prediction, prediction_color, confidence, processing
    global calibration_complete, normal_confidence_values, violent_detections, total_detections
    
    while True:
        if len(frame_buffer) >= sequence_length and not processing:
            processing = True
            
            images = list(frame_buffer)
            
            try:
                transfer_values = get_transfer_values_from_frames(images)
                lstm_input = transfer_values[np.newaxis, ...]
                pred = violence_model.predict(lstm_input, verbose=0)
                
                violent_confidence = float(pred[0][0])
                non_violent_confidence = float(pred[0][1])
                total_detections += 1
                
                # CALIBRATION PHASE: First 50 predictions establish baseline
                if not calibration_complete:
                    normal_confidence_values.append(violent_confidence)
                    
                    if len(normal_confidence_values) >= 50:
                        calibration_complete = True
                        baseline_violent = np.percentile(normal_confidence_values, 75)  # Use 75th percentile
                        print(f"Calibration complete. Normal violent range: {np.min(normal_confidence_values):.3f} - {np.max(normal_confidence_values):.3f}")
                        print(f"Baseline (75th %): {baseline_violent:.3f}")
                    
                    latest_prediction = f"CALIBRATING {len(normal_confidence_values)}/50"
                    prediction_color = (255, 165, 0)
                    confidence = round(violent_confidence * 100, 2)
                
                else:
                    # ADAPTIVE DETECTION LOGIC
                    baseline = np.percentile(normal_confidence_values, 75)
                    
                    # Dynamic threshold based on baseline
                    if violent_confidence > 0.8:  # High confidence violence
                        latest_prediction = "VIOLENT!"
                        prediction_color = (0, 0, 255)
                        confidence = round(violent_confidence * 100, 2)
                        violent_detections += 1
                        print(colored(f"🔴 HIGH CONFIDENCE VIOLENCE: {confidence}%", "red", attrs=['bold']))
                    
                    elif violent_confidence > baseline + 0.25:  # Significant increase from normal
                        latest_prediction = "VIOLENT"
                        prediction_color = (0, 100, 255)  # Orange-red
                        confidence = round(violent_confidence * 100, 2)
                        violent_detections += 1
                        print(colored(f"🟠 Violence detected: {confidence}% (Baseline: {baseline:.3f})", "red"))
                    
                    elif violent_confidence > baseline + 0.15:  # Slight increase
                        latest_prediction = "SUSPICIOUS"
                        prediction_color = (0, 255, 255)  # Yellow
                        confidence = round(violent_confidence * 100, 2)
                        print(f"🟡 Suspicious activity: {confidence}%")
                    
                    else:
                        latest_prediction = "NON-VIOLENT"
                        prediction_color = (0, 255, 0)
                        confidence = round(non_violent_confidence * 100, 2)
                    
                    # Update baseline occasionally with non-violent samples
                    if violent_confidence < baseline and len(normal_confidence_values) < 100:
                        normal_confidence_values.append(violent_confidence)
                        
                    # Debug info
                    if total_detections % 10 == 0:
                        print(f"Detection #{total_detections}: V={violent_confidence:.3f}, NV={non_violent_confidence:.3f}, Baseline={baseline:.3f}")
                        
            except Exception as e:
                print(f"Prediction error: {e}")
                latest_prediction = "Error"
                prediction_color = (0, 255, 255)
            
            processing = False
        
        time.sleep(0.01)

# Start processing thread
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

# ------------------- Webcam setup -------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("[INFO] Webcam started. Please sit normally during calibration.")
print("[INFO] After calibration, test with actual fighting movements")
print("[INFO] Press 'q' to quit, 'r' to recalibrate, 'd' for debug info")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    
    if frame_count % frame_skip == 0:
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(RGB_img, (img_size, img_size))
        resized = resized.astype(np.float16)
        frame_buffer.append(resized)
    
    # Display frame
    display_frame = frame.copy()
    
    # Main prediction
    cv2.putText(display_frame, latest_prediction, (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, prediction_color, 2, cv2.LINE_AA)
    cv2.putText(display_frame, f"{confidence}%", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, prediction_color, 2, cv2.LINE_AA)
    
    # Statistics
    if calibration_complete:
        baseline = np.percentile(normal_confidence_values, 75)
        detection_rate = violent_detections / max(1, total_detections - 50) * 100
        
        cv2.putText(display_frame, f"Baseline: {baseline:.3f}", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"Violent detections: {violent_detections}", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"Detection rate: {detection_rate:.1f}%", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    if processing:
        cv2.putText(display_frame, "Processing...", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Violence Detection (Adaptive Mode)", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        calibration_complete = False
        normal_confidence_values.clear()
        violent_detections = 0
        total_detections = 0
        frame_buffer.clear()
        latest_prediction = "RECALIBRATING"
        prediction_color = (255, 165, 0)
        print("Recalibration started...")
    elif key == ord('d'):
        if normal_confidence_values:
            print(f"\n=== DEBUG INFO ===")
            print(f"Normal samples: {len(normal_confidence_values)}")
            print(f"Violent confidence stats: min={np.min(normal_confidence_values):.3f}, max={np.max(normal_confidence_values):.3f}")
            print(f"25th percentile: {np.percentile(normal_confidence_values, 25):.3f}")
            print(f"75th percentile: {np.percentile(normal_confidence_values, 75):.3f}")
            print(f"Violent detections: {violent_detections}/{total_detections}")

cap.release()
cv2.destroyAllWindows()

print(f"\n=== FINAL STATISTICS ===")
print(f"Total detections: {total_detections}")
print(f"Violent detections: {violent_detections}")
print(f"Detection rate: {violent_detections/max(1, total_detections)*100:.1f}%")