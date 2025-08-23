import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data/train'

data = []
labels = []

print("Starting advanced data processing (translation and scale normalization)...")

for dir_ in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    
    print(f"Processing directory: {dir_}")
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if not img_path.endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # --- ADVANCED NORMALIZATION (Translation + Scale) ---
            data_point_raw = []
            x_ = []
            y_ = []
            
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                
            for landmark in hand_landmarks.landmark:
                # Make coordinates relative to the wrist (translation)
                data_point_raw.append(landmark.x - min(x_))
                data_point_raw.append(landmark.y - min(y_))

            # Find the maximum absolute value to scale by
            max_value = max(map(abs, data_point_raw))
            if max_value == 0: continue # Avoid division by zero

            # Normalize by dividing by the max absolute value (scale)
            data_point_normalized = [val / max_value for val in data_point_raw]
            
            data.append(data_point_normalized)
            labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\nAdvanced processing complete!")
print(f"Saved {len(data)} fully normalized data points to data.pickle")