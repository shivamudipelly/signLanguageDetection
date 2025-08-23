import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- Load Model ---
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# --- Initialize Webcam and MediaPipe ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

# --- Parameters ---
PREDICTION_BUFFER_SIZE = 15
CONFIDENCE_THRESHOLD = 0.85 
PROBABILITY_THRESHOLD = 0.6 

prediction_history = deque(maxlen=PREDICTION_BUFFER_SIZE)
stable_prediction = ""
last_stable_prediction = "" 
sentence = "" 

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret: break

    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    bounding_box_coords = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # --- Normalization (Must match training) ---
        data_point_raw = []
        x_, y_ = [], []
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)
        for landmark in hand_landmarks.landmark:
            data_point_raw.append(landmark.x - min(x_))
            data_point_raw.append(landmark.y - min(y_))
        max_value = max(map(abs, data_point_raw))
        if max_value == 0: continue
        data_point_normalized = [val / max_value for val in data_point_raw]

        # --- Prediction with Confidence Check ---
        probabilities = model.predict_proba([np.asarray(data_point_normalized)])
        max_prob = np.max(probabilities)

        if max_prob >= PROBABILITY_THRESHOLD:
            predicted_class_index = np.argmax(probabilities)
            predicted_character = model.classes_[predicted_class_index]
            prediction_history.append(predicted_character)
        else:
            prediction_history.append(None)

        # --- Smoothing Logic ---
        if len(prediction_history) == PREDICTION_BUFFER_SIZE:
            valid_predictions = [p for p in prediction_history if p is not None]
            if valid_predictions:
                most_common = max(set(valid_predictions), key=valid_predictions.count)
                confidence = valid_predictions.count(most_common) / len(valid_predictions)
                if confidence >= CONFIDENCE_THRESHOLD:
                    stable_prediction = most_common
                else:
                    stable_prediction = ""
            else:
                stable_prediction = ""

        # --- UPDATED LOGIC: Build the sentence, ignoring 'Palm' ---
        if stable_prediction and stable_prediction != last_stable_prediction and stable_prediction != 'Palm':
            sentence += stable_prediction
            last_stable_prediction = stable_prediction

        # --- Drawing ---
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        bounding_box_coords = (x1, y1)

    else:
        # Reset when no hand is detected
        prediction_history.clear()
        stable_prediction = ""
        last_stable_prediction = "" 

    # --- Display the output sentence ---
    cv2.rectangle(frame, (0, H - 50), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, sentence, (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # --- Display the current prediction status ---
    display_text = stable_prediction if stable_prediction and stable_prediction != 'Palm' else "Unknown"
    if results.multi_hand_landmarks and bounding_box_coords:
        cv2.putText(frame, display_text, (bounding_box_coords[0], bounding_box_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('ASL to Text (A-Z)', frame)

    # --- Key controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'): # Press 'c' to clear the sentence
        sentence = ""
        last_stable_prediction = ""

cap.release()
cv2.destroyAllWindows()