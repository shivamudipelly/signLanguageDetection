# import cv2
# import os
# import mediapipe as mp
# import string # We'll use this to get all the letters easily

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # --- Configuration ---
# # Path to the directory where the dataset will be saved
# DATA_DIR = './data/train'

# # --- NEW: Signs to collect (A-Z and a 'Palm'/'Nothing' class) ---
# # We will collect A-Z and also a neutral sign for when no letter is shown.
# SIGNS = list(string.ascii_uppercase) + ['Palm']

# # Number of images to collect per sign
# num_images_per_sign = 120
# # ---------------------

# # Create the main dataset directory if it doesn't exist
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# print(f"Will be collecting data for the following signs: {SIGNS}")
# print(f"Total signs to collect: {len(SIGNS)}")

# # --- Data Collection Loop ---
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# for sign in SIGNS:
#     # Create a directory for the current sign
#     sign_dir = os.path.join(DATA_DIR, sign)
#     if not os.path.exists(sign_dir):
#         os.makedirs(sign_dir)

#     print(f'Starting collection for sign: {sign}')

#     # Prompt the user to get ready
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break
        
#         frame = cv2.flip(frame, 1)

#         # Display instructions
#         cv2.putText(frame, f"Ready? Press 'Space' to collect for '{sign}'", 
#                     (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow('ASL Data Collection', frame)

#         # Wait for spacebar
#         if cv2.waitKey(1) == 32:
#             break

#     # Start collecting images
#     img_counter = 0
#     while img_counter < num_images_per_sign:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame = cv2.flip(frame, 1)
        
#         cv2.putText(frame, f"Collecting... Sign: '{sign}', Image: {img_counter + 1}/{num_images_per_sign}",
#                     (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

#         img_name = os.path.join(sign_dir, f'{sign}_{img_counter}.jpg')
#         cv2.imwrite(img_name, frame)
#         print(f"Saved {img_name}")
#         img_counter += 1
        
#         cv2.imshow('ASL Data Collection', frame)
#         cv2.waitKey(100) # 100ms delay

#     print(f"Finished collecting for sign '{sign}'.")
#     cv2.waitKey(2000) # 2-second pause before the next sign

# print(f"Data collection complete! Dataset saved in '{DATA_DIR}' directory.")
# print("Press 'q' in the camera window to quit.")

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# --- Configuration ---
# Path to the directory where the dataset will be saved
DATA_DIR = './data/train'

# --- NEW: Signs to collect (Numbers 0-9) ---
SIGNS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Number of images to collect per sign
num_images_per_sign = 120
# ---------------------

# The script will add new folders to your existing DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print(f"Will be collecting data for the following signs: {SIGNS}")

# --- Data Collection Loop ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for sign in SIGNS:
    # Create a directory for the current sign
    sign_dir = os.path.join(DATA_DIR, sign)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    print(f'Starting collection for sign: {sign}')

    # Prompt the user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame = cv2.flip(frame, 1)

        # Display instructions
        cv2.putText(frame, f"Ready? Press 'Space' to collect for '{sign}'", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('ASL Data Collection', frame)

        # Wait for spacebar
        if cv2.waitKey(1) == 32:
            break

    # Start collecting images
    img_counter = 0
    while img_counter < num_images_per_sign:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"Collecting... Sign: '{sign}', Image: {img_counter + 1}/{num_images_per_sign}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        img_name = os.path.join(sign_dir, f'{sign}_{img_counter}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        img_counter += 1
        
        cv2.imshow('ASL Data Collection', frame)
        cv2.waitKey(100) # 100ms delay

    print(f"Finished collecting for sign '{sign}'.")
    cv2.waitKey(2000) # 2-second pause before the next sign

print(f"Data collection complete for numbers!")
cap.release()
cv2.destroyAllWindows()