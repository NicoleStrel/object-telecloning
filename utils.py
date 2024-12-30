import numpy as np
import cv2
import utils

#the landmarks for the eyes based on Mediapipe's face mesh model
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(eye, landmarks):
    """Calculate the Eye Aspect Ratio (EAR)"""

    # Define the landmarks for the eye
    if eye == 'left':
        l0, l1, l2, l3, l4, l5 = tuple(LEFT_EYE)
    else:
        l0, l1, l2, l3, l4, l5 = tuple(RIGHT_EYE)

    # Distances
    vertical1 = np.linalg.norm(np.array(landmarks[l1]) - np.array(landmarks[l5]))
    vertical2 = np.linalg.norm(np.array(landmarks[l2]) - np.array(landmarks[l4]))
    horizontal = np.linalg.norm(np.array(landmarks[l0]) - np.array(landmarks[l3]))

    # EAR calculation
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def detect_object_within_hand(frame, hand_bbox, hand_id, objects_in_hands):
    """Detect objects within the detected hand region."""
    x, y, w, h = hand_bbox
    
    # Ensure the bounding box is within frame bounds
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = max(0, x), max(0, y), min(w, frame_width - x), min(h, frame_height - y)

    # Crop the hand region and ensure non empty
    hand_region = frame[y:y+h, x:x+w]
    if hand_region.size == 0: 
        return frame

    # grayscale, blur, and threshold the hand region
    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            boundingRect = cv2.boundingRect(contour)
            cx, cy, cw, ch = boundingRect
            cv2.rectangle(hand_region, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)

            # Save the object image and location
            objects_in_hands[hand_id] = {"img": hand_region[cy:cy+ch, cx:cx+cw], "bbox": boundingRect}

    # Replace the modified hand region back into the frame
    frame[y:y+h, x:x+w] = hand_region
    return frame

def transfer_object_between_hands(frame, hand_dict, transfer_hand_id, objects_in_hands):
    """Transfer an object from one hand to another if possible."""
    idx_mapping = {0: 1, 1: 0}
    holding_hand_id = idx_mapping[transfer_hand_id]
    x, y, w, h = hand_dict[transfer_hand_id]

    if holding_hand_id in objects_in_hands:
        print("Transferring object from hand {} to hand {}".format(holding_hand_id, transfer_hand_id))
        object_dict = objects_in_hands.pop(holding_hand_id)
        object_image = object_dict["img"]
        cx, cy, cw, ch = object_dict["bbox"]

        # Place the object in the center of the new hand region
        center_x, center_y = x + w // 2, y + h // 2
        obj_h, obj_w, _ = object_image.shape

        # Calculate new position
        start_x = max(center_x - obj_w // 2, 0)
        start_y = max(center_y - obj_h // 2, 0)
        end_x = min(start_x + obj_w, frame.shape[1])
        end_y = min(start_y + obj_h, frame.shape[0])

        # Place the object in the new hand
        frame[start_y:end_y, start_x:end_x] = object_image[:end_y-start_y, :end_x-start_x]

    return frame


def find_hand_bbox_and_draw(frame, hand_landmarks):
    """Find and draw the bounding box around the hand."""
    h, w, _ = frame.shape
    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
    hand_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return hand_bbox

def find_eye_landmarks_and_draw(frame, face_landmarks):
    """Find an draw the eye landmarks on the frame."""
    landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]
    for idx in utils.LEFT_EYE + utils.RIGHT_EYE:
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    return landmarks