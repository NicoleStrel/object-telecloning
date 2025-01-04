import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("OpenCV version:", cv2.__version__)
import tensorflow as tf # to speed up import of mediapipe
print("TensorFlow imported successfully!")
import mediapipe as mp
import utils

# Initialize Mediapipe 
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def main():
    cap = cv2.VideoCapture(0)
    blink_threshold = 0.25  
    consecutive_frames = 2  # Frames needed to confirm a blink of the eye
    blink_counter = 0
    blink_transfer = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # flip to selfie mode and convert to color
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for hands
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_dict = {}
            objects_in_hands = {}
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # hand landmarks and bbox
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_bbox = utils.find_hand_bbox_and_draw(frame, hand_landmarks)
                hand_dict[idx] = hand_bbox

                # Detect object within hand
                frame = utils.detect_object_within_hand(frame, hand_bbox, idx, objects_in_hands)
            
            if len(hand_dict) == 2 and blink_transfer:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # Detect or transfer objects
                    frame = utils.transfer_object_between_hands(frame, hand_dict, idx, objects_in_hands)

        # Process frame for face
        face_results = face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = utils.find_eye_landmarks_and_draw(frame, face_landmarks)

                # Calculate EAR for both eyes
                left_ear = utils.calculate_ear('left', landmarks)
                right_ear = utils.calculate_ear('right', landmarks)

                # Check if blink is detected
                if left_ear < blink_threshold and right_ear < blink_threshold:
                        blink_counter += 1
                else:
                    if blink_counter >= consecutive_frames:
                        print("Both Eyes Blinked!")
                        blink_transfer = not blink_transfer
                    blink_counter = 0

        cv2.imshow("Object Telecloning", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()