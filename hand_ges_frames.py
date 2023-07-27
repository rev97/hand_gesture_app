import cv2
import mediapipe as mp

def detect_hands_in_video(video_path):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    hand_gestures = []
    hands_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = mp_hands.process(rgb_frame)

        # If hands are detected, update start_frame and end_frame
        if results.multi_hand_landmarks:
            if not hands_detected:
                hands_detected = True
                start_frame = frame_count

            end_frame = frame_count

            # For demonstration purposes, let's just display the frame with the hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # If hands are no longer detected, save the gesture range
        elif hands_detected:
            hands_detected = False
            hand_gestures.append((start_frame, end_frame))

        # Display the frame with hand landmarks
        cv2.imshow("Hand Recognition", frame)
        
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if hand_gestures:
        for i, (start, end) in enumerate(hand_gestures):
            print(f"Gesture {i+1}: Hand recognition started at frame {start} and ended at frame {end}")
    else:
        print("No hands detected in the video.")

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    detect_hands_in_video(video_path)
