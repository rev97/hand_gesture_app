# Rest of the code remains the same as before...
from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_bounding_box(hand_landmarks, roi_coordinates):
    # Get the bounding box around the hand landmarks
    x_min, x_max = roi_coordinates[0], roi_coordinates[0] + roi_coordinates[2]
    y_min, y_max = roi_coordinates[1], roi_coordinates[1] + roi_coordinates[3]

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * roi_coordinates[2]), int(landmark.y * roi_coordinates[3])
        x += roi_coordinates[0]
        y += roi_coordinates[1]

        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    return x_min, y_min, x_max, y_max


def detect_tapping_gesture(hand_landmarks):
    # Get the landmarks for the thumb and index finger
    thumb_landmark = hand_landmarks.landmark[4]  # Thumb tip
    index_finger_landmark = hand_landmarks.landmark[8]  # Index finger tip

    # Check if the index finger is above the thumb (vertical position)
    if index_finger_landmark.y < thumb_landmark.y:
        return True

    return False

def detect_hands_in_video(video_path, roi_coordinates):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    hand_gestures = []
    hands_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the region of interest
        x, y, w, h = roi_coordinates
        roi_frame = frame[y:y+h, x:x+w]

        # Convert the cropped image from BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

        # Process the cropped frame and detect hands
        results = mp_hands.process(rgb_frame)

        # If hands are detected, update start_frame and end_frame
        if results.multi_hand_landmarks:
            if not hands_detected:
                hands_detected = True
                start_frame = frame_count

            end_frame = frame_count

            # Check for fingers tapping gesture
            for hand_landmarks in results.multi_hand_landmarks:
                if detect_tapping_gesture(hand_landmarks):
                    # Draw a bounding box around the detected tapping gesture
                    x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, roi_coordinates)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            '''
            # Draw hand landmarks on the cropped frame
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    # Map the landmarks back to the original frame coordinates
                    x += roi_coordinates[0]
                    y += roi_coordinates[1]
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            '''

        # If hands are no longer detected, save the gesture range
        elif hands_detected:
            hands_detected = False
            hand_gestures.append((start_frame, end_frame))

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return hand_gestures

# Rest of the code remains the same as before...

def trim_and_save_video(video_path, start_frame, end_frame, output_filename):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the trimmed video in MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= start_frame and frame_count <= end_frame:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()

def calculate_time(frame, fps):
    return frame / fps

@app.route('/', methods=['GET', 'POST'])
def index():
    frames = None
    clips_info = []  # Initialize the clips_info variable outside the block
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('index.html', error='No video file selected.')

        file = request.files['video']

        if file.filename == '':
            return render_template('index.html', error='No video file selected.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            roi_coordinates = (69, 235, 266, 350)
            # Process the video and detect hands
            frames = detect_hands_in_video(video_path, roi_coordinates)

            # Trim the video and save the clips
            clips_info = []
            for i, (start_frame, end_frame) in enumerate(frames):
                output_filename = os.path.join(app.config['OUTPUT_FOLDER'], f'clip_{i + 1}.mp4')
                trim_and_save_video(video_path, start_frame, end_frame, output_filename)

                # Calculate start and end times for each clip
                cap = cv2.VideoCapture(output_filename)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                start_time = round(calculate_time(start_frame, fps),2)
                end_time = round(calculate_time(end_frame, fps),2)
                clips_info.append((start_frame, end_frame, start_time, end_time))

            cap.release()

            # Remove the video file after processing (optional)
            os.remove(video_path)

    return render_template('index.html', frames=frames, clips_info=clips_info)

if __name__ == "__main__":
    app.run(debug=True)
