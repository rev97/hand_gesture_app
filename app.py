from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_hands_in_video(video_path):
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

            # Draw hand landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        # If hands are no longer detected, save the gesture range
        elif hands_detected:
            hands_detected = False
            hand_gestures.append((start_frame, end_frame))

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return hand_gestures

@app.route('/', methods=['GET', 'POST'])
def index():
    frames = None
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

            # Process the video and detect hands
            frames = detect_hands_in_video(video_path)

            # Remove the video file after processing (optional)
            os.remove(video_path)

    return render_template('index.html', frames=frames)

if __name__ == "__main__":
    app.run(debug=True)
