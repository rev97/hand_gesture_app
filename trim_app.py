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

            # Process the video and detect hands
            frames = detect_hands_in_video(video_path)

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
