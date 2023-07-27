import cv2

def select_roi(event, x, y, flags, param):
    # Declare the global variables to modify them
    global roi_top_left, roi_bottom_right, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_top_left = (x, y)
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        roi_bottom_right = (x, y)
        cropping = False

        # Draw a rectangle around the selected ROI
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

        # Display the ROI coordinates
        text = f"ROI: ({roi_top_left[0]}, {roi_top_left[1]}) - ({roi_bottom_right[0]}, {roi_bottom_right[1]})"
        cv2.putText(frame_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Select ROI', frame_copy)

# Initialize variables
roi_top_left = None
roi_bottom_right = None
cropping = False

# Load the video
video_path = '/Users/revanthgottuparthy/Downloads/sample_video_480p.mov'
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video frame.")
    exit()

# Set the mouse callback to select the ROI
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', select_roi)

cv2.imshow('Select ROI', frame)

while True:
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27 or roi_top_left and roi_bottom_right:
        break

cv2.destroyAllWindows()

# Check if ROI coordinates have been set
if roi_top_left and roi_bottom_right:
    x, y = roi_top_left
    w = roi_bottom_right[0] - roi_top_left[0]
    h = roi_bottom_right[1] - roi_top_left[1]

    print("ROI Coordinates:")
    print(f"Top Left: ({x}, {y})")
    print(f"Bottom Right: ({x+w}, {y+h})")

    # Crop the frame to the ROI
    roi_frame = frame[y:y+h, x:x+w]

    # Do further processing with the ROI frame if needed
    # For example, you can pass the 'roi_frame' to your hand detection function.

    # Display the cropped ROI frame
    cv2.imshow('ROI Frame', roi_frame)
    cv2.waitKey(0)

else:
    print("Error: ROI not selected.")
