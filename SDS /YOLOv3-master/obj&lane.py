import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Start the video stream
stream = cv2.VideoCapture('/Users/shrestabanerjee/Downloads/videoplayback (online-video-cutter.com).mp4')

while True:
    # Read a frame from the video stream
    ret, frame = stream.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Perform object detection
    bbox, label, conf = cv.detect_common_objects(frame)

    # Draw bounding boxes and labels on the frame
    output_frame = draw_bbox(frame, bbox, label, conf)

    # Display the frame with object detections
    cv2.imshow('Object Detection', output_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
stream.release()
cv2.destroyAllWindows()
