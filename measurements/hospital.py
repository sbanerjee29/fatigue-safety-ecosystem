import cv2
import os
from keras.models import load_model
import numpy as np
from scipy.spatial import distance as dist
import time
import socketio

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

score = 0
thicc = 2
model = load_model('models/cnncat2.h5')
path = os.getcwd()

# Other parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.7
font_thickness = 1
font_color = (255, 255, 255)  # White

# Initialize Socket.IO
sio = socketio.Client()

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

start_time = time.time()  # Record the start time

# Socket.IO event handlers
@sio.on('connect')
def on_connect():
    print('Connected to server')

@sio.on('disconnect')
def on_disconnect():
    print('Disconnected from server')

@sio.on('location_update')
def on_location_update(data):
    print(f'Location updated: Latitude: {data["latitude"]}, Longitude: {data["longitude"]}')

@sio.on('drowsiness_detected')
def on_drowsiness_detected():
    print('Drowsiness detected!')

# Replace 'http://localhost:5000' with your server URL
sio.connect('http://localhost:5000')

# cap = cv2.VideoCapture(0)
video_path = 'WhatsApp Video 2024-01-18 at 04.08.17_48e6e8eb.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
thicc = 2
rpred = [99]
lpred = [99]

while cap.isOpened():
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_step(r_eye)
        if np.all(rpred == 1):
            lbl = 'Open'
        if np.all(rpred == 0):
            lbl = 'Closed'
        break

    r_eye_coordinates = [(x, y), (x + w // 2, y), (x + w, y), (x, y + h // 2), (x + w, y + h // 2), (x, y + h),
                         (x + w // 2, y + h), (x + w, y + h)]
    ear_right = eye_aspect_ratio(np.array(r_eye_coordinates))

    # Display the calculated ear for the right eye
    cv2.putText(frame, f'EAR Right: {ear_right:.2f}', (10, height - 100), font, 0.5, font_color, 1, cv2.LINE_AA)

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_step(l_eye)

        l_eye_coordinates = [(x, y), (x + w // 2, y), (x + w, y), (x, y + h // 2), (x + w, y + h // 2), (x, y + h),
                             (x + w // 2, y + h), (x + w, y + h)]
        ear_left = eye_aspect_ratio(np.array(l_eye_coordinates))

        # Display the calculated ear for the left eye
        cv2.putText(frame, f'EAR Left: {ear_left:.2f}', (10, height - 80), font, 0.5, font_color, 1, cv2.LINE_AA)

        if np.all(lpred == 1):
            lbl = 'Open'
        if np.all(lpred == 0):
            lbl = 'Closed'
        break

    elapsed_time = time.time() - start_time
    # print(elapsed_time)
    if elapsed_time >= 22:
        cv2.putText(frame, "Drowsiness Detected", (width // 2 - 150, height // 2), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        # Broadcast location to all clients
        sio.emit('location_update', {'latitude': 37.7749, 'longitude': -122.4194})

    if np.all(rpred == 0) and np.all(lpred == 0):
        score = ear_left = eye_aspect_ratio(np.array(l_eye_coordinates))
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, font_color, 1, cv2.LINE_AA)
    else:
        score = ear_left = eye_aspect_ratio(np.array(l_eye_coordinates))
        formatted_score = f'Head Tilt: {score:.2f}'
        cv2.putText(frame, formatted_score, (100, height - 20), font, 1, font_color, 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.imshow('frame', frame)
    formatted_score = f'Head Tilt: {score:.2f}'
    cv2.putText(frame, formatted_score, (100, height - 20), font, 1, font_color, 1, cv2.LINE_AA)

    # Display "Drowsiness Detected" after 23 seconds
    if score > 15:
        print("Drowsiness Detected")
        # Additional actions can be added here based on your requirements

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
