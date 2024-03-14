import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("/Users/shrestabanerjee/Desktop/Lane Detection/yolov4.cfg", "")

# Load class names
with open("/Users/shrestabanerjee/Desktop/Lane Detection/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Customize class names for your specific objects
classes = ["divider", "tree", "road_sign"]

# Set the threshold for object detection confidence
conf_threshold = 0.5

# Load the input image
image = cv2.imread("input_image.jpg")
height, width = image.shape[:2]

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input blob for the network
net.setInput(blob)

# Get the names of output layers
layer_names = net.getUnconnectedOutLayersNames()

# Forward pass through the network
outputs = net.forward(layer_names)

# Initialize lists to store detected objects
detected_objects = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_name = classes[class_id]
            detected_objects.append((class_name, confidence, x, y, x + w, y + h))

# Filter objects by class names of interest
objects_of_interest = [obj for obj in detected_objects if obj[0] in ["divider", "tree", "road_sign"]]

# Draw bounding boxes on the image
for obj in objects_of_interest:
    class_name, confidence, x1, y1, x2, y2 = obj
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display or save the result
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
