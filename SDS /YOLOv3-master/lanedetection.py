import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/shrestabanerjee/Desktop/tata/sample videos/NO20231210-133016-006837F.MP4')

# Initialize variables for smoothing
smoothed_lines = [(0, 0, 0, 0)] * 10  # Initialize with 10 zero lines

alpha = 0.2  # Smoothing factor

while True:
    ret, frame = cap.read()
    frame = frame[:1370,:]
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    height, width = frame.shape[0], frame.shape[1]
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=10, maxLineGap=50)
    
    # Draw detected lanes on the frame
    if lines is not None:
        current_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Collect lines for smoothing
            current_lines.append((x1, y1, x2, y2))
        
        # Smooth the lines
        if len(smoothed_lines) == len(current_lines):
            smoothed_lines = np.array(smoothed_lines) * (1 - alpha) + np.array(current_lines) * alpha

        else:
            smoothed_lines = current_lines
        
        # Draw smoothed lines
       
        for smoothed_line in smoothed_lines:
            x1, y1, x2, y2 = map(int, smoothed_line)  # Convert each element to int
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    # Display the frame with detected lanes
    cv2.imshow('Lane Detection', frame) 
    

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
