# Made and documented by Blijonas

import cv2
import os
import time
import threading

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the default camera
cap = cv2.VideoCapture(0)

# Create a directory to save the face screenshots if it doesn't exist already
save_dir = "face_screenshots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set the cooldown time between screenshots
cooldown_time = 100

# Define a thread class for taking and saving screenshots
class ScreenshotThread(threading.Thread):
    def __init__(self, frame):
        threading.Thread.__init__(self)
        self.frame = frame
        
    def run(self):
        # Generate a unique filename for the screenshot using the current timestamp
        timestamp = int(time.time())
        filename = "face_screenshot_{}.jpg".format(timestamp)
        # Save the screenshot in the designated directory
        cv2.imwrite(os.path.join(save_dir, filename), self.frame)
        # Sleep for the cooldown time before taking another screenshot
        time.sleep(cooldown_time)

# Loop until the user presses 'q' to quit
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face in the original color image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        # Create a new thread for taking and saving a screenshot of the current frame
        screenshot_thread = ScreenshotThread(frame)
        screenshot_thread.start()
    
    # Show the original color image with face rectangles
    cv2.imshow('Face Detection', frame) 
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()