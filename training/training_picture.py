import cv2
import os
import time
from picamera2 import Picamera2
from libcamera import controls

# Define the folder structure
folder_name = input('Give your name: ')
path = r'.\pictures\training\{}'.format(folder_name)

# Create the directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (WIDTH, HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.start()

# Enable continuous autofocus
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the image capture duration
duration = 30  # in seconds
interval = 1  # in seconds

# Define the filename prefix for the saved images
filename_prefix = "image_"

# Start capturing images
start_time = time.time()
current_time = start_time

print("Sit still, I'm taking pictures of your face!")
time.sleep(1)

try:
    counter = 1
    while current_time - start_time <= duration:
        frame = picam2.capture_array()

        if ret:
            # Save the captured frame as an image
            filename = os.path.join(path, filename_prefix + str(counter) + ".jpg")
            cv2.imwrite(filename, frame)
            print("Image saved:", filename)

            # Wait for the specified interval before capturing the next image
            time.sleep(interval)

            counter += 1
            current_time = time.time()
        else:
            print("Error: Failed to capture frame.")
            break
        cv2.imshow('Video', frame)
        k = cv2.waitKey(30) & 0xff
        
finally:
    # Release the camera and close all OpenCV windows
    picam2.release()
    break
