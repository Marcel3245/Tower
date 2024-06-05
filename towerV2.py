from picamera2 import Picamera2
import numpy as np
import cv2
from utils import CvFpsCalc
from utils import gestureRecognition
import serial 
import time 
from libcamera import controls

# FPS Measurement
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Camera settings
WIDTH = 1280
HEIGHT = 720
picam2 = Picamera2()
picam2.preview_configuration.main.size = (WIDTH, HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.start(show_preview=True)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
minW = 20
minH = 20

# Color in BGR
blue = (255, 0, 0)
green = (3, 252, 7)
white = (255, 255, 255)
black = (0, 0, 0)

# Thickness of rectangle drawn around faces
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

# Arduino communication
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
def write_read(x): 
    arduino.write(x.encode()) 
    time.sleep(0.01) 
    data = arduino.readline() 
    return data 

def putText(image, text, position, scale, color, weight):
    cv2.putText(
        image, 
        str(text), 
        position, 
        font, 
        scale, 
        color, 
        weight
    )

def calculate_face_position(x, y, w, h):
    actual_width = 15.0
    focal_length = 1570.14 # in pixels (Focal length in mm × Sensor resolution in pixels)/Sensor width in mm || Sensor Width (mm)=0.5×25.4=12.7mm, Focal Length = 4.28mm
    distance = (actual_width * focal_length) / w
    Xcenter = WIDTH/2
    Ycenter = HEIGHT/2
    XFaceFcenter = x+w/2 -Xcenter
    YFaceFcenter = y+h/2 -Ycenter

    horizontalAngle = round(np.arcsin(np.absolute(actual_width*XFaceFcenter/w)/distance) * 180/np.pi*0.75, 2)
    verticalAngle = round(np.arcsin(np.absolute(actual_width*YFaceFcenter/w)/distance) * 180/np.pi*0.75, 2)

    if XFaceFcenter < 0:
        horizontalAngle*=-1
    if YFaceFcenter > 0:
        verticalAngle*=-1
    
    return horizontalAngle, verticalAngle

def face_detection_process():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('training/pictures/output/trainer.yml')
    faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
    # Iniciate id counter
    id = 0
    # Read the names and IDs from the file
    names_ids_file = 'training/pictures/output/names-ids.txt'
    names = ['None']
    with open(names_ids_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split each line by comma and take the second part (name)
            name = line.split(',')[1].strip()  # strip() to remove leading/trailing whitespace
            names.append(name)

    while True:
        fps = cvFpsCalc.get()
        frame = picam2.capture_array()
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH))
        )
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # If confidence is more than 100> "0" : perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                # cv2.rectangle(frame, (x, y), (x + w, y + h), blue, 2)

            if (id == 'Marcel'):
                # cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
                detectGesture = gestureRecognition(frame)
                detectGesture.detectGesture()
                arduinoValue = f'{calculate_face_position(x, y, w, h)[0]}, {calculate_face_position(x, y, w, h)[1]}'
                print(f'{arduinoValue}, {str(detectGesture.detectGesture())}, {fps}')
                write_read(str(arduinoValue))
                
        cv2.imshow('Video', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit 
            picam2.release()
            print("\n [INFO] Exiting Program and cleanup stuff")
            break   

face_detection_process()
