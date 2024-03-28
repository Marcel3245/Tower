import numpy as np
import cv2
from utils import CvFpsCalc
from utils import gestureRecognition
import serial 
import time

# Arduino communication
# arduino = serial.Serial(port='COM9', baudrate=115200, timeout=.1) 
# def write_read(x): 
#     arduino.write(bytes(x, 'utf-8')) 
#     time.sleep(0.001) 
#     data = arduino.readline() 
#     return data 

# FPS Measurement ########################################################
cvFpsCalc = CvFpsCalc(buffer_len=10)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/pictures/output/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
profileCascadePath = "Cascades/haarcascade_profileface.xml"
profileCascade = cv2.CascadeClassifier(profileCascadePath)

# Color in BGR
blue = (255,0,0)
green = (3, 252, 7)
white = (255,255,255)
black = (0, 0, 0)

# Thickness of rectangle drawn around faces
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
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

# Camera settings
WIDTH = 800
HEIGHT = 600
cap = cv2.VideoCapture(0)
cap.set(3,WIDTH) # set Width
cap.set(4,HEIGHT) # set Height
minW = 0.05*cap.get(3)
minH = 0.05*cap.get(4)

# Define the calculate_distance function
actual_width = 15.0
focal_length = 1000.0
def calculate_face_position(x, y, w, h):
    distance = (actual_width * focal_length) / w
    Xcenter = WIDTH/2
    Ycenter = HEIGHT/2
    XFaceFcenter = x+w/2 -Xcenter
    YFaceFcenter = y+h/2 -(Ycenter*0.8) #Correction to set center higher in Y-axis

    horizontalAngle = round(np.arcsin(np.absolute(actual_width*XFaceFcenter/w)/distance) * 180/np.pi, 2)
    verticalAngle = round(np.arcsin(np.absolute(actual_width*YFaceFcenter/w)/distance) * 180/np.pi, 2)

    if XFaceFcenter < 0:
        horizontalAngle*=-1
    if YFaceFcenter > 0:
        verticalAngle*=-1
    
    return horizontalAngle, verticalAngle

def drawRectangle(image, color, faces):
	for (x, y, w, h) in faces:  image = cv2.rectangle(img,(x,y),(x+w,y+h),color,thickness)
	return image

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

def detectFace(faceCascade):
    face = faceCascade.detectMultiScale(
        gray,     
        scaleFactor = 1.2,    #Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors = 5,     #Parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives.
        minSize = (int(minW), int(minH))    #Minimum rectangle size to be considered a face.
    )
    return face

while True:
    fps = cvFpsCalc.get()
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detectFace(faceCascade)
    profile = detectFace(profileCascade)
    # Detect profile faces in the flipped image to detect profile faces facing right
    flipped = cv2.flip(img, 1)
    profileFlipped  = profileCascade.detectMultiScale(
        flipped,     
        scaleFactor = 1.2,    
        minNeighbors = 5,   
        minSize = (int(minW), int(minH))
    )

    for (x,y,w,h) in faces:     #Rectangle for face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is more than 100> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        if (id == 'Marcel'):
            drawRectangle(img, green, faces)
            putText(img, f"Horizontal angle: {calculate_face_position(x, y, w, h)[0]} degrees", (x, y + h + 15), 0.5, green, 1)
            putText(img, f"Vertical angle: {calculate_face_position(x, y, w, h)[1]} degrees", (x, y + h + 35), 0.5, green, 1)
            detectGesture = gestureRecognition(img)
            detectGesture.detectGesture()
            putText(img, str(detectGesture.detectGesture()), (10, 60), 1, black, 2)
            # arduinoValue = f'{calculate_face_position(x, y, w, h)[0]}, {calculate_face_position(x, y, w, h)[1]}'
            # write_read(arduinoValue)
        
        putText(img, id, (x+5,y-5), 1, blue, 2)
        putText(img, confidence, (x,y+h-5), 1, blue, 1)

    if (all(elem is False for elem in faces)):  
        drawRectangle(img, blue, profile)
        for (x, y, w, h) in profileFlipped:  
            image = cv2.rectangle(img,(WIDTH-x,y),(WIDTH-x-w,y+h),blue,thickness)

    putText(img, "FPS:" + str(fps), (10, 30), 1, black, 2)
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()
