from picamera2 import Picamera2
import numpy as np
import cv2
# import heapq
import time
import board
from utils import CvFpsCalc, gestureRecognition
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from libcamera import controls

# Global Variables
currentPosition_x = 90.00
currentPosition_y = 150.00

# Custom pulse widths
servo_min = 500  # 500us corresponds to 0 degrees
servo_max = 2500 # 2500us corresponds to 180 degrees

i2c = board.I2C()
pca = PCA9685(i2c)
pca.frequency = 50

# Camera settings
WIDTH = 1280
HEIGHT = 720
picam2 = Picamera2()
picam2.preview_configuration.main.size = (WIDTH, HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.start()

# Enable continuous autofocus
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

minW = 20
minH = 20

class FaceDetectionTower:
    def __init__(self):
        global currentPosition_x, currentPosition_y
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.minW = 20
        self.minH = 20

        # Color in BGR
        self.blue = (255, 0, 0)
        self.green = (3, 252, 7)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        # Thickness of rectangle drawn around faces
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS Measurement
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Load face recognizer and cascade
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('training/pictures/output/trainer.yml')
        self.faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
        
        # Load names and IDs
        self.names = ['None']
        names_ids_file = 'training/pictures/output/names-ids.txt'
        with open(names_ids_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.split(',')[1].strip()
                self.names.append(name)

     def write_servo(self, angle, servo_channel):
        my_servo = servo.Servo(pca.channels[servo_channel], actuation_range=180, min_pulse=servo_min, max_pulse=servo_max)
        if servo_channel == 14:
            currentValue = currentPosition_x
        elif servo_channel == 15:
            currentValue = currentPosition_y

        targetValue = angle
        diff = targetValue - currentValue
        for t in range(20):
            y = (-3*diff/4000)*t**2 + (3*diff/200)*t
            currentValue += np.round(y, 2)
            my_servo.angle = currentValue
            time.sleep(.01)
        
    def startingPosition(self):
        global currentPosition_x, currentPosition_y
        currentPosition_x = 90.00
        currentPosition_y = 150.00
        servo_x = servo.Servo(pca.channels[14], actuation_range=180, min_pulse=servo_min, max_pulse=servo_max)
        servo_y = servo.Servo(pca.channels[15], actuation_range=180, min_pulse=servo_min, max_pulse=servo_max)
        servo_x.angle = currentPosition_x
        servo_y.angle = currentPosition_y
        print("Starting position: theta_x = 90.00 degrees, theta_y = 180.00 degrees")
        time.sleep(.1)
        

    def calculate_face_position(self, x, y, w, h):
        actual_width = 15.0
        focal_length = 1570.14
        distance = (actual_width * focal_length) / w
        Xcenter = self.WIDTH / 2
        Ycenter = self.HEIGHT / 2
        XFaceFcenter = x + w / 2 - Xcenter
        YFaceFcenter = y + h / 2 - Ycenter

        horizontalAngle = round(np.arcsin(np.absolute(actual_width * XFaceFcenter / w) / distance) * 180 / np.pi * 0.75, 2)
        verticalAngle = round(np.arcsin(np.absolute(actual_width * YFaceFcenter / w) / distance) * 180 / np.pi * 0.75, 2)

        if XFaceFcenter < 0:
            horizontalAngle *= -1
        if YFaceFcenter > 0:
            verticalAngle *= -1

        return horizontalAngle, verticalAngle

    def face_detection_process(self):
        global currentPosition_x, currentPosition_y
        while True:
            fps = self.cvFpsCalc.get()
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(self.minW), int(self.minH))
            )
            for (x, y, w, h) in faces:
                id, confidence = self.recognizer.predict(gray[y:y + h, x: x + w])
                if confidence < 100:
                    id = self.names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                if id == 'Marcel':
                    detectGesture = gestureRecognition(frame)
                    theta_x, theta_y = self.calculate_face_position(x, y, w, h)[0], self.calculate_face_position(x, y, h, w)[1]
                    currentPosition_x -= theta_x
                    currentPosition_y -= theta_y

                    if currentPosition_x > 180 or currentPosition_x < 0:
                        print("Servo X exceed its position!")
                        self.startingPosition()
                    if currentPosition_y > 180 or currentPosition_y < 0:
                        print("Servo Y exceed its position!")
                        self.startingPosition()
                    else:
                        self.write_servo(round(currentPosition_x, 1), 14)
                        self.write_servo(round(currentPosition_y, 1), 15)
                        print(detectGesture.detectGesture())
                        
            cv2.imshow('Video', frame)
        
            k = cv2.waitKey(30) & 0xff
            if k == 27:  # press 'ESC' to quit
                picam2.release()
                print("\n [INFO] Exiting Program and cleanup stuff")
                break


if __name__ == "__main__":
    face_detection = FaceDetectionTower()
    face_detection.startingPosition()
    face_detection.face_detection_process()
