from picamera2 import Picamera2
import numpy as np
import cv2
import heapq
import time
import board
from utils import CvFpsCalc, gestureRecognition
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

i2c = board.I2C() 
pca = PCA9685(i2c)
pca.frequency = 50


class FaceDetectionTower:
    def __init__(self, width=1280, height=720):
        # Camera settings
        self.WIDTH = width
        self.HEIGHT = height
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (self.WIDTH, self.HEIGHT)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.start()

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
        servo = servo.Servo(pca.channels[servo_channel])
        servo.angle = angle
        time.sleep(.1)

    def put_text(self, image, text, position, scale, color, weight):
        cv2.putText(
            image,
            str(text),
            position,
            self.font,
            scale,
            color,
            weight
        )

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
        while True:
            fps = self.cvFpsCalc.get()
            frame = self.picam2.capture_array()
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
                    horizontal, vertical = round(self.calculate_face_position(x, y, w, h)[0], 1), round(self.calculate_face_position(x, y, w, h)[1], 1)
                    self.write_servo(horizontal, 14)
                    self.write_servo(vertical, 15)
                    return detectGesture.detectGesture()

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # press 'ESC' to quit
                self.picam2.release()
                print("\n [INFO] Exiting Program and cleanup stuff")
                break

class Robot:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x2 = 0
        self.y2 = 0
        self.q1 = 0
        self.q2 = 0
        self.path_points = []  # Store path points for movement
        self.WIDTH = 300
        self.HEIGHT = 300
        self.a1 = 100
        self.a2 = 100
        
    def coordinates(self):   
            grid = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
            cell_width = self.WIDTH/len(grid)
            cell_height = self.HEIGHT/len(grid[0])
            src = [7, 8]
            dest_list = ([5, 4], [7, 3], [7, 8])
            
            for dest in dest_list:
                print(f"Searching path to destination: {dest}")
                a_star = AStarSearch(grid, src, dest)
                a_star.a_star_search()
                path = a_star.trace_path()
                src = dest
                for i in path:
                    vertical = cell_width * (i[1]-.5)
                    horizontal = cell_height * (i[0]-.5)
                    print(f'vertical: {vertical}; horizontal: {horizontal}')
                    self.path_points.append((vertical, horizontal))  # Collect path points
                    
    def movement(self):
        Robot.movement()
        if self.path_points:
            self.mx, self.my = self.path_points.pop(0)  # Update position to the next path point

        C = ((self.WIDTH/2 - self.mx)**2 + (.15 * self.HEIGHT - self.my)**2 - self.a1**2 - self.a2**2) / (2 * self.a1 * self.a2)

        if -1 <= C <= 1:  # Only calculate q1 and q2 if the target is reachable
            # Inverse Kinematics
            q2 = np.arccos(C)
            q1 = np.arctan2((.15 * self.HEIGHT - self.my), (self.WIDTH / 2 - self.mx)) - np.arctan2(self.a2 * np.sin(q2), self.a1 + self.a2 * np.cos(q2))
            angle_q1 = round((0.5 * np.pi + q1) * -180 / np.pi, 1)
            angle_q2 = round((-0.5 * np.pi + q1 + q2) * -180 / np.pi, 1)
            FaceDetectionTower.write_servo(angle_q1, 0)
            FaceDetectionTower.write_servo(angle_q2, 1)       
                    
                    

class AStarSearch:
    def __init__(self, grid, src, dest):
        self.ROW = len(grid)
        self.COL = len(grid[0])
        self.grid = grid
        self.src = src
        self.dest = dest
        self.cell_details = [[self.Cell() for _ in range(self.COL)] for _ in range(self.ROW)]

    class Cell:
        def __init__(self):
            self.parent_i = 0
            self.parent_j = 0
            self.f = float('inf')  # Total cost of the cell (g + h)
            self.g = float('inf')  # Cost from start to this cell
            self.h = 0             # Heuristic cost from this cell to destination

    def is_valid(self, row, col):
        return (row >= 0) and (row < self.ROW) and (col >= 0) and (col < self.COL)

    def is_unblocked(self, row, col):
        return self.grid[row][col] == 0

    def is_destination(self, row, col):
        return row == self.dest[0] and col == self.dest[1]

    def calculate_h_value(self, row, col):
        return ((row - self.dest[0]) ** 2 + (col - self.dest[1]) ** 2) ** 0.5
    
    def trace_path(self):
        path = []
        row = self.dest[0]
        col = self.dest[1]

        while not (self.cell_details[row][col].parent_i == row and self.cell_details[row][col].parent_j == col):
            path.append((row, col))
            temp_row = self.cell_details[row][col].parent_i
            temp_col = self.cell_details[row][col].parent_j
            row = temp_row
            col = temp_col

        path.append((row, col))
        path.reverse()
        return path

    def a_star_search(self):
        if not self.is_valid(self.src[0], self.src[1]) or not self.is_valid(self.dest[0], self.dest[1]):
            print("Source or destination is invalid")
            return

        if not self.is_unblocked(self.src[0], self.src[1]) or not self.is_unblocked(self.dest[0], self.dest[1]):
            print("Source or the destination is blocked")
            return

        if self.is_destination(self.src[0], self.src[1]):
            print("We are already at the destination")
            return


        closed_list = [[False for _ in range(self.COL)] for _ in range(self.ROW)]

        i, j = self.src
        self.cell_details[i][j].f = 0
        self.cell_details[i][j].g = 0
        self.cell_details[i][j].h = 0
        self.cell_details[i][j].parent_i = i
        self.cell_details[i][j].parent_j = j

        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
        found_dest = False

        while len(open_list) > 0:
            p = heapq.heappop(open_list)
            i, j = p[1], p[2]
            closed_list[i][j] = True
                        #↓,     ^,      >,      <,  Diagonal>↓,     Diagonal>^,     Diagonal<↓,     Diagonal<^      
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dir in directions:
                new_i, new_j = i + dir[0], j + dir[1]

                if self.is_valid(new_i, new_j) and self.is_unblocked(new_i, new_j) and not closed_list[new_i][new_j]:
                    if self.is_destination(new_i, new_j):
                        self.cell_details[new_i][new_j].parent_i = i
                        self.cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        self.trace_path()
                        found_dest = True
                        return
                    else:
                        g_new = self.cell_details[i][j].g + 1.0
                        h_new = self.calculate_h_value(new_i, new_j)
                        f_new = g_new + h_new

                        if self.cell_details[new_i][new_j].f == float('inf') or self.cell_details[new_i][new_j].f > f_new:
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            self.cell_details[new_i][new_j].f = f_new
                            self.cell_details[new_i][new_j].g = g_new
                            self.cell_details[new_i][new_j].h = h_new
                            self.cell_details[new_i][new_j].parent_i = i
                            self.cell_details[new_i][new_j].parent_j = j

        if not found_dest:
            print("Failed to find the destination cell")
    
    

if __name__ == "__main__":
    face_detection = FaceDetectionTower()
    face_detection.face_detection_process()
    if face_detection.face_detection_process() == "Walk straight":
        Robot.movement()
