import mediapipe as mp
import cv2
import numpy as np
from math import dist

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class gestureRecognition(object):
    def __init__(self, img):
        self.img = img
    
    @staticmethod
    def orientation(coordinate_landmark_0, coordinate_landmark_9): 
        x0 = coordinate_landmark_0[0]
        y0 = coordinate_landmark_0[1]
        
        x9 = coordinate_landmark_9[0]
        y9 = coordinate_landmark_9[1]
        
        if abs(x9 - x0) < 0.05:      #since tan(0) --> ∞
            m = 1000000000
        else:
            m = abs((y9 - y0)/(x9 - x0))       
            
        if m>=0 and m<=1:
            if x9 > x0:
                return "Right"
            else:
                return "Left"
        if m>1:
            if y9 < y0:       #since, y decreases upwards
                return "Up"
            else:
                return "Down"


    def detectGesture(self):
        # Perform hand gesture recognition
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(self.img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarksCords = np.empty((0, 2))
            for index in range(0, 21):
                x = float(str(results.multi_hand_landmarks[-1].landmark[int(index)]).split('\n')[0].split(" ")[1])
                y = float(str(results.multi_hand_landmarks[-1].landmark[int(index)]).split('\n')[1].split(" ")[1])
                cords = np.array([x, y])
                landmarksCords = np.r_[landmarksCords, [cords]]

            p0x = landmarksCords[0][0] # coordinates of landmark 0 "Palm"
            p0y = landmarksCords[0][1]

            p5x = landmarksCords[5][0] # coordinates of landmark 5
            p5y = landmarksCords[5][1]

            p3x = landmarksCords[3][0] # coordinates of mid thumb finger
            p3y = landmarksCords[3][1]
            d03 = dist([p5x, p5y], [p3x, p3y])

            p4x = landmarksCords[4][0] # coordinates of tip thumb 
            p4y = landmarksCords[4][1]
            d04 = dist([p5x, p5y], [p4x, p4y])

            p7x = landmarksCords[7][0] # coordinates of mid index finger
            p7y = landmarksCords[7][1]
            d07 = dist([p0x, p0y], [p7x, p7y])

            p8x = landmarksCords[8][0] # coordinates of tip index
            p8y = landmarksCords[8][1]
            d08 = dist([p0x, p0y], [p8x, p8y])

            p11x = landmarksCords[11][0] # coordinates of mid middle finger
            p11y = landmarksCords[11][1]
            d011 = dist([p0x, p0y], [p11x, p11y])

            p12x = landmarksCords[12][0] # coordinates of tip middle finger
            p12y = landmarksCords[12][1]
            d012 = dist([p0x, p0y], [p12x, p12y])

            p15x = landmarksCords[15][0] # coordinates of mid ring finger
            p15y = landmarksCords[15][1]
            d015 = dist([p0x, p0y], [p15x, p15y])

            p16x = landmarksCords[16][0] # coordinates of tip ring finger
            p16y = landmarksCords[16][1]
            d016 = dist([p0x, p0y], [p16x, p16y])

            p19x = landmarksCords[19][0] # coordinates of mid pinky finger
            p19y = landmarksCords[19][1]
            d019 = dist([p0x, p0y], [p19x, p19y])

            p20x = landmarksCords[20][0] # coordinates of tip pinky finger
            p20y = landmarksCords[20][1]
            d020 = dist([p0x, p0y], [p20x, p20y])

            close = []
            if d03>d04: 
                close.append(1)
            if d07>d08:
                close.append(2)
            if d011>d012:
                close.append(3)
            if d015>d016:
                close.append(4)
            if d019>d020:
                close.append(5)

            if self.orientation((p0x, p0y), (landmarksCords[9][0], landmarksCords[9][1])) is not None:
                orientation_result = self.orientation((p0x, p0y), (landmarksCords[9][0], landmarksCords[9][1]))
            
            # Check left or right side "pointing finger"
            if close == [3, 4, 5] or close == [1, 3, 4, 5]:
                if orientation_result == 'Left':
                    return 'Check left'
                if orientation_result == 'Right':
                    return 'Check right'
                
            # Walk, stop and turn left or right "on open hand"
            if close == []:
                if orientation_result == 'Right':
                    return "Turn right"
                if orientation_result == 'Left':
                    return "Turn left"
                if orientation_result == 'Up':
                    return "Stop"
                if orientation_result == 'Down':
                    return "Walk straight"

    
            

            