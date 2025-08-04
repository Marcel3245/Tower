<p align="center">
  <img width="617" height="426" alt="image" src="https://github.com/user-attachments/assets/e50f22e0-69be-464d-97fd-18b05be81a58" />
</p>

# Face and Gesture Controlled Camera Tower

## Introduction

Our bodies have various sensors that allow us to gather information from our surroundings and make decisions based on that input. These sensors include sight, smell, touch, taste, and hearing. We have similar technology sensors that translate the real world into electrical signals. For example, a gas sensor mimics the sense of smell, a capacitive touch sensor replicates touch, a microphone corresponds to hearing, and a camera imitates sight, which is the focus of this discussion.

After collecting data from these sensors, we need to process it so the microprocessor can interpret it effectively. In the case of a camera, we can use algorithms and machine learning for tasks such as colour detection, shape recognition, or even facial recognition.

## OpenCV

OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. One of its key applications is face recognition. For this, we use a Cascade classifier along with a pre-designed database. The classifier's primary task is binary classification: determining whether an object is a face (1) or not (0).

Cascade classifiers are trained using positive samples (images containing the object of interest) and negative samples (arbitrary images of the same size that do not contain the object). Specifically, we use Haar cascade object detection. This method starts by isolating the face from the background using certain features, such as identifying the vertical line of the face by recognizing the contrast between "light" skin and the background.

This process is efficient because once the supposed face boundaries are identified, additional methods, like eye detection, are employed. A key characteristic of human faces is that the eye region is typically darker than the nose and cheeks. By applying a white-black-white marker, we obtain new data to compute using weights gathered during training. Another crucial feature is that the eyes are darker than the bridge of the nose.

By proceeding step by step, we save a significant amount of time, which is vital for real-time face recognition. Overall, there are more than 6000 features, each weighted differently. Describing all of them would be impractical, but for a more detailed explanation, you can refer to [this article](https://towardsdatascience.com/face-detection-with-haar-cascades-72710620b8eb).

## Mediapipe

For real-time gesture recognition, I use a library called Mediapipe. Initially, it detects the palm's location using an algorithm similar to the one used in face recognition. This optimization helps streamline the computation of the entire gesture. Once the hand is located and enclosed within a bounding box, the next step is identifying specific points on the hand. Our model is trained to search for 21 landmarks.

Even if the model doesn’t see all the points or the hand is partially visible, it can predict their positions by leveraging its understanding of the hand's consistent internal pose representation. The model has learned how a typical hand appears, with five fingers, four extending almost straight from the palm and one slightly rotated (the thumb). It functions similarly to our brain, which can predict the location of fingers even if they are not fully visible, like when someone shows a thumb-up gesture.

> "After palm detection over the whole image, our subsequent hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates inside the detected hand regions via regression, that is, direct coordinate prediction." (Source: [Mediapipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html))
<p align="center">
  <img  width="616" height="225" alt="image" src="https://github.com/user-attachments/assets/4c5665c7-4e3f-489c-9100-159305d528ce" />
</p>
Now the question arises: how can a computer recognize gestures with just a few coordinates? The answer is simple: mathematics. By knowing the positions of key points such as the wrist (0) and the middle finger MCP (12), we can determine if the hand is facing **"Up"**, **"Down"**, **"Right"**, or **"Left"**.

First, we calculate the absolute value of the x-coordinates and check if this value is close to zero (`|x9-x0|`). This step is crucial for further calculations involving the tangent, because `tan(90°) → ∞` approaches infinity. If the absolute value is not close to 0 (i.e., greater than 0.05), we proceed to calculate the tangent of our absolute values:
`m = (y9 - y0) / (x9 - x0)`

Next, we check if our `m` value is greater than 1. If it is, the hand is facing either **"Up"** or **"Down"**. To determine the exact direction, we compare the y-coordinates of our points. Assuming a Cartesian coordinate system with the origin at the bottom-left corner, we can use the following logic: if `y9 > y0`, the hand is facing **"Up"**; if `y9 < y0`, the hand is facing **"Down"**.

However, if our `m` value is less than or equal to 1 and greater than 0, the hand is pointing either **"Left"** or **"Right"**. To determine this, we compare the x-coordinates: if `x9 > x0`, the hand is pointing **"Right"**; if `x9 < x0`, the hand is pointing **"Left"**.
<p align="center">
  <img width="521" height="395" alt="image" src="https://github.com/user-attachments/assets/9b8dd415-57e1-4bdb-ba5a-8c5e2b2fbf73" />
</p>
Other gestures, like **"thumbs-up"**, **"point"**, **"fist"**, or **"pinch"**, can be recognized by checking which fingers are folded. To determine if a finger is folded, we use landmarks on different parts of the finger: the tip, PIP (proximal interphalangeal joint), and wrist. By measuring the distances between the tip and wrist, and between the PIP and wrist, we can determine the finger's state.

If the distance from the tip to the wrist is greater than the distance from the PIP to the wrist, it means the finger is unfolded (`tip-wrist > pip-wrist` = "finger is unfolded"). Conversely, if the distance from the tip to the wrist is less than the distance from the PIP to the wrist, the finger is folded (`tip-wrist < pip-wrist` = "finger is folded"). This method allows us to understand various gestures. For example, in a **"thumbs-up"** gesture, the pinky, ring, middle, and index fingers are folded, while the thumb is extended. In programming, we can represent this by assigning values to each finger: 0 for folded and 1 for unfolded (in this example, would be thumb-1).
<p align="center">
  <img align="center" width="429" height="245" alt="image" src="https://github.com/user-attachments/assets/b061f9aa-542d-43e0-8a4e-1ce10c4f8544" />
</p>

## Face Position and Camera Adjustment Math

Once we have detected the desired face, we need to determine its position relative to our video display (window) and adjust the camera's direction accordingly. Using OpenCV, we can easily detect the face and draw a rectangle around it. To find the estimated center of the face, we take the width and height of the rectangle, divide each by two, and add the respective x and y coordinates of the rectangle's top-left corner.

Here's how to calculate the center of the face in the video frame:

1.  **Calculate the center of the face:**
    ```
    X_FaceCenter = x + (w/2)
    Y_FaceCenter = y + (h/2)
    ```
2.  **Adjust for the video frame center:**
    ```
    X_Center = Video_WIDTH / 2
    Y_Center = Video_HEIGHT / 2
    ```
3.  **Determine the face's position relatively to the frame center:**
    ```
    X = X_FaceCenter - X_Center
    Y = Y_FaceCenter - Y_Center
    ```
By calculating these values, we can determine how far the face is from the center of the video frame and adjust the camera's position accordingly to keep the face centered.

<p align="center">
  <img width="386" height="270" alt="image" src="https://github.com/user-attachments/assets/6c34b5d2-bb4e-404c-b051-e70c3f689f20" />
</p>

To adjust the camera's position to center the face, we calculate the horizontal and vertical angles based on the vector distance between the face center and our desired center point. Because the camera adjusts by rotating horizontally and vertically, we need to determine how many degrees it should move in each direction.

1.  **Calculate the vector distance:**
    Let `△X` be the horizontal distance between the face center and the desired center point.
    Let `△Y` be the vertical distance between the face center and the desired center point.

2.  **Convert distance from pixels to real-world units (cm):**
    Using optics principles, we apply the lens equation modification: `x = (h/H) * y`
    *   `x` is the distance in centimeters from the camera to the face (desired distance).
    *   `W` is the actual width of the face (approximately 15 cm for an average face).
    *   `w` is the width of the face on the video (in pixels).
    *   `y` is the focal length of the camera, a fixed value.

    Therefore: `x = 15*Ww`

3.  **Calculate the angle to move the camera:**
    Convert the horizontal and vertical distances to angles using the inverse sine function:
    ```
    θ_horizontal = arcsin(X/x)
    θ_vertical = arcsin(Y/z)
    ```
    These angles determine how much the camera needs to rotate horizontally and vertically to position the face at the desired center point in the real world, considering the optics and physical distances involved.

<p align="center">
  <img width="580" height="263" alt="image" src="https://github.com/user-attachments/assets/63222740-f714-4044-b97b-2ad14b0bcff6" />
</p>

## Multiprocessing for Smooth Camera Movement

To achieve smoother camera movement, I implemented multiprocessing, which allows the system to handle multiple tasks simultaneously. With multiprocessing, the CPU can execute several processes concurrently, enabling simultaneous control of two servos. Without multiprocessing, the camera would move sequentially: first horizontally and then vertically. This sequential movement isn't efficient and can result in jerky, stair-like motions.

By using multiprocessing, the camera can move both servos simultaneously. This allows the system to choose the fastest path, typically diagonal movement, which results in smoother transitions between horizontal and vertical adjustments.

## Hardware Setup

### Microprocessor
For this project, I chose to use the Raspberry Pi 4B with 4 GB of RAM. It features a Quad-core Cortex-A72 (ARM v8) 64-bit SoC running at 1.8 GHz and includes a 40-pin GPIO header. This setup provides ample computational power for real-time face and gesture recognition, as well as streaming live video output.

### Camera
I opted for the IMX519 camera module, which offers a resolution of 16 MP with active pixels of 4656x3496 and a pixel size of 1.22μm×1.22μm. One of its notable advantages is autofocus capability, which will be beneficial for future stages of the project. The camera can achieve high frame rates, up to 120 fps at 1280x720 resolution (although our usage will likely be lower due to the computational demands of face and gesture recognition). Details on connecting this camera to the Raspberry Pi can be found [here](https://www.arducam.com/docs/cameras-for-raspberry-pi/native-raspberry-pi-cameras/16mp-autofocus-camera-for-raspberry-pi/).

### Servo control
To control the servos, I selected the PCA9685 16-channel servo driver. While 16 channels may initially seem excessive for controlling two servos, this setup allows for scalability in future expansions of the project, such as integrating with a full robot dog. The additional channels will be useful for managing multiple servos and ensuring a streamlined control system.

### Servo
For servo actuators, I opted for the Miuzei Micro Servo Metal Gear 90 9G. These servos are known for their reliability, and the use of metal gears minimizes the likelihood of steps being skipped during rotation. This ensures precise control over the camera's positioning.

<p align="center">
  <img width="432" height="202" alt="image" src="https://github.com/user-attachments/assets/c8a0029d-a5bc-41e5-8cf4-c0a99137c326" />
</p>

## Mechanical Assembly

To create the parts, I decided to print them using my Ender 3, 3D printer. Printing them will ensure easy manufacturing and are lightweight. You can find the printing files and details such as temperature, infill, and support settings [here](<link-to-your-stl-files>).

<p align="center">
  <img width="525" height="317" alt="image" src="https://github.com/user-attachments/assets/0e54fb30-fdab-4c19-8efa-86aeb98ccb66" />
</p>

1.  **Servo Installation:**
    Place the servo inside the body, ensuring the cables are routed through prepared holes.
    
<p align="center">    
  <img width="368" height="337" alt="image" src="https://github.com/user-attachments/assets/a38fd375-3376-4055-8bfd-af0147b3c047" />
</p>


3.  **Attaching Tower Feet:**
    From the bottom, screw the plastic servo horn into the "tower-feet" component.

<p align="center">    
  <img width="444" height="319" alt="image" src="https://github.com/user-attachments/assets/5cff3f18-1d98-4f16-ad44-410af9b3471a" />
</p>

4.  **Assembling the Tower Head:**
    Combine the tower head bottom and top parts, ensuring the camera and servo are securely inside and connected. For easier setup, secure it from the top using M3x5mm screws (note: the bottom part houses the camera cable output).

<p align="center">    
  <img width="475" height="329" alt="image" src="https://github.com/user-attachments/assets/7a836be0-c4c3-4b15-8165-d9fd1a769aa6" />
</p>

5.  **Servo Calibration:**
    After everything is connected, calibrate all the servos. You can do this manually or use the `servo_calibration.py` script from my repository. Ensure the camera faces the correct direction at 0 degrees (starting position shown in the first picture of this chapter).

6.  **Mounting the Head to the Body:**
    Insert another plastic servo horn into the top of the "body," opposite the rounded hole. Carefully mount the head into the body, applying gentle pressure. If necessary, use glue to secure any broken parts.

<p align="center">    
  <img width="398" height="211" alt="image" src="https://github.com/user-attachments/assets/e755ae3c-1495-4323-a408-efe2e9756255" />
</p>

## Final Result

<p align="center">    
  <img width="243" height="301" alt="image" src="https://github.com/user-attachments/assets/7a652d4e-4422-41fe-934b-9a502ac0495f" />
</p>

## Software Setup

Because our Python code will run on the Raspberry Pi, which works on Raspbian (based on the Debian GNU/Linux - C/C++), I have a few recommendations on how to start with installation. To make it possible to run Python code, we have to create a Python-based virtual environment, but before that, there are a few things we have to do first. In the command line interface of Raspberry Pi (It will work only with the screen and HDMI cable if you don’t have it. You have to slightly change the code and get rid of this line: `cv2.imshow('Video', frame)`):

```bash
# Update and upgrade your system
sudo apt update && sudo apt full-upgrade

# Install IMX519 dependencies from the official Arducam documentation
# Follow instructions from: https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/16MP-IMX519/
# After installation, reboot
sudo reboot

# Check if the camera works
libcamera-still -t 5000 -n -o test.jpg

# Install required packages
sudo apt install -y python3-picamera2
sudo apt-get install cmake && sudo apt install git && sudo pip3 install virtualenv

# Clone the repository and navigate into it
git clone https://github.com/Marcel3245/Tower.git && cd Tower

# Create and activate a Python virtual environment
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Prepare for pictures and train the face recognition model
cd training
python training_picture.py
python face_training.py

# Assuming all hardware is connected, calibrate the servos
cd ~/Tower
python servo_calibration.py

# Run the main application
python towerV3.py
```


<p align="center">    
  <img width="512" height="206" alt="image" src="https://github.com/user-attachments/assets/3b08e249-efcf-4f48-94e1-4e5850043029" />
</p>

Follow these steps to set up and run your project on the Raspberry Pi, enabling face and gesture recognition capabilities using the IMX519 camera and servo-controlled mechanism. Good luck!
