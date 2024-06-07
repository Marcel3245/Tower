import board
import time
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

# Set up PCA9685
i2c = board.I2C()
pca = PCA9685(i2c)
pca.frequency = 50

# Custom pulse widths
servo_min = 500  # 500us corresponds to 0 degrees
servo_max = 2500 # 2500us corresponds to 180 degrees

# Configure the servo with custom pulse widths
servo_x = servo.Servo(pca.channels[0], actuation_range=180, min_pulse=servo_min, max_pulse=servo_max)
servo_y = servo.Servo(pca.channels[1], actuation_range=180, min_pulse=servo_min, max_pulse=servo_max)

# Test servo movement
servo_x.angle = 120
time.sleep(1)
# servo_x.angle = 130
# time.sleep(1)
servo_y.angle = 20
time.sleep(1)
# servo_y.angle = 90
# time.sleep(1)

currentPosition_x = 90.00
currentPosition_y = 180.00
servo_x = servo.Servo(pca.channels[14], actuation_range=180, min_pulse=servo_min, max_pulse=servo_max)
servo_y = servo.Servo(pca.channels[15], actuation_range=180, min_pulse=servo_min, max_pulse=servo_max)
servo_x.angle = currentPosition_x
servo_y.angle = currentPosition_y
print("Starting position: theta_x = 90.00 degrees, theta_y = 180.00 degrees")
time.sleep(.1)
