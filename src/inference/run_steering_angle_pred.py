import sys
import os
import cv2
from subprocess import call
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from model_training.steering_angle import model

from ultralytics import YOLO
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

class SteeringAnglePredictor:
    def __init__(self, model_path):
        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, model_path)
        self.smoothed_angle = 0
        self.model = model
        
    def predict_angle(self, image):
        output = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})
        radians = output[0][0]
        degrees = radians * (180.0 / 3.14159265)
        return degrees
    
    def smooth_angle(self, predicted_angle):
        if self.smoothed_angle == 0:
            self.smoothed_angle = predicted_angle
        else:
            difference = predicted_angle - self.smoothed_angle
            if difference != 0:  # Ensure no division by zero
                abs_difference = abs(difference)
                scaled_difference = pow(abs_difference, 2.0 / 3.0)
                update = 0.2 * scaled_difference * (difference / abs_difference)
                self.smoothed_angle += update
        
        return self.smoothed_angle  # Ensure a valid return value

        
    def close(self):
        self.session.close()

class DrivingSimulator:
    def __init__(self, predictor, data_dir, steering_image_path, is_windows=False):
        self.predictor = predictor
        self.data_dir = data_dir
        self.is_windows = is_windows

        # Load steering wheel image
        self.steering_image = cv2.imread(steering_image_path, cv2.IMREAD_UNCHANGED)

        if self.steering_image is None:
            raise ValueError(f"Could not load steering wheel image from {steering_image_path}")

        # Convert to RGBA if necessary
        if self.steering_image.shape[-1] == 3:  # If no alpha channel, add one
            self.steering_image = cv2.cvtColor(self.steering_image, cv2.COLOR_BGR2BGRA)

        # Ensure it's square to prevent distortion
        sw_h, sw_w = self.steering_image.shape[:2]
        size = max(sw_h, sw_w)
        self.steering_image = cv2.resize(self.steering_image, (size, size))

    def start_simulation(self):
        i = 0
        while cv2.waitKey(10) != ord('q'):
            full_image = cv2.imread(os.path.join(self.data_dir, f"{i}.jpg"))
            if full_image is None:
                print(f"Image {i}.jpg not found, stopping simulation.")
                break

            resized_image = cv2.resize(full_image[-150:], (200, 66)) / 255.0

            predicted_angle = self.predictor.predict_angle(resized_image)
            smoothed_angle = self.predictor.smooth_angle(predicted_angle)

            if not self.is_windows:
                call("clear")
            print(f"Predicted steering angle: {predicted_angle:.2f} degrees")

            self.display_frames(full_image, smoothed_angle)
            i += 1

        cv2.destroyAllWindows()

    def display_frames(self, full_image, smoothed_angle):
        # Display the main driving frame
        cv2.imshow("frame", full_image)

        # Get steering wheel size
        sw_h, sw_w, _ = self.steering_image.shape

        # Create a black background for the steering wheel window
        steering_display = np.zeros((sw_h, sw_w, 4), dtype=np.uint8)

        # Rotate the steering wheel
        rotation_matrix = cv2.getRotationMatrix2D((sw_w // 2, sw_h // 2), -smoothed_angle, 1)
        rotated_steering = cv2.warpAffine(self.steering_image, rotation_matrix, (sw_w, sw_h))

        # Apply transparency (keep only the non-black pixels)
        alpha_channel = rotated_steering[:, :, 3]  # Extract alpha channel
        mask = alpha_channel > 0  # Mask where pixels are non-transparent

        #Apply the mask to place the rotated steering wheel on a black background
        steering_display[mask] = rotated_steering[mask]

        #Convert to BGR for display
        steering_display_bgr = cv2.cvtColor(steering_display, cv2.COLOR_BGRA2BGR)

        #Overlay text with predicted angle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            steering_display_bgr, 
            f"{smoothed_angle:.2f} deg",  # Use "deg" instead of "°"
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, cv2.LINE_AA
        )

        # Show the steering wheel separately
        cv2.imshow("steering wheel", steering_display_bgr)

    
if __name__ == "__main__":
    model_path = '../../saved_models/steering_angle/90epoch/model.ckpt'
    data_dir = '../../data/driving_dataset'
    steering_wheel_image_path = '../../data/steering_wheel.jpg'
    
    # IF RUNNING ON WINDOWS
    is_windows = os.name == 'nt' # FALSE OTHERWISE
    
    predictor = SteeringAnglePredictor(model_path)
    simulator = DrivingSimulator(predictor, data_dir, steering_wheel_image_path, is_windows)
    
    try:
        simulator.start_simulation()
    finally:
        predictor.close()