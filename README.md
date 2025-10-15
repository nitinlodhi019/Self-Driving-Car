# Full Self-Driving System (FSD)

A modular Full Self-Driving (FSD) pipeline designed to simulate a self-driving car's perception stack.

---
# Project Structure
```bash
SELF_DRIVING_CAR/
├── data/
│   └── driving_dataset/ 
│   └── steering_wheel.jpg
├── model_training/
│   ├── lane_segmentation/ 
│   │   └── lane_segmentation.ipynb
│   └── steering_angle/
│       ├── model_training/ 
│       ├── driving_data.py
│       ├── model.py
│       └── train.py
├── saved_models/   
│   ├── lane_segmentation/
│   ├── object_detection/
│   └── steering_angle/
├── src/
│   └── inference/
│       ├── run_lane_segmentation_obj_detection.py
│       └── run_steering_angle_pred.py
├── requirements.txt
└── README.md
```

# How it Works

## Steering Angle Prediction
- Reference - [NVIDIA Paper](https://arxiv.org/pdf/1604.07316)
- Predict the steering angle using dashcam frames and a custom trained deep learning model.
- Findings from the original paper - no maxpooling, no batch normalization, no dropouts, neurons are not in the power of 2.
- What we do differently - Add dropout layers as we have only 25mins of data, while the original paper has more than 70 hours of data.

## Lane Detection
- Reference: [Realtime Lane Detection for Self-Driving Cars Using OpenCV](https://www.labellerr.com/blog/real-time-lane-detection-for-self-driving-cars-using-opencv/#:~:text=Lane%20detection%20in%20self%2Ddriving,autonomous%20driving%20and%20driver%20assistance)
- The lane detection algorithm begins by converting the input image into a grayscale format and applying a Gaussian blur to reduce noise and smooth transitions, which enhances the performance of the subsequent edge detection step. Following this, Canny edge detection is applied to extract the prominent gradients in the image, and a region-of-interest mask is used to isolate the roadway, thereby eliminating extraneous details and focusing on where the lane markings are expected to appear. Once the relevant edges are identified, the Hough Transform method is employed to detect line segments that could represent lane boundaries. These segments are then meticulously filtered and averaged based on predefined slope ranges that help differentiate between left and right lane lines; this involves analyzing the slope and midpoint of each line segment to categorize them appropriately. Finally, to ensure consistent detection over time even in the presence of temporary disruptions or noise, an exponential moving average is applied to smooth the detected lane lines across consecutive frames, resulting in a steady and reliable overlay of lane markers on the original image


## Object Segmentation using YOLOv11
- Reference - [YOLO Documentation](https://docs.ultralytics.com/tasks/segment/)
- The project uses the Ultralytics YOLO model(YOLOv11s-seg) to perform object segmentation on the original image. The segmentation result is blended with the lane detection output using weighted overlay to display both lane markings and detected objects in one unified output.


# Some Images


# Run Locally
- This project requires python 3.9

- Clone the project

    ```bash
    git clone 
    ```

- Create a new Environment

    ```bash
    python -m venv venv
    source sdc/bin/activate
    ```

- Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

- Go to saved_models directory using
    ```bash
    cd saved_models
    ```
    - Download the PreTrained models and model weights from [Drive Link]()
    - Paste the downloaded weights in the corresponding directories.

- Go back to the parent directory using 
    ```bash
    cd ../
    ```
- Go to the inference directory using
    ```bash
    cd src
    cd inference
    ```
- To run steering angle predicition
    ```bash
    python run_steering_angle_pred.py
    ```
- To run lane detection and object detection
    ```bash
    python run_lane_segmentation_obj_detection.py
    ```

# Future Work

- Integrate modules into a unified simulator pipeline
- Add more robust lane segmentation techniques.
- Add automated accleration and braking pipeline based on objects detected.

# References and Resources

- [Steering Angle Predicition](https://arxiv.org/pdf/1604.07316)
- [Lane Segementation Reference](https://www.labellerr.com/blog/real-time-lane-detection-for-self-driving-cars-using-opencv/#:~:text=Lane%20detection%20in%20self%2Ddriving,autonomous%20driving%20and%20driver%20assistance)
- [YOLO](https://docs.ultralytics.com/tasks/segment/)
- [Lane Segementation Using YOLO(Not optimal but still there)](https://universe.roboflow.com/aditya-choudhary-ehv9p/l-s-kvbur)
- [Models Used](https://drive.google.com/drive/u/5/folders/1DoqE9ZykFXq8HYu9mghvkIn8PWeBSBFJ)

