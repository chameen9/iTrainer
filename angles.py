import mediapipe as mp
import pandas as pd
import csv
import streamlit as st
import numpy as np                  # for array manipulation and mathematical operations
import cv2                          # open-source computer vision library
import os                           # for interacting with the operating system
import requests                     # for making HTTP requests
import streamlit_lottie as lottie   # for displaying animation in streamlit
from PIL import Image               # for opening and manipulating image files
import time                         # for getting current time and performing time-related operations
import datetime


# initialize mediaPipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_holistic = mp.solutions.holistic # Holistic model
pose = mp_pose.Pose()
# initialize mediaPipe pose solution

# No of photos in the folder
sequence_length = 70

def mediapipe_detection(image, model):
    #print("entered mediapipe detection")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results
    #print("left mediapipe detection")

def draw_landmarks(image, results):  # draw landmarks and connection
    mp_drawing.draw_landmarks(image, results.pose_landmarks.landmark, mp_holistic.POSE_CONNECTIONS)

def draw_styled_landmarks(image, results):  # draw the dots and connections on images using colors

    mp_drawing.draw_landmarks(image, results.pose_landmarks.landmark, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

def extract_body_keypoints(results):
    kp = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))
    #print (kp)
    #st.write(kp)
    return kp

def load_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return img

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return angle

with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    action = 'drive'

    # Loop through video length, sequence length
    for frame_num in range(sequence_length):  # sequence_length
        photo_path = "dataset\drive\\"

        frame = cv2.imread(photo_path + '\\' + str(frame_num) + ".png")

        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec((255, 0, 0), 1, 1),
                                  mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
                                  )

        final = extract_body_keypoints(results)

        # LEFT side coordinates
        left_hip = final[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_shoulder = final[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = final[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = final[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_knee = final[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = final[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # RIGHT side coordinates
        right_hip = final[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_shoulder = final[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = final[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = final[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_knee = final[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = final[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Left side calculations
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Right side calculations
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        all_angles = np.array([left_knee_angle,
                               left_shoulder_angle,
                               left_elbow_angle,
                               right_knee_angle,
                               right_shoulder_angle,
                               right_elbow_angle,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                               ])

        DATA_PATH = 'output'
        OUT_PHOTO_PATH = ''
        FOLDER_NAME = 'outphoto'
        output_photo_path = os.path.join(OUT_PHOTO_PATH, FOLDER_NAME, action)
        npy_path = os.path.join(DATA_PATH, action, str(frame_num))

        # convert the array
        all_angles = all_angles.reshape(-1, 1)

        # concatenate body coordinates with angles
        #final = np.concatenate((final, all_angles), axis=1)
        final = np.concatenate((all_angles[0], all_angles[1], all_angles[2], all_angles[3], all_angles[4], all_angles[5], np.array(['drive'], dtype=object)))
        #final = all_angles

        print(final)

        with open("drive_shot.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(final)

        # save npy files
        # np.save(npy_path, final)