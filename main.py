import mediapipe as mp
import streamlit as st
import sqlite3
import pickle                       # For serializing and deserializing objects
import pandas as pd                 # For data analysis and manipulation
import numpy as np                  # For array manipulation and mathematical operations
import cv2                          # For computer vision tasks
import os                           # For interacting with the operating system
import requests                     # For making HTTP requests
import streamlit_lottie as lottie   # For displaying animation in streamlit
from PIL import Image               # For opening and manipulating image files
from datetime import datetime

################################################################################################
######################################## WEB CONFIG ############################################
################################################################################################
xIcon = Image.open("images/Xicon.ico")

#Set Page config
st.set_page_config(page_title="i Trainer", page_icon = xIcon, layout="wide")

#check lottieurl
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#Local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html = True)

bt_shot = None
bw_pose = None

local_css("style/style.css")

#Lottie SRC
lottie_gradient = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_anre6w2q.json")
logo = Image.open("images/Logo.png")

################################################################################################
######################################## MODEL CONFIG ##########################################
################################################################################################

# initialize mediaPipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_holistic = mp.solutions.holistic # Holistic model
pose = mp_pose.Pose()
# initialize mediaPipe pose solution

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

def load_image(image_file):
    img = Image.open(image_file)
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

connection = sqlite3.connect('data.db')
cursor = connection.cursor()

table_create_command = """CREATE TABLE IF NOT EXISTS
    details(id INTEGER PRIMARY KEY AUTOINCREMENT, acc_value FLOAT, status TEXT, date DATETIME)
"""

def cal_left_knee_angle_acc(left_knee_angle,shot_type):
    left_knee_angle_acc = 0.00
    if (shot_type == 'Drive'):
        if (left_knee_angle >= 180):
            left_knee_angle_acc = 1.00
        elif (left_knee_angle < 90.00):
            left_knee_angle_acc = 1.00
        elif (left_knee_angle >= 90 and left_knee_angle < 100):
            left_knee_angle_acc = 3.00
        elif (left_knee_angle >= 100 and left_knee_angle < 110):
            left_knee_angle_acc = 5.00
        elif (left_knee_angle >= 110 and left_knee_angle < 120):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 120 and left_knee_angle < 130):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 130 and left_knee_angle < 140):
            left_knee_angle_acc = 9.00
        elif (left_knee_angle >= 140 and left_knee_angle < 150):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 150 and left_knee_angle < 160):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 160 and left_knee_angle < 170):
            left_knee_angle_acc = 5.00
        elif (left_knee_angle >= 170 and left_knee_angle < 180):
            left_knee_angle_acc = 3.00
        elif (left_knee_angle == 135):
            left_knee_angle_acc = 10.00

        return left_knee_angle_acc

    elif(shot_type == 'Pullshot'):
        if (left_knee_angle >= 180):
            left_knee_angle_acc = 1.00
        elif (left_knee_angle < 120.00):
            left_knee_angle_acc = 1.00
        elif (left_knee_angle >= 120 and left_knee_angle < 130):
            left_knee_angle_acc = 3.00
        elif (left_knee_angle >= 130 and left_knee_angle < 140):
            left_knee_angle_acc = 5.00
        elif (left_knee_angle >= 140 and left_knee_angle < 150):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 150 and left_knee_angle < 160):
            left_knee_angle_acc = 9.00
        elif (left_knee_angle >= 160 and left_knee_angle < 170):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 170 and left_knee_angle < 180):
            left_knee_angle_acc = 5.00

        elif (left_knee_angle == 155):
            left_knee_angle_acc = 10.00

        return left_knee_angle_acc

    elif (shot_type == 'Sweep'):
        if (left_knee_angle >= 180):
            left_knee_angle_acc = 1.00
        elif (left_knee_angle < 110.00):
            left_knee_angle_acc = 1.00
        elif (left_knee_angle >= 110 and left_knee_angle < 120):
            left_knee_angle_acc = 3.00
        elif (left_knee_angle >= 120 and left_knee_angle < 130):
            left_knee_angle_acc = 5.00
        elif (left_knee_angle >= 130 and left_knee_angle < 140):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 140 and left_knee_angle < 150):
            left_knee_angle_acc = 9.00
        elif (left_knee_angle >= 150 and left_knee_angle < 160):
            left_knee_angle_acc = 7.00
        elif (left_knee_angle >= 160 and left_knee_angle < 170):
            left_knee_angle_acc = 5.00
        elif (left_knee_angle >= 170 and left_knee_angle < 180):
            left_knee_angle_acc = 3.00

        elif (left_knee_angle == 145):
            left_knee_angle_acc = 10.00
        return left_knee_angle_acc

    else:
        return left_knee_angle_acc

def cal_left_shoulder_angle_acc(left_shoulder_angle,shot_type):
    left_shoulder_angle_acc = 0.00

    if(shot_type == 'Drive'):
        if (left_shoulder_angle >= 120):
            left_shoulder_angle_acc = 1.00
        elif (left_shoulder_angle < 90.00):
            left_shoulder_angle_acc = 1.00
        elif (left_shoulder_angle >= 90 and left_shoulder_angle < 100):
            left_shoulder_angle_acc = 5.00
        elif (left_shoulder_angle >= 100 and left_shoulder_angle < 110):
            left_shoulder_angle_acc = 9.00
        elif (left_shoulder_angle >= 110 and left_shoulder_angle < 120):
            left_shoulder_angle_acc = 5.00
        elif (left_shoulder_angle == 95):
            left_shoulder_angle_acc = 10.00

        return left_shoulder_angle_acc

    elif (shot_type == 'Pullshot'):
        if (left_shoulder_angle >= 90):
            left_shoulder_angle_acc = 1.00
        elif (left_shoulder_angle < 40.00):
            left_shoulder_angle_acc = 1.00
        elif (left_shoulder_angle >= 40 and left_shoulder_angle < 50):
            left_shoulder_angle_acc = 3.00
        elif (left_shoulder_angle >= 50 and left_shoulder_angle < 60):
            left_shoulder_angle_acc = 5.00
        elif (left_shoulder_angle >= 60 and left_shoulder_angle < 70):
            left_shoulder_angle_acc = 9.00
        elif (left_shoulder_angle >= 70 and left_shoulder_angle < 80):
            left_shoulder_angle_acc = 5.00
        elif (left_shoulder_angle >= 80 and left_shoulder_angle < 90):
            left_shoulder_angle_acc = 3.00
        elif (left_shoulder_angle == 65):
            left_shoulder_angle_acc = 10.00

        return left_shoulder_angle_acc

    elif (shot_type == 'Sweep'):
        if (left_shoulder_angle >= 50):
            left_shoulder_angle_acc = 1.00
        elif (left_shoulder_angle < 0.00):
            left_shoulder_angle_acc = 1.00
        elif (left_shoulder_angle >= 0 and left_shoulder_angle < 10):
            left_shoulder_angle_acc = 5.00
        elif (left_shoulder_angle >= 10 and left_shoulder_angle < 20):
            left_shoulder_angle_acc = 7.00
        elif (left_shoulder_angle >= 20 and left_shoulder_angle < 30):
            left_shoulder_angle_acc = 9.00
        elif (left_shoulder_angle >= 30 and left_shoulder_angle < 40):
            left_shoulder_angle_acc = 7.00
        elif (left_shoulder_angle >= 40 and left_shoulder_angle < 50):
            left_shoulder_angle_acc = 5.00
        elif (left_shoulder_angle == 25):
            left_shoulder_angle_acc = 10.00

        return left_shoulder_angle_acc

    else:
        return left_shoulder_angle_acc

def cal_left_elbow_angle_acc(left_elbow_angle,shot_type):
    left_elbow_angle_acc = 0.00

    if(shot_type == 'Drive'):
        if (left_elbow_angle >= 150):
            left_elbow_angle_acc = 1.00
        elif (left_elbow_angle < 90.00):
            left_elbow_angle_acc = 1.00
        elif (left_elbow_angle >= 90 and left_elbow_angle < 100):
            left_elbow_angle_acc = 5.00
        elif (left_elbow_angle >= 110 and left_elbow_angle < 120):
            left_elbow_angle_acc = 7.00
        elif (left_elbow_angle >= 120 and left_elbow_angle < 130):
            left_elbow_angle_acc = 9.00
        elif (left_elbow_angle >= 130 and left_elbow_angle < 140):
            left_elbow_angle_acc = 7.00
        elif (left_elbow_angle >= 140 and left_elbow_angle < 150):
            left_elbow_angle_acc = 5.00
        elif (left_elbow_angle == 120):
            left_elbow_angle_acc = 10.00

        return left_elbow_angle_acc

    elif (shot_type == 'Pullshot'):
        if (left_elbow_angle >= 150):
            left_elbow_angle_acc = 1.00
        elif (left_elbow_angle < 100.00):
            left_elbow_angle_acc = 1.00
        elif (left_elbow_angle >= 100 and left_elbow_angle < 110):
            left_elbow_angle_acc = 5.00
        elif (left_elbow_angle >= 110 and left_elbow_angle < 120):
            left_elbow_angle_acc = 7.00
        elif (left_elbow_angle >= 120 and left_elbow_angle < 130):
            left_elbow_angle_acc = 9.00
        elif (left_elbow_angle >= 130 and left_elbow_angle < 140):
            left_elbow_angle_acc = 7.00
        elif (left_elbow_angle >= 140 and left_elbow_angle < 150):
            left_elbow_angle_acc = 5.00
        elif (left_elbow_angle == 125):
            left_elbow_angle_acc = 10.00

        return left_elbow_angle_acc

    elif (shot_type == 'Sweep'):
        if (left_elbow_angle >= 160):
            left_elbow_angle_acc = 1.00
        elif (left_elbow_angle < 110.00):
            left_elbow_angle_acc = 1.00
        elif (left_elbow_angle >= 110 and left_elbow_angle < 120):
            left_elbow_angle_acc = 5.00
        elif (left_elbow_angle >= 120 and left_elbow_angle < 130):
            left_elbow_angle_acc = 7.00
        elif (left_elbow_angle >= 130 and left_elbow_angle < 140):
            left_elbow_angle_acc = 9.00
        elif (left_elbow_angle >= 140 and left_elbow_angle < 150):
            left_elbow_angle_acc = 7.00
        elif (left_elbow_angle >= 150 and left_elbow_angle < 160):
            left_elbow_angle_acc = 5.00
        elif (left_elbow_angle == 135):
            left_elbow_angle_acc = 10.00

        return left_elbow_angle_acc

    else:
        return left_elbow_angle_acc

def cal_right_knee_angle_acc(right_knee_angle, shot_type):
    right_knee_angle_acc = 0.00

    if(shot_type == 'Drive'):
        if (right_knee_angle >= 160):
            right_knee_angle_acc = 1.00
        elif (right_knee_angle < 130.00):
            right_knee_angle_acc = 1.00
        elif (right_knee_angle >= 130 and right_knee_angle < 140):
            right_knee_angle_acc = 5.00
        elif (right_knee_angle >= 140 and right_knee_angle < 150):
            right_knee_angle_acc = 9.00
        elif (right_knee_angle >= 150 and right_knee_angle < 160):
            right_knee_angle_acc = 5.00
        elif (right_knee_angle == 135):
            right_knee_angle_acc = 10.00

        return right_knee_angle_acc

    elif (shot_type == 'Pullshot'):
        if (right_knee_angle >= 180):
            right_knee_angle_acc = 1.00
        elif (right_knee_angle < 130.00):
            right_knee_angle_acc = 1.00
        elif (right_knee_angle >= 130 and right_knee_angle < 140):
            right_knee_angle_acc = 5.00
        elif (right_knee_angle >= 140 and right_knee_angle < 150):
            right_knee_angle_acc = 7.00
        elif (right_knee_angle >= 150 and right_knee_angle < 160):
            right_knee_angle_acc = 9.00
        elif (right_knee_angle >= 160 and right_knee_angle < 170):
            right_knee_angle_acc = 7.00
        elif (right_knee_angle >= 170 and right_knee_angle < 180):
            right_knee_angle_acc = 5.00
        elif (right_knee_angle == 155):
            right_knee_angle_acc = 10.00

        return right_knee_angle_acc

    elif (shot_type == 'Sweep'):
        if (right_knee_angle >= 170):
            right_knee_angle_acc = 1.00
        elif (right_knee_angle < 120.00):
            right_knee_angle_acc = 1.00
        elif (right_knee_angle >= 120 and right_knee_angle < 130):
            right_knee_angle_acc = 5.00
        elif (right_knee_angle >= 130 and right_knee_angle < 140):
            right_knee_angle_acc = 7.00
        elif (right_knee_angle >= 140 and right_knee_angle < 150):
            right_knee_angle_acc = 9.00
        elif (right_knee_angle >= 150 and right_knee_angle < 160):
            right_knee_angle_acc = 7.00
        elif (right_knee_angle >= 160 and right_knee_angle < 170):
            right_knee_angle_acc = 5.00
        elif (right_knee_angle == 145):
            right_knee_angle_acc = 10.00

        return right_knee_angle_acc

    else:
        return right_knee_angle_acc

def cal_right_shoulder_angle_acc(right_shoulder_angle, shot_type):
    right_shoulder_angle_acc = 0.00

    if(shot_type == 'Drive'):
        if (right_shoulder_angle >= 110):
            right_shoulder_angle_acc = 1.00
        elif (right_shoulder_angle < 50.00):
            right_shoulder_angle_acc = 1.00
        elif (right_shoulder_angle >= 50 and right_shoulder_angle < 60):
            right_shoulder_angle_acc = 5.00
        elif (right_shoulder_angle >= 60 and right_shoulder_angle < 70):
            right_shoulder_angle_acc = 7.00
        elif (right_shoulder_angle >= 70 and right_shoulder_angle < 80):
            right_shoulder_angle_acc = 8.00
        elif (right_shoulder_angle >= 80 and right_shoulder_angle < 90):
            right_shoulder_angle_acc = 10.00
        elif (right_shoulder_angle >= 90 and right_shoulder_angle < 100):
            right_shoulder_angle_acc = 8.00
        elif (right_shoulder_angle >= 100 and right_shoulder_angle < 110):
            right_shoulder_angle_acc = 6.00
        elif (right_shoulder_angle == 85):
            right_shoulder_angle_acc = 10.00

        return right_shoulder_angle_acc

    elif (shot_type == 'Pullshot'):
        if (right_shoulder_angle >= 80):
            right_shoulder_angle_acc = 1.00
        elif (right_shoulder_angle < 30.00):
            right_shoulder_angle_acc = 1.00
        elif (right_shoulder_angle >= 30 and right_shoulder_angle < 40):
            right_shoulder_angle_acc = 5.00
        elif (right_shoulder_angle >= 40 and right_shoulder_angle < 50):
            right_shoulder_angle_acc = 7.00
        elif (right_shoulder_angle >= 50 and right_shoulder_angle < 60):
            right_shoulder_angle_acc = 9.00
        elif (right_shoulder_angle >= 60 and right_shoulder_angle < 70):
            right_shoulder_angle_acc = 7.00
        elif (right_shoulder_angle >= 70 and right_shoulder_angle < 80):
            right_shoulder_angle_acc = 5.00
        elif (right_shoulder_angle == 55):
            right_shoulder_angle_acc = 10.00

        return right_shoulder_angle_acc

    elif (shot_type == 'Sweep'):
        if (right_shoulder_angle >= 60):
            right_shoulder_angle_acc = 1.00
        elif (right_shoulder_angle < 0.00):
            right_shoulder_angle_acc = 1.00
        elif (right_shoulder_angle >= 00 and right_shoulder_angle < 10):
            right_shoulder_angle_acc = 5.00
        elif (right_shoulder_angle >= 10 and right_shoulder_angle < 20):
            right_shoulder_angle_acc = 7.00
        elif (right_shoulder_angle >= 20 and right_shoulder_angle < 30):
            right_shoulder_angle_acc = 9.00
        elif (right_shoulder_angle >= 30 and right_shoulder_angle < 40):
            right_shoulder_angle_acc = 7.00
        elif (right_shoulder_angle >= 40 and right_shoulder_angle < 50):
            right_shoulder_angle_acc = 5.00
        elif (right_shoulder_angle == 25):
            right_shoulder_angle_acc = 10.00

        return right_shoulder_angle_acc

    else:
        return right_shoulder_angle_acc

def cal_right_elbow_angle_acc(right_elbow_angle, shot_type):
    right_elbow_angle_acc = 0.00

    if(shot_type == 'Drive'):
        if (right_elbow_angle >= 160):
            right_elbow_angle_acc = 1.00
        elif (right_elbow_angle < 90.00):
            right_elbow_angle_acc = 1.00
        elif (right_elbow_angle >= 90 and right_elbow_angle < 100):
            right_elbow_angle_acc = 5.00
        elif (right_elbow_angle >= 110 and right_elbow_angle < 120):
            right_elbow_angle_acc = 7.00
        elif (right_elbow_angle >= 120 and right_elbow_angle < 130):
            right_elbow_angle_acc = 9.00
        elif (right_elbow_angle >= 130 and right_elbow_angle < 140):
            right_elbow_angle_acc = 7.00
        elif (right_elbow_angle >= 140 and right_elbow_angle < 150):
            right_elbow_angle_acc = 7.00
        elif (right_elbow_angle >= 150 and right_elbow_angle < 160):
            right_elbow_angle_acc = 5.00
        elif (right_elbow_angle == 125):
            right_elbow_angle_acc = 10.00

        return right_elbow_angle_acc

    elif (shot_type == 'Pullshot'):
        if (right_elbow_angle >= 140):
            right_elbow_angle_acc = 1.00
        elif (right_elbow_angle < 90.00):
            right_elbow_angle_acc = 1.00
        elif (right_elbow_angle >= 90 and right_elbow_angle < 100):
            right_elbow_angle_acc = 5.00
        elif (right_elbow_angle >= 90 and right_elbow_angle < 100):
            right_elbow_angle_acc = 7.00
        elif (right_elbow_angle >= 110 and right_elbow_angle < 120):
            right_elbow_angle_acc = 9.00
        elif (right_elbow_angle >= 120 and right_elbow_angle < 130):
            right_elbow_angle_acc = 7.00
        elif (right_elbow_angle >= 130 and right_elbow_angle < 140):
            right_elbow_angle_acc = 5.00
        elif (right_elbow_angle == 115):
            right_elbow_angle_acc = 10.00

        return right_elbow_angle_acc

    elif (shot_type == 'Sweep'):
        if (right_elbow_angle >= 170):
            right_elbow_angle_acc = 1.00
        elif (right_elbow_angle < 90.00):
            right_elbow_angle_acc = 1.00
        elif (right_elbow_angle >= 90 and right_elbow_angle < 100):
            right_elbow_angle_acc = 3.00
        elif (right_elbow_angle >= 110 and right_elbow_angle < 120):
            right_elbow_angle_acc = 5.00
        elif (right_elbow_angle >= 120 and right_elbow_angle < 130):
            right_elbow_angle_acc = 7.00
        elif (right_elbow_angle >= 130 and right_elbow_angle < 140):
            right_elbow_angle_acc = 9.00
        elif (right_elbow_angle >= 140 and right_elbow_angle < 150):
            right_elbow_angle_acc = 7.00
        elif (right_elbow_angle >= 150 and right_elbow_angle < 160):
            right_elbow_angle_acc = 5.00
        elif (right_elbow_angle >= 160 and right_elbow_angle < 170):
            right_elbow_angle_acc = 3.00
        elif (right_elbow_angle == 135):
            right_elbow_angle_acc = 10.00

        return right_elbow_angle_acc

    else:
        return right_elbow_angle_acc

def create_frames(uploaded_file):
    # Create a folder to store the frames
    folder_name = "video_frames"
    vid_folder_name = "video_file"
    if os.path.exists(vid_folder_name):
        for file_name in os.listdir(vid_folder_name):
            file_path = os.path.join(vid_folder_name, file_name)
            os.remove(file_path)
        os.rmdir(vid_folder_name)
    if os.path.exists(folder_name):
        # Remove the existing folder and its contents
        for file_name in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file_name)
            os.remove(file_path)
        os.rmdir(folder_name)
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(vid_folder_name, exist_ok=True)

    # If the user has uploaded a file
    if uploaded_file is not None:
        # Save the uploaded video file to the folder
        video_path = os.path.join(vid_folder_name, uploaded_file.name)
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Create a VideoCapture object to read the uploaded video
        video_capture = cv2.VideoCapture(video_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Wrap the frame saving code in a Streamlit spinner
        with st.spinner("Creating  frames from uploaded video..."):
            # Loop over the frames of the video
            for frame_index in range(frame_count):
                # Set the position of the VideoCapture object to the next frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

                # Read the next frame from the video
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Save the frame as a PNG image with a unique name
                file_name = os.path.join(folder_name, f"{frame_index}.png")
                cv2.imwrite(file_name, frame)

        # Release the VideoCapture object
        video_capture.release()
        #st.success("Frames Created !")
def get_image_count():
    directory_path = "video_frames"

    # List all files in the directory
    files = os.listdir(directory_path)

    # Count the number of files with image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    image_count = 0
    for file in files:
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_count += 1

    st.write("Number of frames created :", image_count)
    return image_count

with open('shots_model.pickle', 'rb') as f:
    shot_model = pickle.load(f)

sequence_length = 70

acc_array = []

def analyze_frames(act, sequence_length):
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        action = act

        with st.spinner("Analyzing Created Frames..."):
            # Loop through video length, sequence length
            for frame_num in range(sequence_length):  # sequence_length
                photo_path = "video_frames\\"

                frame = cv2.imread(photo_path + '\\' + str(frame_num) + ".png")

                image, results = mediapipe_detection(frame, holistic)

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

                left_knee_angle_acc = cal_left_knee_angle_acc(left_knee_angle, action)
                left_shoulder_angle_acc = cal_left_shoulder_angle_acc(left_shoulder_angle, action)
                left_elbow_angle_acc = cal_left_elbow_angle_acc(left_elbow_angle, action)
                right_knee_angle_acc = cal_right_knee_angle_acc(right_knee_angle, action)
                right_shoulder_angle_acc = cal_right_shoulder_angle_acc(right_shoulder_angle, action)
                right_elbow_angle_acc = cal_right_elbow_angle_acc(right_elbow_angle, action)

                # Make a prediction and get the class probabilities
                score = shot_model.predict_proba(
                    [[left_knee_angle, left_shoulder_angle, left_elbow_angle, right_knee_angle,
                      right_shoulder_angle, right_elbow_angle]])

                # Get the probability of each shot type
                drive_prob = score[0][0]
                pull_prob = score[0][1]
                sweep_prob = score[0][2]

                # change the prob for each shot
                if(action == 'Drive'):
                    final_acc = drive_prob * 100 * 0.4 + left_knee_angle_acc + left_shoulder_angle_acc + left_elbow_angle_acc + right_knee_angle_acc + right_shoulder_angle_acc + right_elbow_angle_acc

                elif (action == 'Sweep'):
                    final_acc = sweep_prob * 100 * 0.4 + left_knee_angle_acc + left_shoulder_angle_acc + left_elbow_angle_acc + right_knee_angle_acc + right_shoulder_angle_acc + right_elbow_angle_acc

                elif (action == 'Pullshot'):
                    final_acc = pull_prob * 100 * 0.4 + left_knee_angle_acc + left_shoulder_angle_acc + left_elbow_angle_acc + right_knee_angle_acc + right_shoulder_angle_acc + right_elbow_angle_acc
                # st.write('Total Accuracy : {:.2f}%'.format(final_acc))
                acc_array.append(round(final_acc, 2))

        max_value_index = acc_array.index(max(acc_array))

        st.markdown("""
            <style>
                .max-acc {
                    font-size: 34px;
                    font-weight: bold;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
        # st.write('Min Value: ' + str(min(acc_array)))
        if (max(acc_array) >= 75):
            st.balloons()
            st.markdown("""
                    <p>
                    Shot Performance : <i style='color:green'>Awsome</i>
                    </p>
                    """, unsafe_allow_html=True)
            description = 'You have done a great job !'
            status = 'Awsome'
        elif (max(acc_array) >= 55 and max(acc_array) < 75):
            st.markdown("""
                    <p>
                    Shot Performance : <i style='color:blue'>Great</i>
                    </p>
                    """, unsafe_allow_html=True)
            description = 'Keep doing practice. almost reached to the target !'
            status = 'Great'
        elif (max(acc_array) >= 30 and max(acc_array) < 55):
            st.markdown("""
                    <p>
                    Shot Performance : <i style='color:yellow'>Neutral</i>
                    </p>
                    """, unsafe_allow_html=True)
            description = 'Try harder !'
            status = 'Neutral'
        elif (max(acc_array) >= 00 and max(acc_array) < 30):
            st.markdown("""
                    <p>
                    Shot Performance : <i style='color:red'>Weak</i>
                    </p>
                    """, unsafe_allow_html=True)
            description = 'Practice more to become a star !'
            status = 'Weak'
        else:
            pass
            status = 'Error'

        #Db Operations
        cursor.execute(table_create_command)
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        values = (max(acc_array), status, current_datetime)
        cursor.execute("INSERT INTO details (acc_value, status, date) VALUES (?, ?, ?)", values)
        connection.commit()
        connection.close()

        st.markdown(f"Recorded Maximum Accuracy : <span class='max-acc'>  {max(acc_array)}%</span>",
                    unsafe_allow_html=True)

        chart_data = []
        for i, acc in enumerate(acc_array):
            chart_data.append({'Frame': i, 'Accuracy': acc})

        # create a DataFrame from the data
        df = pd.DataFrame(chart_data)
        st.subheader('Accuracy Variety')
        st.line_chart(df, x="Frame", y="Accuracy")

        # set the directory where the images are stored
        image_dir = "video_frames"

        # construct the path to the image using os.path.join()
        image_path = os.path.join(image_dir, f"{max_value_index}.png")

        max_value_image = Image.open(image_path)
        caption = 'Frame: ' + str(max_value_index)
        st.write(' ')
        st.subheader('Best Performed Moment')
        st.image(max_value_image, caption=caption, use_column_width=True)

        st.write(' ')
        st.subheader('Description')
        st.info(description)

        st.write(' ')
        st.write('---')
        with st.expander(':information_source: How this rate calculated?'):
            st.subheader('40% From responsible model')
            st.write('This AI model gives the probability (a scientific value) of shot been accurate according to it.')
            st.subheader('60% From body coordinates')
            st.write(
                'In this software it calculates 6 angles of your body with the extracted body coordinates. '
                'then you will get an accurate rate for each angle based on the standard values that angles '
                'should be when perform a accurate shot.')

            data_table = [
                {'Factor': 'AI Model', 'Weight': '40%'},
                {'Factor': 'Left Knee Angle', 'Weight': '10%'},
                {'Factor': 'Right Knee Angle', 'Weight': '10%'},
                {'Factor': 'Left Shoulder Angle', 'Weight': '10%'},
                {'Factor': 'Right Shoulder Angle', 'Weight': '10%'},
                {'Factor': 'Left Elbow Angle', 'Weight': '10%'},
                {'Factor': 'Right Elbow Angle', 'Weight': '10%'},
                {'Factor': 'Total', 'Weight': '100%'}
            ]

            # Create a Pandas DataFrame from your data
            df = pd.DataFrame(data_table)

            hide_table_row_index = """
                                <style>
                                thead tr th:first-child {display:none}
                                tbody th {display:none}
                                </style>
                                """

            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            # Display the table using Streamlit
            st.table(df)

            st.subheader('Accuracy Rate Formula:')
            st.markdown("""<p><b>Final Accuracy</b> = <i>(AI Model Probability * 100 * 0.4) + Left Knee Angle + Left Shoulder Angle +
                                Left Elbow Angle + Right Knee Angle + Right Shoulder Angle + Right Elbow Angle</i></p>""",
                        unsafe_allow_html=True)

        with st.expander(':confused: Not Satisfied With Accurate Rate?'):
            st.subheader('Possible ways to increase the accuracy rate. ')
            st.markdown("""
                <ul>
                    <li>Record the video from side view.</li>
                    <li>Try to crop the video that contains only the performed shot.</li>
                    <li>Record with high quality camera.</li>
                    <li>Use high fps value while recording.</li>
                </ul><br>&nbsp;
            """, unsafe_allow_html=True)

################################################################################################
########################################### PROGRAM ############################################
################################################################################################

with st.container():
    left, center, right = st.columns(3)
    with left:
        st.empty()
    with center:

        st.image(logo, caption=None, width=None, use_column_width=True, clamp=False)

        # st.markdown("<h2 style='text-align: center; color: white;'>i Trainer</h2>", unsafe_allow_html=True)
        st.write("---")
        # lottie.st_lottie(lottie_coding, height=100, width=100, key="ball",loop=False,reverse=False)
        st.write("Please select the training mode that you want.")
        st.write("##")
        mode = st.selectbox('Select the Mode', ('Select', 'Batting', 'Bowling', 'View Data'))

        if (mode == 'Select'):
            st.empty()

        if (mode == 'Batting'):
            batting_shots = ['Select', 'Cover Drive', 'Sweep', 'Pull']
            shot = st.selectbox('Select the Batting Shot', batting_shots)
            if(shot == 'Select'):
                pass

            if (shot == 'Cover Drive'):
                bt_shot = 'Drive'

                st.set_option('deprecation.showfileUploaderEncoding', False)

                # Use the file_uploader function to allow the user to upload a video file
                drive_uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "MOV", "mkv"])

                if drive_uploaded_file is not None:
                    drivestartbtn = st.button('Start')

                    if drivestartbtn:
                        create_frames(drive_uploaded_file)
                        sequence_length = get_image_count()

                        #main method
                        analyze_frames('Drive', sequence_length)

            if (shot == 'Sweep'):
                bt_shot = 'Sweep'

                st.set_option('deprecation.showfileUploaderEncoding', False)

                # Use the file_uploader function to allow the user to upload a video file
                sweep_uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "MOV", "mkv"])

                if sweep_uploaded_file is not None:
                    sweepstartbtn = st.button('Start')

                    if sweepstartbtn:
                        create_frames(sweep_uploaded_file)
                        sequence_length = get_image_count()

                        # main method
                        analyze_frames('Sweep', sequence_length)

            if (shot == 'Pull'):
                bt_shot = 'Pullshot'

                st.set_option('deprecation.showfileUploaderEncoding', False)

                # Use the file_uploader function to allow the user to upload a video file
                pullshot_uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "MOV", "mkv"])

                if pullshot_uploaded_file is not None:
                    pullshotstartbtn = st.button('Start')

                    if pullshotstartbtn:
                        create_frames(pullshot_uploaded_file)
                        sequence_length = get_image_count()

                        # main method
                        analyze_frames('Pullshot', sequence_length)

        if (mode == 'Bowling'):
            bowling_poseses = ['Off-Spin', 'Fast', 'Yoker']
            b_pose = st.selectbox('Select the Bowling Pose', bowling_poseses)
            if (b_pose == 'Off-Spin'):
                bw_pose = 'Spin'

                spinbtn = st.button('Start')

                if spinbtn:

                    st.write('done')

            if (b_pose == 'Fast'):
                bw_pose = 'Fast'
                pass
            if (b_pose == 'Yoker'):
                bw_pose = 'Yoker'
                pass

        if (mode == 'View Data'):
            cursor.execute("SELECT id AS 'ID', acc_value AS 'Accuracy Rate', status AS 'Status', date AS 'Date' FROM details ORDER BY date")
            dtls = cursor.fetchall()
            # Define the HTML table header
            table = "<table align='center'>\n<thead>\n<tr>\n<th>#</th>\n\n<th>Accuracy</th>\n<th>Status</th>\n<th>Date</th>\n</tr>\n</thead>\n<tbody>\n"

            # Loop through the rows of the result set and add each row to the table
            for row in dtls:
                table += "<tr>\n<td>{}</td>\n\n<td>{}%</td>\n<td>{}</td>\n<td>{}</td>\n</tr>\n".format(row[0], row[1], row[2], row[3])

            # Close the table
            table += "</tbody>\n</table>"

            # Display the table using Markdown
            st.markdown(table, unsafe_allow_html=True)

            # Close the database connection
            connection.close()


        with right:
            st.empty()

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.empty()
with col2:
    st.empty()
with col3:
    st.empty()
with col4:
    lottie.st_lottie(lottie_gradient)
with col5:
    st.empty()
with col6:
    st.empty()
with col7:
    st.empty()

left2, center2, right2 = st.columns(3)
with left2:
    st.empty()
with center2:
    st.markdown("<p style='text-align: center; color: white;'>The AI Powered Performance Analyzer</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>&copy; 2023 <i>iTrainer</i></p>", unsafe_allow_html=True)
with right2:
    st.empty()