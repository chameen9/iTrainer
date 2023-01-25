import mediapipe as mp
import numpy as np
import cv2
import os
import streamlit as st
import requests
import streamlit_lottie as lottie
from PIL import Image
import time

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
        mode = st.selectbox('Select the Mode', ('Select', 'Batting', 'Bowling'))

        if (mode == 'Select'):
            st.empty()

        if (mode == 'Batting'):
            batting_shots = ['Drive', 'Sweep', 'pullshot']
            shot = st.selectbox('Select the Batting Shot', batting_shots)
            if (shot == 'Drive'):
                bt_shot = 'drive'

                drivestartbtn = st.button("Start")

                if drivestartbtn:

                    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5) as holistic:
                        action = bt_shot

                        # Loop through video length, sequence length
                        with st.spinner('Processing...'):
                            # Loop through video length, sequence length
                            for frame_num in range(sequence_length):
                                start_time = time.time()
                                photo_path = "dataset\dirve\\"

                                frame = cv2.imread(photo_path + '\\' + str(frame_num) + ".png")

                                image, results = mediapipe_detection(frame, holistic)

                                # Draw landmarks
                                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                          mp_drawing.DrawingSpec((255, 0, 0), 1, 1),
                                                          mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
                                                          )

                                final = extract_body_keypoints(results)

                                DATA_PATH = 'output'
                                OUT_PHOTO_PATH = ''
                                FOLDER_NAME = 'outphoto'
                                output_photo_path = os.path.join(OUT_PHOTO_PATH, FOLDER_NAME ,action)
                                npy_path = os.path.join(DATA_PATH, action, str(frame_num) + action)
                                np.save(npy_path, final)
                                cv2.imwrite(output_photo_path + '\\' + str(frame_num) + "_with_landmarks.png", image)

                                end_time = time.time()
                                processing_time = end_time - start_time
                                time.sleep(processing_time)
                            st.success("Operation Completed")
            if (shot == 'Sweep'):
                bt_shot = 'Sweep'
            if (shot == 'pullshot'):
                bt_shot = 'pullshot'

        if (mode == 'Bowling'):
            bowling_poseses = ['Spin', 'Fast', 'Yoker']
            b_pose = st.selectbox('Select the Bowling Pose', bowling_poseses)
            if (b_pose == 'Spin'):
                bw_pose = 'Spin'
            if (b_pose == 'Fast'):
                bw_pose = 'Fast'
            if (b_pose == 'Yoker'):
                bw_pose = 'Yoker'
        with right:
            st.empty()

col1,col2,col3,col4,col5,col6,col7 = st.columns(7)
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
    st.markdown("<p style='text-align: center; color: white;'>The AI Powered personnel trainer</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>&copy; 2023 <i>iTrainer</i></p>", unsafe_allow_html=True)
with right2:
    st.empty()