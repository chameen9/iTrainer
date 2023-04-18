import mediapipe as mp
import pickle
import numpy as np                  # for array manipulation and mathematical operations
import cv2                          # open-source computer vision library

# initialize mediaPipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_holistic = mp.solutions.holistic # Holistic model
pose = mp_pose.Pose()

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

def cal_left_knee_angle_acc(left_knee_angle,shot_type):
    left_knee_angle_acc = 0.00
    if (shot_type == 'drive'):
        if (left_knee_angle > 180):
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

    elif(shot_type == 'pullshot'):
        return left_knee_angle_acc

    elif (shot_type == 'sweep'):
        return left_knee_angle_acc

    else:
        return left_knee_angle_acc

def cal_left_shoulder_angle_acc(left_shoulder_angle,shot_type):
    left_shoulder_angle_acc = 0.00

    if(shot_type == 'drive'):
        if (left_shoulder_angle > 120):
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

    elif (shot_type == 'pullshot'):
        return left_knee_angle_acc

    elif (shot_type == 'sweep'):
        return left_knee_angle_acc

    else:
        return left_knee_angle_acc

def cal_left_elbow_angle_acc(left_elbow_angle,shot_type):
    left_elbow_angle_acc = 0.00

    if(shot_type == 'drive'):
        if (left_elbow_angle > 150):
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

    elif (shot_type == 'pullshot'):
        return left_elbow_angle_acc

    elif (shot_type == 'sweep'):
        return left_elbow_angle_acc

    else:
        return left_elbow_angle_acc

def cal_right_knee_angle_acc(right_knee_angle, shot_type):
    right_knee_angle_acc = 0.00

    if(shot_type == 'drive'):
        if (right_knee_angle > 160):
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

    elif (shot_type == 'pullshot'):
        return right_knee_angle_acc

    elif (shot_type == 'sweep'):
        return right_knee_angle_acc

    else:
        return right_knee_angle_acc

def cal_right_shoulder_angle_acc(right_shoulder_angle, shot_type):
    right_shoulder_angle_acc = 0.00

    if(shot_type == 'drive'):
        if (right_shoulder_angle >110):
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

    elif (shot_type == 'pullshot'):
        return right_shoulder_angle_acc

    elif (shot_type == 'sweep'):
        return right_shoulder_angle_acc

    else:
        return right_shoulder_angle_acc

def cal_right_elbow_angle_acc(right_elbow_angle, shot_type):
    right_elbow_angle_acc = 0.00

    if(shot_type == 'drive'):
        if (right_elbow_angle > 160):
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

    elif (shot_type == 'pullshot'):
        return right_elbow_angle_acc

    elif (shot_type == 'sweep'):
        return right_elbow_angle_acc

    else:
        return right_elbow_angle_acc

with open('shots_model.pickle', 'rb') as f:
    model = pickle.load(f)

with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    action = 'drive'

    # Loop through video length, sequence length
    for frame_num in range(sequence_length):  # sequence_length
        photo_path = "dataset\drive\\"

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
        score = model.predict_proba([[left_knee_angle, left_shoulder_angle, left_elbow_angle, right_knee_angle,
                                      right_shoulder_angle, right_elbow_angle]])

        # Get the probability of each shot type
        drive_prob = score[0][0]
        pull_prob = score[0][1]
        sweep_prob = score[0][2]

        print(str(frame_num) + ' ------------------------')
        print('Drive Shot Probability: {:.2f}'.format(drive_prob))
        print('Pull Shot Probability : {:.2f}'.format(pull_prob))
        print('Sweep Shot Probability: {:.2f}'.format(sweep_prob))
        print('=============================')
        print('Drive Shot acc     : {:.2f}'.format(drive_prob / 10 * 4))
        print('Left Knee Angle Acc     : ' + str(left_knee_angle_acc))
        print('Left shoulder Angle Acc : ' + str(left_shoulder_angle_acc))
        print('Left elbow Angle Acc    : ' + str(left_elbow_angle_acc))
        print('Right knee Angle Acc    : ' + str(right_knee_angle_acc))
        print('Right shoulder Angle Acc: ' + str(right_shoulder_angle_acc))
        print('Right elbow Angle Acc   : ' + str(right_elbow_angle_acc))
        print('-----------------------------')
        print('Total Accuracy : {:.2f}%'.format(((drive_prob / 10 * 4) * 100) +
                                                left_knee_angle_acc +
                                                left_shoulder_angle_acc +
                                                left_elbow_angle_acc +
                                                right_knee_angle_acc +
                                                right_shoulder_angle_acc +
                                                right_elbow_angle_acc))
        print('-----------------------------')
