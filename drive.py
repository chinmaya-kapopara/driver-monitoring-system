# Required Libraries
import dlib
import cv2
import time
import pickle
import pyttsx3
import imutils
import numpy as np
import face_recognition as fr
from threading import Thread
from imutils import face_utils
from playsound import playsound
from scipy.spatial import distance as dist

face_detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Face landmarks detector

camera = cv2.VideoCapture(0)

# Offline voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 250)

# Initializing Variables
[x1, y1, x2, y2] = [0, 0, 0, 0]
[sleep, drowsy, alert] = [0, 0, 0]
[focused, distracted] = [0, 0]
driver_status_color = (0, 255, 0)
attention_status_color = (0, 255, 0)
setup_status_color = (0, 0, 0)
setup_frame = 0
face_not_found = 0
FRAME_THRESHOLD = 25
OPEN_EYE_THRESHOLD = 0
CLOSED_EYE_THRESHOLD = 0

setup_status = ""
driver_status = "AWAKE"
attention_status = "FOCUSED"

recognition = False
alarm = False
open_eyes_setup = False
closed_eyes_setup = False

main_face = dlib.rectangle()

# Predefined 3d face points for orientation calculation
predefined_3d_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip point
    (0.0, -330.0, -65.0),  # Chin centre point
    (-225.0, 170.0, -135.0),  # Left eye left corner point
    (225.0, 170.0, -135.0),  # Right eye right corner point
    (-150.0, -150.0, -125.0),  # Mouth left corner point
    (150.0, -150.0, -125.0)  # Mouth right corner point
])

# Camera properties for orientation calculation
size = [450, 800]
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)

camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

dist_coeffs = np.zeros((4, 1))  # Assuming no distortion


# temp = 0
# setup_frame2 = 0

# Play music file
def play_thread(audio):
    global alarm
    if not alarm:
        alarm = True
        playsound(audio)
        alarm = False


# Music thread to play in background
def play(audio):
    t0 = Thread(target=play_thread, args=(audio,), daemon=True)
    t0.start()


# Calculate Eye aspect ratio
def EAR_calculate(arr):
    vertical = (dist.euclidean(arr[1], arr[5]) + dist.euclidean(arr[2], arr[4])) / 2
    horizontal = dist.euclidean(arr[0], arr[3])
    ratio = vertical / horizontal
    return ratio


# Calculates if driver is drowsy, sleepy or alert by eyes
def eye_status_calculate(ratio):
    if ratio > OPEN_EYE_THRESHOLD:
        return 2
    elif ratio >= CLOSED_EYE_THRESHOLD:
        return 1
    else:
        return 0


# Calculates Mouth aspect ratio
def MAR_calculate(arr):
    vertical = dist.euclidean(arr[2], arr[6])
    horizontal = dist.euclidean(arr[0], arr[4])
    ratio = vertical / horizontal
    return ratio


# Calculates if driver if feeling drowsy by yawn
def mouth_status_calculate(ratio):
    if ratio <= 0.5:
        return 2
    else:
        return 1


# Calculates eye region
def eye_detection(img, eye_landmarks):
    # Making only eye visible using masking
    black_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(black_mask, [eye_landmarks], True, 255, 1)
    cv2.fillPoly(black_mask, [eye_landmarks], 255)
    img = cv2.bitwise_and(img, img, mask=black_mask)

    # Separating eye region
    mini_x = np.min(eye_landmarks[:, 0])
    maxi_x = np.max(eye_landmarks[:, 0])
    mini_y = np.min(eye_landmarks[:, 1])
    maxi_y = np.max(eye_landmarks[:, 1])
    eye_region = img[mini_y: maxi_y, mini_x: maxi_x]

    eye_region = cv2.resize(eye_region, None, fx=5, fy=5)  # Enlarge frame
    cv2.medianBlur(eye_region, 3)  # Reduce noise

    return eye_region


# Calculates white pixels for gaze detection
def gaze_detection(eye):
    h, w = eye.shape

    left_side = eye[0: h, 0: int(w / 2)]
    left_side_white = cv2.countNonZero(left_side)

    right_side = eye[0: h, int(w / 2): w]
    right_side_white = cv2.countNonZero(right_side)

    return left_side_white, right_side_white


# Play program start sound file
playsound("start1.mp3")
time.sleep(1)

name_list = []
encode_list = []

while True:
    frame = camera.read()[1]  # Capturing frame
    frame = imutils.resize(frame, width=800)  # Resizing frame for faster processing

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting to gray frame for better face detection
    faces = face_detector(gray_frame)  # Detecting faces

    if not recognition or len(faces) != 0:  # Skipping loop if no face found
        face_not_found = 0

        # Finding drivers face by finding face with maximum area
        max_face_area = -1
        for face in faces:
            face_area = (face.right() - face.left()) * (face.bottom() - face.top())
            if face_area > max_face_area:
                max_face_area = face_area
                main_face = face

        # Calculating corner points of face
        x1 = main_face.left()
        y1 = main_face.top()
        x2 = main_face.right()
        y2 = main_face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing face rectangle

        if not recognition:
            # Loading stored file for face recognition (in order)
            with open('drivers.pkl', 'rb') as file:
                name_list = pickle.load(file)  # Name list
                encode_list = pickle.load(file)  # Face encodings list
                oet_list = pickle.load(file)  # Open eye threshold list
                cet_list = pickle.load(file)  # Closed eye threshold list

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converting frame from BGR to RGB for face recognition

            # Waiting until a face is detected
            try:
                loc = fr.face_locations(rgb)[0]  # Finding face for face recognition
            except IndexError:  # Exception handling if no face is found
                time.sleep(1)
                continue

            enc = fr.face_encodings(rgb, [loc])[0]  # Extracting facial features

            dist_face = fr.face_distance(encode_list, enc)  # Comparing face with stored faces
            match_index = np.argmin(dist_face)  # Calculating the most matching face
            # print(dist_face[match_index])

            # Checking confidence of face recognition
            if dist_face[match_index] < 0.45:
                print("Driver:", name_list[match_index])
                playsound("verification_successful.mp3")
                time.sleep(1)
                engine.say("Welcome" + name_list[match_index])  # Greeting the driver with name
                engine.runAndWait()  # Waiting until voice engine runs
                OPEN_EYE_THRESHOLD = oet_list[match_index]  # Fetching open eye threshold of detected driver
                CLOSED_EYE_THRESHOLD = cet_list[match_index]  # Fetching closed eye threshold of detected driver
                print(OPEN_EYE_THRESHOLD, CLOSED_EYE_THRESHOLD)
            else:
                # Ending program if face not recognised
                playsound("verification_failed.mp3")
                break

            recognition = True

        landmarks = predictor(gray_frame, main_face)  # Extracting facial landmarks
        landmarks = face_utils.shape_to_np(landmarks)  # Converting to np array for ease

        frame_points = np.array([landmarks[30],  # Nose tip point
                                 landmarks[8],  # Chin centre point
                                 landmarks[36],  # Left eye left corner point
                                 landmarks[45],  # Right eye right corner point
                                 landmarks[48],  # Mouth left corner point
                                 landmarks[54], ],  # Mouth right corner point
                                dtype="double")

        # # Drawing orientation line from tip of the nose
        # (success, rotation_vector, translation_vector) = cv2.solvePnP(predefined_3d_points, frame_points, camera_matrix,
        #                                                              dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
        #                                                translation_vector, camera_matrix, dist_coeffs)
        # p1 = int(frame_points[0][0]), int(frame_points[0][1])
        # p2 = int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])
        # cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # Calculating face orientation
        rotation_vector = cv2.solvePnP(predefined_3d_points, frame_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)[1]
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        # Storing separate eye landmarks
        left_eye_landmarks = landmarks[36:42]
        right_eye_landmarks = landmarks[42:48]

        # Calculating Eye aspect ratio (EAR) by taking average of those of both eyes
        left_eye_EAR = EAR_calculate(left_eye_landmarks)
        right_eye_EAR = EAR_calculate(right_eye_landmarks)
        EAR = (left_eye_EAR + right_eye_EAR) / 2

        mouth_landmarks = landmarks[60:68]  # Storing mouth landmarks
        MAR = MAR_calculate(mouth_landmarks)  # Calculating mouth aspect ratio (MAR)

        sleepiness_status = min(eye_status_calculate(EAR), mouth_status_calculate(MAR))

        # Orientation calculation (absolute)
        if MAR <= 0.3:  # Avoiding orientation errors in case of yawning
            x_orientation = 180 - abs(angles[0])  # Up and down orientation
            y_orientation = abs(angles[1])  # Left and right orientation
            z_orientation = abs(angles[2])  # Tilt orientation
        else:
            x_orientation = y_orientation = z_orientation = 0.0

        if x_orientation <= 15 and y_orientation <= 25:  # Side scene or phone distraction detection

            # Drowsiness detection using eye tracking and yawn tracking
            if sleepiness_status == 0 or z_orientation > 20:
                sleep += 1

                # Confirming drowsiness when action is detected consistently
                if sleep > FRAME_THRESHOLD:
                    [sleep, drowsy, alert] = [0, 0, 0]
                    driver_status = "SLEEPING"
                    driver_status_color = (0, 0, 255)
                    play("alert2.mp3")

            # Sleepiness detection using eye tracking
            elif sleepiness_status == 1:
                drowsy += 1

                # Confirming sleepiness when action is detected consistently
                if drowsy > FRAME_THRESHOLD:
                    [sleep, drowsy, alert] = [0, 0, 0]
                    driver_status = "DROWSY"
                    driver_status_color = [100, 100, 255]
                    play("alert1.mp3")

            # Alert detection using eye tracking
            else:
                alert += 1

                # Confirming alertness when action is detected consistently
                if alert > FRAME_THRESHOLD - 15:
                    [sleep, drowsy, alert] = [0, 0, 0]
                    driver_status = "AWAKE"
                    driver_status_color = (0, 255, 0)
                    drowsy_alarm = False
                    sleepy_alarm = False

            gray_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31,
                                               4)  # Converting frame to binary frame
            gray_frame = cv2.bilateralFilter(gray_frame, 15, 75, 75)  # Removing unwanted noise in frame

            # Getting both eye frames for gaze detection
            left_eye = eye_detection(gray_frame, left_eye_landmarks)
            right_eye = eye_detection(gray_frame, right_eye_landmarks)
            # Getting white pixels for gaze detection
            left_eye_left_white, left_eye_right_white = gaze_detection(left_eye)
            right_eye_left_white, right_eye_right_white = gaze_detection(right_eye)

            cv2.imshow("left eye", left_eye)  # Displaying an eye

            # Calculating gaze ratio
            try:
                GR = (left_eye_left_white + right_eye_left_white) / (
                        left_eye_right_white + right_eye_right_white)
            except ZeroDivisionError:
                GR = 5

            # Using gaze ratio to detect side scene distraction
            if EAR > OPEN_EYE_THRESHOLD:  # Avoiding errors when eyes are not fully opened
                if 0.6 < GR < 1.1:
                    focused += 1
                else:
                    distracted += 1
        else:
            EAR = 0.0
            MAR = 0.0
            GR = 0.0
            if x_orientation > 15:
                distracted += 2
            else:
                distracted += 1

            # if setup_frame2 == 99:
            #     print(temp/100)
            #     temp = 0
            #     setup_frame2 = 0
            # else:
            #     temp += GR
            #     setup_frame2 += 1

        # Confirming focused behavior when action is detected consistently
        if focused > 2 * FRAME_THRESHOLD:
            attention_status = "FOCUSED"
            attention_status_color = (0, 255, 0)
            [focused, distracted] = [0, 0]

        # Confirming distracted behaviour when action is detected consistently
        elif distracted > 3 * FRAME_THRESHOLD:
            play("alert3.mp3")
            attention_status = "DISTRACTED"
            attention_status_color = (0, 0, 255)
            [focused, distracted] = [0, 0]

        # Drawing various driver status and core variables on frame
        cv2.putText(frame, "(" + str(round(x_orientation, 1)) + ", " + str(round(y_orientation, 1)) + ", " +
                    str(round(z_orientation, 1)) + ")", (260, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        cv2.putText(frame, driver_status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, driver_status_color, 3)
        cv2.putText(frame, "EAR = " + str(round(EAR, 3)), (560, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255], 3)
        cv2.putText(frame, "MAR = " + str(round(MAR, 3)), (560, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255], 3)
        cv2.putText(frame, attention_status, (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, attention_status_color, 3)
        cv2.putText(frame, "GR = " + str(round(GR, 3)), (560, 440), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 255), 3)

        # Drawing eyes and mouth
        cv2.polylines(frame, [left_eye_landmarks], True, (255, 255, 0), 1)
        cv2.polylines(frame, [right_eye_landmarks], True, (255, 255, 0), 1)
        cv2.polylines(frame, [mouth_landmarks], True, (255, 255, 0), 1)

        # Drawing 68 face landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)
    else:
        face_not_found += 1

        # Face not detected alarm
        if face_not_found > FRAME_THRESHOLD:
            play("alert4.mp3")
            face_not_found = 0

    cv2.imshow("Frame", frame)  # Displaying frame

    # Exiting program when 'Esc' is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Exit process
camera.release()
cv2.destroyAllWindows()
