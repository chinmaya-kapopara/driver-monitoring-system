# Required Libraries
import dlib
import cv2
import time
import pickle
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

# Initializing Variables
[x1, y1, x2, y2] = [0, 0, 0, 0]
[sleep, drowsy, alert] = [0, 0, 0]
setup_status_color = (0, 0, 0)
setup_frame = 0
disoriented = 0
face_not_found = 0
FRAME_THRESHOLD = 25
OPEN_EYE_THRESHOLD = 0
CLOSED_EYE_THRESHOLD = 0

setup_status = ""

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


# Play program start sound file
playsound("start2.mp3")
time.sleep(1)

name_list = []
encode_list = []
oet_list = []
cet_list = []
#
# with open('drivers.pkl', 'wb') as file:
#     pickle.dump(name_list, file)  # Name list
#     pickle.dump(encode_list, file)  # Encoding list
#     pickle.dump(oet_list, file)  # Open eye threshold list
#     pickle.dump(cet_list, file)  # Closed eye threshold list

while True:
    frame = camera.read()[1]  # Capturing frame
    frame = imutils.resize(frame, width=800)  # Resizing frame for faster processing

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting to gray frame for better face detection
    faces = face_detector(gray_frame)  # Detecting faces

    if not recognition or len(faces) != 0:  # Skipping loop if no face found
        face_not_found = 0

        # Finding drivers face by finding maximum area face
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

            # Loading stored file for face recognition
            with open('drivers.pkl', 'rb') as file:
                name_list = pickle.load(file)  # Name list
                encode_list = pickle.load(file)  # Face encodings list
                oet_list = pickle.load(file)  # Open eye threshold list
                cet_list = pickle.load(file)  # Closed eye threshold list

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converting frame from BRG to RGB for face recognition

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
            if dist_face[match_index] < 0.48:
                print("Driver:", name_list[match_index])
                playsound("already_verified.mp3")
                play("reverify.mp3")

                # Asking driver for re-verification
                if input("Do you still want to re-verify? (Y/N) : ").lower() == 'n':
                    playsound("ok.mp3")
                    break  # Exiting program if driver doesn't want to re-verify

                playsound("ok.mp3")

                # Deleting previously stored data
                name_list.pop(match_index)
                encode_list.pop(match_index)
                oet_list.pop(match_index)
                cet_list.pop(match_index)

            play("enter_name.mp3")
            name_list.append(input("Please enter your name: "))  # Appending new name to the list
            encode_list.append(enc)  # Appending new face encoding to the list

            if dist_face[match_index] >= 0.48:
                playsound("verification_successful.mp3")

            playsound("setup.mp3")
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

        rotation_vector = \
            cv2.solvePnP(predefined_3d_points, frame_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)[1]
        rmat = cv2.Rodrigues(rotation_vector)[0]
        face_angles = cv2.RQDecomp3x3(rmat)[0]

        x_orientation = 180 - abs(face_angles[0])  # Up and down orientation
        y_orientation = abs(face_angles[1])  # Left and right orientation
        z_orientation = abs(face_angles[2])  # Tilt orientation

        # Storing separate eye landmarks
        left_eye_landmarks = landmarks[36:42]
        right_eye_landmarks = landmarks[42:48]

        # Calculating Eye aspect ratio (EAR) by taking average of those of both eyes
        left_eye_EAR = EAR_calculate(left_eye_landmarks)
        right_eye_EAR = EAR_calculate(right_eye_landmarks)
        EAR = (left_eye_EAR + right_eye_EAR) / 2

        mouth_landmarks = landmarks[60:68]  # Storing mouth landmarks
        if x_orientation < 15 and y_orientation < 15 and z_orientation < 15:  # Making sure the driver is looking straight

            # Open eye threshold calculation
            if not open_eyes_setup:

                # Instructing driver to keep eyes open
                if setup_frame == 0:
                    setup_status = "Keep Eyes Open!"
                    setup_status_color = (255, 0, 0)
                    play("open_eyes_setup.mp3")

                # Calculating open eye threshold by taking average
                elif setup_frame > 180:
                    OPEN_EYE_THRESHOLD /= 140
                    open_eyes_setup = True
                    setup_frame = -1

                # Giving driver some time to get ready
                elif setup_frame >= 40:
                    OPEN_EYE_THRESHOLD += EAR

                setup_frame += 1

            # Closed eye threshold calculation
            elif not closed_eyes_setup:

                # Instructing driver to keep eyes closed
                if setup_frame == 0:
                    setup_status = "Keep Eyes Closed!"
                    setup_status_color = (255, 0, 0)
                    play("closed_eyes_setup.mp3")

                # Calculating open eye threshold by taking average
                elif setup_frame > 180:
                    setup_status = "recognition SUCCESSFUL!"
                    setup_status_color = (0, 255, 255)
                    CLOSED_EYE_THRESHOLD /= 140

                    closed_eyes_setup = True
                    setup_frame = -1

                    OPEN_EYE_THRESHOLD = (OPEN_EYE_THRESHOLD + CLOSED_EYE_THRESHOLD) / 2 - 0.01  # Calculating actual open eye threshold
                    CLOSED_EYE_THRESHOLD += 0.015  # Calculating actual closed eye threshold

                    # Making sure driver has followed instructions properly
                    if CLOSED_EYE_THRESHOLD >= OPEN_EYE_THRESHOLD:
                        playsound("setup_failed.mp3")
                        break
                    else:
                        playsound("setup_successful.mp3")

                    # Rounding threshold values to 3 digit precision
                    OPEN_EYE_THRESHOLD = round(OPEN_EYE_THRESHOLD, 3)
                    CLOSED_EYE_THRESHOLD = round(CLOSED_EYE_THRESHOLD, 3)

                    oet_list.append(OPEN_EYE_THRESHOLD)   # Appending new open eye threshold to the list
                    cet_list.append(CLOSED_EYE_THRESHOLD)   # Appending new closed eye threshold to the list

                    print("Open Eye Threshold =", OPEN_EYE_THRESHOLD)
                    print("Closed Eye Threshold =", CLOSED_EYE_THRESHOLD)

                    # Storing updated data into file
                    with open('drivers.pkl', 'wb') as file:
                        pickle.dump(name_list, file)  # Name list
                        pickle.dump(encode_list, file)  # Encoding list
                        pickle.dump(oet_list, file)  # Open eye threshold list
                        pickle.dump(cet_list, file)  # Closed eye threshold list

                    break

                # Giving driver some time to get ready
                elif setup_frame >= 40:
                    CLOSED_EYE_THRESHOLD += EAR

                setup_frame += 1
        else:
            disoriented += 1

        # Confirming disoriented head (sleepiness) when action is detected consistently
        if disoriented > FRAME_THRESHOLD:
            play("alert5.mp3")
            disoriented = 0

        cv2.putText(frame, setup_status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, setup_status_color, 3)  # Drawing instructions on frame

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
