import asyncio
import time

import cv2
import mediapipe as mp
import numpy as np
from db import delete_last_insert, landmark_values_to_db
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

# ALPHABET_ONLY = True enables tracking for only right hand landmarks
ALPHABET_ONLY = True

HAND_MODEL_PATH = "./slap/models/hand_landmarker.task"
POSE_MODEL_PATH = "./slap/models/pose_landmarker_lite.task"

def extract_landmark_values(results):
    landmark_values = {}
    if not ALPHABET_ONLY:
        pose_landmarks_list = results["pose"].pose_landmarks
        landmark_values["Pose"] = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks_list[0]]).flatten()
    hand_landmarks_list = results["hand"].hand_landmarks

    for i, landmarks in enumerate(hand_landmarks_list):
        handedness_info = results["hand"].handedness[i][0]
        hand_label = handedness_info.category_name

        # hand_label can either be "Left" or "Right"
        if ALPHABET_ONLY and hand_label == "Left":
            continue
        landmark_values[hand_label] = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

    if "Left" not in landmark_values:
        landmark_values["Left"] = np.zeros(21*3)
    if "Right" not in landmark_values:
        landmark_values["Right"] = np.zeros(21*3)

    if ALPHABET_ONLY:
        return landmark_values["Right"]
    return landmark_values

def draw_landmarks_on_frame(image, detection_results):
    if not ALPHABET_ONLY:
        pose_landmarks_list = detection_results["pose"].pose_landmarks
    hand_landmarks_list = detection_results["hand"].hand_landmarks
    annotated_image = image

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    hand_landmark_style = drawing_styles.get_default_hand_landmarks_style()
    hand_connection_style = drawing_styles.get_default_hand_connections_style()

    if not ALPHABET_ONLY:
        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style,
            )
    for hand_landmarks in hand_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_style,
            connection_drawing_spec=hand_connection_style,
        )
    return annotated_image

def mediapipe_detection(image, timestamp_start, pose_model, hand_model):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image.flags.writeable = False
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    results = {}
    timestamp = int((time.perf_counter_ns() - timestamp_start) * 10^6)
    if not ALPHABET_ONLY:
        results["pose"] = pose_model.detect_for_video(image, timestamp)
    results["hand"] = hand_model.detect_for_video(image, timestamp)
    
#    image.flags.writeable = True
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def draw_overlay_on_frame(frame, symbol):
    h = frame.shape[0]
    w = frame.shape[1]
    cv2.putText(
        frame,
        f"Capture symbol: {symbol}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2,
        2,
    )
    cv2.putText(
        frame,
        "SPACE to capture, 1 to undo,",
        (20,h-70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2,
        2,
    )
    cv2.putText(
        frame,
        "Q to quit",
        (20,h-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2,
        2,
    )

async def main():
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    hand_options=HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    with (
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        timestamp_start = time.perf_counter_ns()
        cap = cv2.VideoCapture(0)
        symbol_to_capture = "A"
        while cap.isOpened():
            ret, frame = cap.read()

            results = mediapipe_detection(
                                                 frame,
                                                 timestamp_start=timestamp_start,
                                                 pose_model=pose_landmarker,
                                                 hand_model=hand_landmarker,
                                             )

            frame = draw_landmarks_on_frame(frame, results)

           
            frame = cv2.flip(frame, 1)

            draw_overlay_on_frame(frame, symbol_to_capture)
             
            cv2.imshow("SLAP", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('t'):
                landmark_values = extract_landmark_values(results)
                print(landmark_values)

            elif key == ord(' '):
                landmark_values = extract_landmark_values(results)
                await landmark_values_to_db(landmark_values, symbol_to_capture)

            elif key == ord('1'):
                await delete_last_insert()

            elif key == ord('a'):
                symbol_to_capture = 'A'

            elif key == ord('b'):
                symbol_to_capture = 'B'
            
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
