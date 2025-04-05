import av
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration,WebRtcMode
import time
import csv
import copy
import itertools
from collections import Counter, deque
import keyboard
import cv2
# Custom imports (you'll need to provide these files)
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Initialize classifiers
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load labels
def load_labels(label_path):
    with open(label_path, encoding='utf-8-sig') as f:
        return [row[0] for row in csv.reader(f)]

keypoint_classifier_labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
point_history_classifier_labels = load_labels('model/point_history_classifier/point_history_classifier_label.csv')

# Gesture mapping
gesture_map = {
    "No key": None,
    "Q": "up",
    "W": "down",
    "E": "left",
    "R": "right"
}

# Constants
history_length = 16
DEBOUNCE_DELAY = 0.2

class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.point_history = deque(maxlen=history_length)
        self.finger_gesture_history = deque(maxlen=history_length)
        self.current_gesture = None
        self.last_gesture_time = 0
        self.pressed_key = None
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        
    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0

        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value if max_value != 0 else 0

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)
        base_x, base_y = 0, 0

        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]
            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
        return temp_point_history

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

            # Index finger
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

            # Middle finger
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

            # Ring finger
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

            # Little finger
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

            # Palm
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index in [4, 8, 12, 16, 20]:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def draw_bounding_rect(self, image, brect):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if finger_gesture_text != "":
            cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv2.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
        return image

    def draw_info(self, image, fps):
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        debug_image = copy.deepcopy(img)
        
        # Process the image
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        fps = self.cvFpsCalc.get()

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                               results.multi_handedness):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(
                    debug_image, self.point_history)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                gesture = gesture_map.get(keypoint_classifier_labels[hand_sign_id])

                if gesture != self.current_gesture:
                    current_time = time.time()
                    if current_time - self.last_gesture_time >= DEBOUNCE_DELAY:
                        if self.pressed_key and (self.pressed_key != gesture):
                            keyboard.release(self.pressed_key)
                        if gesture is not None:
                            keyboard.press(gesture)
                            self.pressed_key = gesture

                        self.current_gesture = gesture
                        self.last_gesture_time = current_time

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    self.finger_gesture_history).most_common()

                # Drawing part
                debug_image = self.draw_bounding_rect(debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            self.point_history.append([0, 0])

        debug_image = self.draw_point_history(debug_image, self.point_history)
        debug_image = self.draw_info(debug_image, fps)

        return av.VideoFrame.from_ndarray(debug_image, format="bgr24")

def main():
    st.title("Hand Gesture Recognition")
    st.write("Welcome! Control actions with these hand gestures:")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("up.jpg", width=100)
        st.write("Up (Q)")
    with col2:
        st.image("down.jpg", width=100)
        st.write("Down (W)")
    with col3:
        st.image("left.jpg", width=100)
        st.write("Left (E)")
    with col4:
        st.image("right.jpg", width=100)
        st.write("Right (R)")
    
    st.write("### Camera Feed")
    webrtc_ctx = webrtc_streamer(
        key="hand-gesture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HandGestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.info("Note: Keyboard simulation will only work when the browser window is focused.")

if __name__ == '__main__':
    main()
