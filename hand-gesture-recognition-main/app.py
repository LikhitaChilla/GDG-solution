import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
import time
import csv
import copy
import itertools
from collections import Counter, deque
import keyboard

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
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        debug_image = copy.deepcopy(img)
        
        # Process the image
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        fps = self.cvFpsCalc.get()

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Hand sign classification
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
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

                # Drawing landmarks and info
                debug_image = self.draw_bounding_rect(debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    "",
                )
        else:
            self.point_history.append([0, 0])

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
    
    # Initialize WebRTC only once
    if 'webrtc_ctx' not in st.session_state:
        st.session_state.webrtc_ctx = webrtc_streamer(
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
