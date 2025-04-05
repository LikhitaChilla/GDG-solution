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

# ========================
# Configuration & Setup
# ========================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_hands = mp.solutions.hands

# ========================
# Helper Functions
# ========================
def load_labels(label_path):
    """Load label names from CSV"""
    with open(label_path, encoding='utf-8-sig') as f:
        return [row[0] for row in csv.reader(f)]

# ========================
# Constants & Mappings
# ========================
history_length = 16
DEBOUNCE_DELAY = 0.2

gesture_map = {
    "No key": None,
    "Q": "up",
    "W": "down",
    "E": "left",
    "R": "right"
}

# ========================
# Video Processor Class
# ========================
class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.point_history = deque(maxlen=history_length)
        self.finger_gesture_history = deque(maxlen=history_length)
        self.current_gesture = None
        self.last_gesture_time = 0
        self.pressed_key = None
        
        # Initialize MediaPipe Hands instance for this processor
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        
        # Initialize classifiers
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        
        # Load labels
        self.keypoint_labels = load_labels('hand-gesture-recognition-main/model/keypoint_classifier/keypoint_classifier_label.csv')
        self.point_history_labels = load_labels('hand-gesture-recognition-main/model/point_history_classifier/point_history_classifier_label.csv')

    def calc_bounding_rect(self, image, landmarks):
        """Calculate bounding box around hand"""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        """Convert landmarks to pixel coordinates"""
        image_width, image_height = image.shape[1], image.shape[0]
        return [[min(int(lm.x * image_width), image_width - 1), 
                min(int(lm.y * image_height), image_height - 1)] 
               for lm in landmarks.landmark]

    def pre_process_landmark(self, landmark_list):
        """Normalize landmark coordinates"""
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
        for idx in range(len(temp_landmark_list)):
            temp_landmark_list[idx][0] -= base_x
            temp_landmark_list[idx][1] -= base_y
        
        # Normalize
        flattened = list(itertools.chain.from_iterable(temp_landmark_list))
        max_val = max(map(abs, flattened)) or 1  # Avoid division by zero
        return [x/max_val for x in flattened]

    def draw_landmarks(self, image, landmark_point):
        """Draw hand landmarks and connections"""
        if len(landmark_point) > 0:
            # Draw connections
            connections = [
                (2, 3), (3, 4),         # Thumb
                (5, 6), (6, 7), (7, 8),  # Index
                (9, 10), (10, 11), (11, 12),  # Middle
                (13, 14), (14, 15), (15, 16), # Ring
                (17, 18), (18, 19), (19, 20), # Little
                (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)  # Palm
            ]
            
            for start, end in connections:
                cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), 
                        (0, 0, 0), 6)
                cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), 
                        (255, 255, 255), 2)
            
            # Draw keypoints
            for idx, point in enumerate(landmark_point):
                color = (255, 255, 255)
                radius = 5
                if idx in [4, 8, 12, 16, 20]:  # Fingertips
                    radius = 8
                cv2.circle(image, tuple(point), radius, color, -1)
                cv2.circle(image, tuple(point), radius, (0, 0, 0), 1)
        
        return image

    def recv(self, frame):
        """Main processing function for each frame"""
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            debug_image = copy.deepcopy(img)
            
            # Process hand detection
            results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                    results.multi_handedness):
                    # Get landmarks
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed = self.pre_process_landmark(landmark_list)
                    
                    # Classify gesture
                    hand_sign_id = self.keypoint_classifier(pre_processed)
                    gesture = gesture_map.get(self.keypoint_labels[hand_sign_id], None)
                    
                    # Handle key presses with debounce
                    current_time = time.time()
                    if gesture != self.current_gesture and (current_time - self.last_gesture_time) >= DEBOUNCE_DELAY:
                        if self.pressed_key:
                            keyboard.release(self.pressed_key)
                        if gesture:
                            keyboard.press(gesture)
                        
                        self.current_gesture = gesture
                        self.pressed_key = gesture
                        self.last_gesture_time = current_time
                    
                    # Draw UI
                    brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                    debug_image = self.draw_landmarks(debug_image, landmark_list)
                    
                    # Display gesture info
                    cv2.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[1] - 22), 
                                 (0, 0, 0), -1)
                    info_text = f"{handedness.classification[0].label[0:]}: {self.keypoint_labels[hand_sign_id]}"
                    cv2.putText(debug_image, info_text, (brect[0] + 5, brect[1] - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(debug_image, format="bgr24")
        
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return frame

# ========================
# Streamlit UI
# ========================
def main():
    st.title("👋 Hand Gesture Recognition")
    st.markdown("""
    Control your computer with hand gestures!  
    The app will simulate keyboard presses based on your hand movements.
    """)
    
    # Gesture examples
    cols = st.columns(4)
    gesture_info = [
        ("Up", "hand-gesture-recognition-main/up.jpg", "Q gesture → Up arrow"),
        ("Down", "hand-gesture-recognition-main/down.jpg", "W gesture → Down arrow"),
        ("Left", "hand-gesture-recognition-main/left.jpg", "E gesture → Left arrow"),
        ("Right", "hand-gesture-recognition-main/right.jpg", "R gesture → Right arrow")
    ]
    
    for col, (name, img, desc) in zip(cols, gesture_info):
        with col:
            st.image(img, width=150)
            st.caption(f"**{name}**")
            st.caption(desc)
    
    # WebRTC Streamer
    st.subheader("Camera Feed")
    ctx = webrtc_streamer(
        key="hand-gesture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HandGestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.info("ℹ️ Note: Keyboard simulation only works when the browser window is focused.")
    st.warning("⚠️ Make sure to allow camera permissions when prompted!")

if __name__ == "__main__":
    main()
