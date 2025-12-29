import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
from collections import deque

import numpy as np
import streamlit as st
import cv2
import mediapipe as mp
import tensorflow as tf

# Import after deps exist (streamlit-webrtc uses it)
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ==================
# CONFIG
# ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN = 30
MIN_DYNAMIC_FRAMES = 20
MOTION_THRESH = 0.15

STATIC_MODEL_PATH = os.path.join(BASE_DIR, "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(BASE_DIR, "dynamic_model.keras")
STATIC_LABELS_PATH = os.path.join(BASE_DIR, "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(BASE_DIR, "labels_dynamic.npy")

st.set_page_config(page_title="🤖 Sign Language Translator", layout="wide")

# ==================
# LOAD MODELS
# ==================
@st.cache_resource
def load_assets():
    static_model = tf.keras.models.load_model(STATIC_MODEL_PATH, compile=False)
    dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH, compile=False)
    labels_static = np.load(STATIC_LABELS_PATH, allow_pickle=True)
    labels_dynamic = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)
    return static_model, dynamic_model, labels_static, labels_dynamic

try:
    m_static, m_dynamic, le_static, le_dynamic = load_assets()
except Exception as e:
    st.error("❌ Model loading failed.")
    st.exception(e)
    st.stop()

# ==================
# FEATURE ENGINEERING (same as your training)
# ==================
def get_120_features(coords_seq):
    feats = []
    for frame in coords_seq:
        wrist = frame[0]
        norm = frame - wrist
        scale = np.max(np.linalg.norm(norm, axis=1))
        if scale > 0:
            norm /= scale
        bones = np.array([norm[i] - norm[0] for i in range(1, 21)]).flatten()  # 60
        feats.append(bones)

    feats = np.array(feats)  # (T,60)
    if len(feats) > 1:
        vel = np.diff(feats, axis=0)
        vel = np.vstack([vel, np.zeros((1, 60))])
    else:
        vel = np.zeros_like(feats)

    return np.concatenate([feats, vel], axis=1)  # (T,120)

def calculate_motion_intensity(features, window=3):
    if len(features) < window:
        return 0.0
    v = features[-window:, 60:]
    return float(np.mean(np.linalg.norm(v, axis=1)))

def pad_sequence(seq, target_len):
    if len(seq) >= target_len:
        return seq[-target_len:]
    return seq + [seq[-1]] * (target_len - len(seq))

# ==================
# STATE MACHINE
# ==================
class GestureDetector:
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LEN)
        self.last_motion_time = time.time()
        self.dynamic = False

    def update(self, coords, motion):
        t = time.time()
        if coords is not None:
            self.buffer.append(coords)

        if motion > MOTION_THRESH:
            self.last_motion_time = t
            if len(self.buffer) >= MIN_DYNAMIC_FRAMES:
                self.dynamic = True

        still_time = t - self.last_motion_time

        if self.dynamic and still_time < 0.5:
            return "Dynamic"
        if still_time > 2:
            self.dynamic = False
            return "Static"
        return "Transition"

# ==================
# VIDEO PROCESSOR
# ==================
class Processor(VideoProcessorBase):
    def __init__(self):
        self.detector = GestureDetector()
        self.last_label = "Ready"

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.drawer = mp.solutions.drawing_utils

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = self.hands.process(rgb)
        label = self.last_label

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            self.drawer.draw_landmarks(img, lm, mp.solutions.hands.HAND_CONNECTIONS)

            buf = list(self.detector.buffer) + [coords]
            feats = get_120_features(buf)
            motion = calculate_motion_intensity(feats)
            mode = self.detector.update(coords, motion)

            if mode == "Dynamic" and len(buf) >= MIN_DYNAMIC_FRAMES:
                seq = pad_sequence(buf, SEQ_LEN)
                x = get_120_features(seq).reshape(1, SEQ_LEN, 120)
                pred = m_dynamic.predict(x, verbose=0)
                label = str(le_dynamic[int(np.argmax(pred))])

            elif mode == "Static":
                x = feats[-1].reshape(1, 120)
                pred = m_static.predict(x, verbose=0)
                label = str(le_static[int(np.argmax(pred))])

            self.last_label = label

        cv2.rectangle(img, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(img, self.last_label, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================
# UI
# ==================
st.title("🤖 Communication Friend – AI-Based Sign Language Translation")

webrtc_streamer(
    key="sign-lang",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
