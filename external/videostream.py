import cv2
import mediapipe as mp
import streamlit as st
import socket
import numpy as np
import time

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="ESP32 Hand Gesture Recognition", layout="wide")

st.title("ESP32 Hand Gesture Recognition")
st.write("Recognizing gestures from ESP32 video stream")

# -----------------------------
# UI controls
# -----------------------------
col1, col2 = st.columns([1, 3])

with col1:
    run_stream = st.checkbox("Start ESP32 Stream", value=True)
    show_landmarks = st.checkbox("Show landmarks", value=True)
    max_num_hands = st.slider("Max hands", 1, 2, 1)
    detection_conf = st.slider("Detection confidence", 0.1, 1.0, 0.7, 0.05)
    tracking_conf = st.slider("Tracking confidence", 0.1, 1.0, 0.7, 0.05)

frame_placeholder = col2.empty()
label_placeholder = st.empty()

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -----------------------------
# Gesture helper functions
# -----------------------------
def finger_up(lm, tip_idx, pip_idx):
    return lm[tip_idx].y < lm[pip_idx].y

def thumb_up(lm, handedness_label):
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]

    if handedness_label == "Right":
        return thumb_tip.x < thumb_ip.x
    else:
        return thumb_tip.x > thumb_ip.x

def classify_gesture(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark

    index_is_up = finger_up(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_is_up = finger_up(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_is_up = finger_up(lm, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_is_up = finger_up(lm, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    thumb_is_up = thumb_up(lm, handedness_label)

    up_count = sum([thumb_is_up, index_is_up, middle_is_up, ring_is_up, pinky_is_up])

    if up_count == 0:
        return "Fist"
    if up_count >= 4 and index_is_up and middle_is_up and ring_is_up and pinky_is_up:
        return "Open Palm"
    if index_is_up and middle_is_up and not ring_is_up and not pinky_is_up:
        return "Peace"
    if index_is_up and not middle_is_up and not ring_is_up and not pinky_is_up:
        return "Pointing"
    if thumb_is_up and not index_is_up and not middle_is_up and not ring_is_up and not pinky_is_up:
        return "Thumbs Up"

    return "Unknown"

# -----------------------------
# UDP setup
# -----------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", 4432))
sock.settimeout(0.01)

command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
command_sock.bind(("0.0.0.0", 4431))

buffer = b''
MAX_BUFFER = 200000

ESP32_IP = "192.168.1.27"  # change if needed
COMMAND_PORT = 4431
COMMAND_INTERVAL_SECONDS = 0.15
NO_HAND_STOP_SECONDS = 0.5


def gesture_to_command(handedness_label, gesture):
    if gesture == "Thumbs Up":
        return "forward"
    if gesture == "Peace":
        return "backward"
    if gesture == "Pointing":
        return "right" if handedness_label == "Right" else "left"
    if gesture in ("Open Palm", "Fist"):
        return "stop"
    return "no hand"

# -----------------------------
# Main loop
# -----------------------------
if run_stream:
    st.info("Receiving ESP32 stream...")
    last_command = None
    last_command_time = 0.0
    last_detection_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        model_complexity=1,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    ) as hands:

        while run_stream:
            try:
                # Receive UDP packets
                data, addr = sock.recvfrom(2048)

                if addr[0] != ESP32_IP:
                    continue

                buffer += data
                #print(len(buffer))
                if len(buffer) > MAX_BUFFER:
                    print("overflow")
                    buffer = b''

                # Process complete JPEG frames
                while True:
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9', start)

                    if start != -1 and end != -1 and end > start:
                        jpg = buffer[start:end+2]
                        buffer = buffer[end+2:]

                        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

                        if frame is None:
                            continue

                        # -----------------------------
                        # Gesture Recognition
                        # -----------------------------
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        results = hands.process(rgb)
                        detected_labels = []
                        command_to_send = None

                        if results.multi_hand_landmarks and results.multi_handedness:
                            for hand_landmarks, handedness in zip(
                                results.multi_hand_landmarks,
                                results.multi_handedness
                            ):
                                handedness_label = handedness.classification[0].label
                                gesture = classify_gesture(hand_landmarks, handedness_label)
                                detected_labels.append(f"{handedness_label}: {gesture}")
                                if command_to_send is None:
                                    gesture_to_send = gesture_to_command(handedness_label, gesture)
                                    if gesture_to_send is not None:
                                        command_to_send = "gesture:" + gesture_to_send

                                if show_landmarks:
                                    mp_drawing.draw_landmarks(
                                        frame,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style(),
                                    )

                                h, w, _ = frame.shape
                                xs = [p.x for p in hand_landmarks.landmark]
                                ys = [p.y for p in hand_landmarks.landmark]

                                x_min = int(min(xs) * w)
                                y_min = max(int(min(ys) * h) - 10, 30)

                                cv2.putText(
                                    frame,
                                    f"{handedness_label}: {gesture}",
                                    (x_min, y_min),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2
                                )

                        current_time = time.time()

                        if command_to_send is not None:
                            last_detection_time = current_time
                        elif current_time - last_detection_time >= NO_HAND_STOP_SECONDS:
                            command_to_send = "gesture:stop"
                       
                        if command_to_send is not None and (
                            command_to_send != last_command
                            or current_time - last_command_time >= COMMAND_INTERVAL_SECONDS
                        ):
                            print(f"sending: {command_to_send}")
                            command_sock.sendto(command_to_send.encode('utf-8'), (ESP32_IP, COMMAND_PORT))
                            last_command = command_to_send
                            last_command_time = current_time

                        # -----------------------------
                        # Display
                        # -----------------------------
                        if detected_labels:
                            label_placeholder.success(
                                f"{' | '.join(detected_labels)} | command: {command_to_send or last_command or 'none'}"
                            )
                        else:
                            label_placeholder.info(f"No hand detected | command: {command_to_send or last_command or 'none'}")

                        frame_placeholder.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            width='stretch'
                        )

                    else:
                        break

            except socket.timeout:
                pass

            # allow Streamlit to update checkbox
            run_stream = st.session_state.get("Start ESP32 Stream", run_stream)
