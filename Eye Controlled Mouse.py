import cv2
import mediapipe as mp
import pyautogui
import time

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
drawing = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Video capture
cap = cv2.VideoCapture(0)

# Drag state
dragging = False
last_click_time = 0
last_scroll_time = 0
last_drag_time = 0
click_cooldown = 1.0
scroll_cooldown = 1.0
drag_hold_time = 1.2

# Eye landmarks (based on MediaPipe)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS_CENTER = 474

def get_eye_distance(landmarks, top_id, bottom_id, height):
    return abs(landmarks[top_id].y - landmarks[bottom_id].y) * height

def norm_to_screen(x, y, frame_w, frame_h):
    screen_x = int((x / frame_w) * screen_w)
    screen_y = int((y / frame_h) * screen_h)
    return screen_x, screen_y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    ih, iw, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Move mouse with right iris
        if len(face_landmarks) > RIGHT_IRIS_CENTER:
            iris = face_landmarks[RIGHT_IRIS_CENTER]
            screen_x, screen_y = norm_to_screen(iris.x * iw, iris.y * ih, iw, ih)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

        # Blink detection
        left_eye = get_eye_distance(face_landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, ih)
        right_eye = get_eye_distance(face_landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, ih)

        blink_threshold = 5
        now = time.time()

        left_blink = left_eye < blink_threshold
        right_blink = right_eye < blink_threshold
        both_blink = left_blink and right_blink

        # Left click
        if left_blink and not right_blink and now - last_click_time > click_cooldown:
            pyautogui.click()
            print("Left click")
            last_click_time = now

        # Right click
        elif right_blink and not left_blink and now - last_click_time > click_cooldown:
            pyautogui.rightClick()
            print("Right click")
            last_click_time = now

        # Scroll (both eyes blink briefly)
        elif both_blink and now - last_scroll_time > scroll_cooldown:
            pyautogui.scroll(-50)
            print("Scroll down")
            last_scroll_time = now

        # Start Drag
        if left_blink and not right_blink:
            if now - last_drag_time > drag_hold_time and not dragging:
                pyautogui.mouseDown()
                dragging = True
                last_drag_time = now
                print("Drag started")

        # End Drag
        if dragging and not left_blink:
            pyautogui.mouseUp()
            dragging = False
            print("Drag ended")

    cv2.imshow("Eye Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
