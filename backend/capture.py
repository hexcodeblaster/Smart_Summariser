import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define left and right eye indices
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the image horizontally for a selfie view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmark.landmark[i] for i in left_eye_indices]
            right_eye_landmarks = [face_landmark.landmark[i] for i in right_eye_indices]
            h, w, c = frame.shape
            for left_eye_landmark in left_eye_landmarks:
                x = int(left_eye_landmark.x * w)
                y = int(left_eye_landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            for right_eye_landmark in right_eye_landmarks:
                x = int(right_eye_landmark.x * w)
                y = int(right_eye_landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
