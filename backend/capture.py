import cv2
import mediapipe as mp
import pygame

def display_calibration_points():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    screen.fill((0, 0, 0))
    calibration_points = [(100, 100), (640, 360), (1180, 620)]  # Example points
    font = pygame.font.Font(None, 36)

    collected_points = []
    for point in calibration_points:
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 0, 0), point, 10)
        pygame.display.flip()

        # Wait for the user to look at the point and press SPACE
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    collected_points.append(point)
                    waiting = False

    pygame.quit()
    return collected_points

def detect_pupil(frame, eye_landmarks):
    h, w, c = frame.shape
    x_min = int(min([landmark.x for landmark in eye_landmarks]) * w)
    y_min = int(min([landmark.y for landmark in eye_landmarks]) * h)
    x_max = int(max([landmark.x for landmark in eye_landmarks]) * w)
    y_max = int(max([landmark.y for landmark in eye_landmarks]) * h)

    # Crop the eye region
    eye_roi = frame[y_min:y_max, x_min:x_max]
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

    cv2.imshow("eye roi", eye_roi)

    # Apply thresholding
    _, threshold = cv2.threshold(gray_eye, 100, 255, cv2.THRESH_BINARY_INV)

    # Detect contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) > 5:
            # Get the center of the pupil
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            # Adjust cx and cy to global frame coordinates
            global_cx = int(cx + x_min)
            global_cy = int(cy + y_min)
            return global_cx, global_cy, eye_roi

    return None, None, eye_roi


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

            # Detect the pupil for each eye
            left_pupil_x, left_pupil_y, left_eye_roi = detect_pupil(frame, left_eye_landmarks)
            right_pupil_x, right_pupil_y, right_eye_roi = detect_pupil(frame, right_eye_landmarks)

            # Draw the pupil on the frame
            if left_pupil_x and left_pupil_y:
                cv2.circle(frame, (left_pupil_x, left_pupil_y), 1, (0, 255, 0), -1)
            if right_pupil_x and right_pupil_y:
                cv2.circle(frame, (right_pupil_x, right_pupil_y), 1, (0, 255, 0), -1)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
