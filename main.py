import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam

# Конфигурация отображения
config = {
    "show_hands": True,
    "show_pose": True,
}

CAPTURE_DEVICE = 0
cap = cv2.VideoCapture(CAPTURE_DEVICE)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

last_face_mask = None
last_eye_mask = None
last_mouth_mask = None

def create_dense_face_mask(frame_shape, face_landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    points = []
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)

    hull = cv2.convexHull(points)
    cv2.fillPoly(mask, [hull], 255)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask

def create_eye_mask(frame_shape, face_landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Индексы ключевых точек глаз в MediaPipe Face Mesh
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    for eye_indices in [left_eye_indices, right_eye_indices]:
        points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

def create_mouth_mask(frame_shape, face_landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Внешний контур губ
    outer_lip_indices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 185
    ]

    # Внутренний контур губ (отверстие рта)
    inner_lip_indices = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 291
    ]

    # Создаём маску внешнего контура губ
    outer_points = np.array(
        [[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in outer_lip_indices],
        dtype=np.int32
    )
    cv2.fillPoly(mask, [outer_points], 255)

    # Создаём маску внутреннего контура губ
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    inner_points = np.array(
        [[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in inner_lip_indices],
        dtype=np.int32
    )
    cv2.fillPoly(inner_mask, [inner_points], 255)

    # Вырезаем внутреннюю часть рта из маски
    mask = cv2.subtract(mask, inner_mask)

    # Немного расширяем маску для сглаживания краёв
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def add_noise(image, mask):
    noise = np.random.normal(loc=0, scale=50, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    result = image.copy()
    result[mask == 255] = noisy_image[mask == 255]
    return result

def anonymize_face(frame, face_mask, eye_mask):
    # Создаём маску лица без глаз
    face_no_eyes_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(eye_mask))

    # Размытие лица без глаз
    blurred = cv2.GaussianBlur(frame, (99, 99), 100)

    # Добавляем шум к размытой области лица без глаз
    blurred_noisy = add_noise(blurred, face_no_eyes_mask)

    # Собираем итоговое изображение
    eyes_region = cv2.bitwise_and(frame, frame, mask=eye_mask)
    face_region = cv2.bitwise_and(blurred_noisy, blurred_noisy, mask=face_no_eyes_mask)
    background_mask = cv2.bitwise_not(face_mask)
    background = cv2.bitwise_and(frame, frame, mask=background_mask)

    result = cv2.add(background, face_region)
    result = cv2.add(result, eyes_region)

    return result

with pyvirtualcam.Camera(width=640, height=420, fps=60) as cam:
    print(f'Виртуальная камера запущена: {cam.device}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Изменение размера кадра
        resized_frame = cv2.resize(frame, (640, 420))

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb_frame)

        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]
            last_face_mask = create_dense_face_mask(resized_frame.shape, face_landmarks)
            last_eye_mask = create_eye_mask(resized_frame.shape, face_landmarks)
            last_mouth_mask = create_mouth_mask(resized_frame.shape, face_landmarks)

        if last_face_mask is not None and last_eye_mask is not None and last_mouth_mask is not None:
            resized_frame = anonymize_face(resized_frame, last_face_mask, last_eye_mask)

            # Отобразить рот (аналогично глазам)
            mouth_region = cv2.bitwise_and(resized_frame, resized_frame, mask=last_mouth_mask)
            resized_frame[last_mouth_mask == 255] = mouth_region[last_mouth_mask == 255]

        # Рисуем руки и позу (если включено в конфиге)
        if config["show_hands"]:
            results_hands = hands.process(rgb_frame)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * resized_frame.shape[1])
                        y = int(landmark.y * resized_frame.shape[0])
                        cv2.circle(resized_frame, (x, y), 2, color=(255, 0, 0), thickness=-1)

        if config["show_pose"]:
            results_pose = pose.process(rgb_frame)
            if results_pose.pose_landmarks:
                for landmark in results_pose.pose_landmarks.landmark:
                    x = int(landmark.x * resized_frame.shape[1])
                    y = int(landmark.y * resized_frame.shape[0])
                    cv2.circle(resized_frame, (x, y), 2, color=(0, 0, 255), thickness=-1)

        cam.send(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        cv2.imshow('Anonymized Face', resized_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('h'):
            config["show_hands"] = not config["show_hands"]
            print(f"Отображение рук: {'Включено' if config['show_hands'] else 'Выключено'}")
        elif key == ord('p'):
            config["show_pose"] = not config["show_pose"]
            print(f"Отображение позы: {'Включено' if config['show_pose'] else 'Выключено'}")

cap.release()
cv2.destroyAllWindows()

