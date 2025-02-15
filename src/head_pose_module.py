import warnings
warnings.filterwarnings("ignore")

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import math

# Load the model
model = pickle.load(open('./model.pkl', 'rb'))

# Define column names
cols = []
for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
    for dim in ('x', 'y'):
        cols.append(pos+dim)

face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_features(img):
    NOSE, FOREHEAD, LEFT_EYE = 1, 10, 33
    MOUTH_LEFT, CHIN, RIGHT_EYE, MOUTH_RIGHT = 61, 199, 263, 291

    result = face_mesh.process(img)
    face_features = []

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                    face_features.extend([lm.x, lm.y])

    return face_features

def normalize(poses_df):
    normalized_df = poses_df.copy()

    for dim in ['x', 'y']:
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] -= poses_df['nose_'+dim]

        diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] /= diff

    return normalized_df

def compute_rmat(pitch, yaw, roll):
    yaw = -yaw
    rotation_matrix_ = cv2.Rodrigues(np.array([pitch, yaw, roll], dtype=np.float32))[0]
    axes_points = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    return rotation_matrix_ @ axes_points

def compute_tvec(nose_x, nose_y, eye_dist):
    eye_dist_ref = 0.064    # Average Adult Pupillary Distance
    fx = 1000
    fy = 1000
    px = 640
    py = 360
    tz = (eye_dist_ref / eye_dist) * (fx + fy) / 2.0 * (fx / fy)
    tx = (nose_x - px) * tz / fx
    ty = (nose_y - py) * tz / fy

    return np.array([tx, ty, tz])

def compute_cMo(pitch, yaw, roll, nose_x, nose_y, eye_dist):
    T = np.eye(4)
    T[:3, :3] = compute_rmat(pitch, yaw, roll)
    T[:3, 3] = compute_tvec(nose_x, nose_y, eye_dist)
    return T

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll], dtype=np.float32))[0]
    # axes_points = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    axes_points = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(np.int32)
    axes_points[0, :] += int(tx)
    axes_points[1, :] += int(ty)

    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

def process_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    img_h, img_w, _ = img.shape
    text = ''

    face_features = extract_features(img)
    if face_features:
        face_features_df = pd.DataFrame([face_features], columns=cols)
        face_features_normalized = normalize(face_features_df)
        pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
        nose_x = int(face_features_df['nose_x'].values[0] * img_w)
        nose_y = int(face_features_df['nose_y'].values[0] * img_h)
        eye_left_x = int(face_features_df['left_eye_x'].values[0] * img_w)
        eye_left_y = int(face_features_df['left_eye_y'].values[0] * img_h)
        eye_right_x = int(face_features_df['right_eye_x'].values[0] * img_w)
        eye_right_y = int(face_features_df['right_eye_y'].values[0] * img_h)
        eye_dist = math.sqrt((eye_left_x - eye_right_x)**2 + (eye_left_y - eye_right_y)**2)
        img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)
        cMo = compute_cMo(pitch_pred, yaw_pred, roll_pred, nose_x, nose_y, eye_dist)

        if pitch_pred > 0.3:
            text = 'Top'
            if yaw_pred > 0.3:
                text = 'Top Left'
            elif yaw_pred < -0.3:
                text = 'Top Right'
        elif pitch_pred < -0.3:
            text = 'Bottom'
            if yaw_pred > 0.3:
                text = 'Bottom Left'
            elif yaw_pred < -0.3:
                text = 'Bottom Right'
        elif yaw_pred > 0.3:
            text = 'Left'
        elif yaw_pred < -0.3:
            text = 'Right'
        else:
            text = 'Forward'

    cv2.putText(img, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img