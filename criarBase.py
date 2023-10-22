import cv2
import pickle
import os
import numpy as np

import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

capture = cv2.VideoCapture(0)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

pathCinza50 = './baseOficial/600img/cinza/50x50/'
pathCinza100 = './baseOficial/600img/cinza/100x100/'
pathColor50 = './baseOficial/600img/colorida/50x50/'
pathColor100 = './baseOficial/600img/colorida/100x100/'

qtd = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while capture.isOpened():
  
    ret, frame = capture.read()

    if ret == True:
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_eye_cord1 = _normalized_to_pixel_coordinates(
                face_landmarks[445].x, 
                face_landmarks[445].y, 
                width, 
                height)
            left_eye_cord2 = _normalized_to_pixel_coordinates(
                face_landmarks[453].x, 
                face_landmarks[453].y, 
                width, 
                height)
            left_eye_cord3 = _normalized_to_pixel_coordinates(
                face_landmarks[440].x, 
                face_landmarks[440].y, 
                width, 
                height)
            left_eye_cord4 = _normalized_to_pixel_coordinates(
                face_landmarks[450].x, 
                face_landmarks[450].y, 
                width, 
                height)
            right_eye_cord1 = _normalized_to_pixel_coordinates(
                face_landmarks[225].x, 
                face_landmarks[225].y, 
                width, 
                height)
            right_eye_cord2 = _normalized_to_pixel_coordinates(
                face_landmarks[233].x, 
                face_landmarks[233].y, 
                width, 
                height)
            
            left_eye_50 = cv2.resize(frame[left_eye_cord1[1]: left_eye_cord2[1], left_eye_cord2[0]: left_eye_cord1[0]], (50,50))
            right_eye_50 = cv2.resize(frame[right_eye_cord1[1]: right_eye_cord2[1], right_eye_cord1[0]: right_eye_cord2[0]], (50,50))
            cv2.imwrite(pathColor50 + 'esq_' + str(qtd) + '.jpg', left_eye_50)
            cv2.imwrite(pathColor50 + 'dir_' + str(qtd) + '.jpg', right_eye_50)

            left_eye_100 = cv2.resize(frame[left_eye_cord1[1]: left_eye_cord2[1], left_eye_cord2[0]: left_eye_cord1[0]], (100,100))
            right_eye_100 = cv2.resize(frame[right_eye_cord1[1]: right_eye_cord2[1], right_eye_cord1[0]: right_eye_cord2[0]], (100,100))
            cv2.imwrite(pathColor100 + 'esq_' + str(qtd) + '.jpg', left_eye_100)
            cv2.imwrite(pathColor100 + 'dir_' + str(qtd) + '.jpg', right_eye_100)

            left_gray_eye_50 = cv2.cvtColor(left_eye_50, cv2.COLOR_BGR2GRAY)
            right_gray_eye_50 = cv2.cvtColor(right_eye_50, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(pathCinza50 + 'esq_' + str(qtd) + '.jpg', left_gray_eye_50)
            cv2.imwrite(pathCinza50 + 'dir_' + str(qtd) + '.jpg', right_gray_eye_50)

            left_gray_eye_100 = cv2.cvtColor(left_eye_100, cv2.COLOR_BGR2GRAY)
            right_gray_eye_100 = cv2.cvtColor(right_eye_100, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(pathCinza100 + 'esq_' + str(qtd) + '.jpg', left_gray_eye_100)
            cv2.imwrite(pathCinza100 + 'dir_' + str(qtd) + '.jpg', right_gray_eye_100)

            qtd += 1

          
        cv2.imshow('esq', left_eye_50)
        cv2.imshow('dir', right_eye_50)
        cv2.imshow('webCam', frame)
    #clicar na tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
      break

capture.release()
cv2.destroyAllWindows()