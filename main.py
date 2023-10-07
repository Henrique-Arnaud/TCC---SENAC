import cv2
import datetime as dt
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split

import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()
capture = cv2.VideoCapture(0)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

features = []
labels = ['aberto', 'fechado']

tempoA = dt.datetime.now()

predictAnterior = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 1
lineType               = 2

sonolencia = False
#path utilizado para criar nosso dataset (selecionar uma pasta vazia para depois separar as imagens em aberto ou fechado)
path = './nossoDataset/'
tempoDecorrido = 0
#capture.set(cv2.CAP_PROP_FPS, 10)
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

            left_eye = cv2.resize(frame[left_eye_cord1[1]: left_eye_cord2[1], left_eye_cord2[0]: left_eye_cord1[0]], (50,50))
            right_eye = cv2.resize(frame[right_eye_cord1[1]: right_eye_cord2[1], right_eye_cord1[0]: right_eye_cord2[0]], (50,50))
    
        
              #eye_cropped = (eye_cropped.reshape(-1, 2))
            left_gray_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            left_gray_eye = cv2.equalizeHist(left_gray_eye) 
            right_gray_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
              #cv2.imshow("webcam2", gray_eye)
            #cv2.imwrite(path + dt.datetime.now().strftime('IMG-%Y-%m-%d-%H%M%S') + '.jpg', gray_eye)
              #if(tempoDecorrido.seconds >= 3):
            features.append(np.array(left_gray_eye).flatten())
            features.append(np.array(left_gray_eye).flatten())
            xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.5)
          
            prediction = model.predict(xtrain)
            categories = ['aberto', 'fechado']
            print('prediction: ', categories[prediction[0]])
          
            features.pop()
            features.pop()
            if predictAnterior == 1: 
              if prediction[0] == 1:
                if sonolencia == False:
                  tempoB = dt.datetime.now()
                  tempoDecorrido = tempoB - tempoA
                  if tempoDecorrido.seconds > 1.5:
                    sonolencia = True

              else:
                predictAnterior = 0
                if sonolencia == True:
                  sonolencia = False
                  cv2.putText(frame,'',
                          bottomLeftCornerOfText,
                          font,
                          fontScale,
                          fontColor,
                          thickness,
                          lineType)
            else:
                if prediction[0] == 1:
                    tempoA = dt.datetime.now()
                    predictAnterior = 1
        print(sonolencia)
        if sonolencia == True:
                cv2.putText(frame,'Sono Detectado!',
                      bottomLeftCornerOfText,
                      font,
                      fontScale,
                      fontColor,
                      thickness,
                      lineType)
        cv2.imshow('webCam', frame)
    #if sonolencia == True:
    #  cv2.putText(frame,'Sono Detectado!',
    #                  bottomLeftCornerOfText,
    #                  font,
    #                  fontScale,
    #                  fontColor,
    #                  thickness,
    #                  lineType)
    # if(eye_center):

    #clicar na tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
      break

capture.release()
cv2.destroyAllWindows()