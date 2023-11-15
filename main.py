import cv2
import datetime as dt
import pickle
import os
import numpy as np
import time
from pygame import mixer
#from sklearn.model_selection import train_test_split

import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


#pick = open('model.sav', 'rb')
pick = open('modeloTesteLeo/model2.sav', 'rb')
model = pickle.load(pick)
pick.close()
capture = cv2.VideoCapture(0)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

mixer.init()
mixer.music.load("./beep.mp3")
#features = []
#labels = ['aberto', 'fechado']

tempoA = dt.datetime.now()

predictAnterior = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
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

            left_eye = cv2.resize(frame[left_eye_cord1[1]: left_eye_cord2[1], left_eye_cord2[0]: left_eye_cord1[0]], (100,100))
            right_eye = cv2.resize(frame[right_eye_cord1[1]: right_eye_cord2[1], right_eye_cord1[0]: right_eye_cord2[0]], (100,100))
        
              #eye_cropped = (eye_cropped.reshape(-1, 2))
            left_gray_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            left_gray_eye = cv2.equalizeHist(left_gray_eye) 
            right_gray_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            right_gray_eye = cv2.equalizeHist(right_gray_eye) 
              #cv2.imshow("webcam2", gray_eye)
            #cv2.imwrite(path + dt.datetime.now().strftime('IMG-%Y-%m-%d-%H%M%S') + '.jpg', gray_eye)
              #if(tempoDecorrido.seconds >= 3):
            #features.append(np.array(left_gray_eye).flatten())
            #features.append(np.array(left_gray_eye).flatten())
            #xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.5)
            
            
            #flat_data_test = np.array([(left_eye).flatten(),(right_eye).flatten()])
            #df=pd.DataFrame(flat_data_test) 
            #x=df.iloc[:,:-20000] #input data 
            #y=df.iloc[:,-1] #output data
            #prediction = model.predict(x)
            prediction = model.predict([np.array(left_gray_eye).flatten(), np.array(right_gray_eye).flatten()])
            categories = ['aberto', 'fechado']
            print('prediction esquerdo: ', categories[prediction[0]])
            print('prediction direito: ', categories[prediction[1]])
          
            #features.pop()
            #features.pop()
            if predictAnterior == 1: 
              if prediction[0] and prediction[1] == 1:
                #if sonolencia == False:
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
                if prediction[0] and prediction[1] == 1:
                    tempoA = dt.datetime.now()
                    predictAnterior = 1
            cv2.putText(frame, 'Esquerdo: ' + categories[prediction[0]],
                      (10, 360),
                      font,
                      1,
                      fontColor,
                      thickness,
                      lineType)
            cv2.putText(frame, 'Direito: ' + categories[prediction[1]],
                      (10, 400),
                      font,
                      1,
                      fontColor,
                      thickness,
                      lineType)
            cv2.putText(frame, 'Tempo: ' + str(tempoDecorrido),
                      (10, 450),
                      font,
                      1,
                      fontColor,
                      thickness,
                      lineType)
        #print(sonolencia)
        if sonolencia == True:
          mixer.music.play()
          while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)
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
mixer.quit()
capture.release()
cv2.destroyAllWindows()