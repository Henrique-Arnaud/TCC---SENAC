#Importação 
import cv2
import datetime as dt
import pickle
import os
import numpy as np
import time
from pygame import mixer
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#Utilização do Modelo de IA
pick = open('modelos/model1.sav', 'rb')
model = pickle.load(pick)
pick.close()

#Utilização da câmera para capturar o vídeo em tempo real
capture = cv2.VideoCapture(0)

width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

#Utilização do som para alertar o estado de sonolência
mixer.init()
mixer.music.load("./beep.mp3")

#Parâmetros dos textos inseridos na tela
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2

sonolencia = False
predictAnterior = 0
tempoA = dt.datetime.now()
tempoDecorrido = 0
categories = ['aberto', 'fechado']

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  #Loop enquanto a câmera está aberta
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

            #Recorte e Redimensionamento do olho esquerdo e do olho direito
            left_eye = cv2.resize(frame[left_eye_cord1[1]: left_eye_cord2[1], left_eye_cord2[0]: left_eye_cord1[0]], (50,50))
            right_eye = cv2.resize(frame[right_eye_cord1[1]: right_eye_cord2[1], right_eye_cord1[0]: right_eye_cord2[0]], (50,50))

            
            #Prediction para Modelos Coloridos
            #cv2.imwrite('olhos/' + 'esquerdo' + '.jpg', left_eye)
            #cv2.imwrite('olhos/' + 'direito' + '.jpg', right_eye)
            #olhos = []
            #for img in os.listdir('olhos'):
              #imgPath = os.path.join('olhos', img)
              #eyeImg = cv2.imread(imgPath, 0)
              #try:
                #olhos.append(eyeImg.flatten())
              #except Exception as e:
                #pass
            #prediction = model.predict(olhos)


            #Prediction para Modelos Cinzas
            left_gray_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            left_gray_eye = cv2.equalizeHist(left_gray_eye) 
            right_gray_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            right_gray_eye = cv2.equalizeHist(right_gray_eye) 
            prediction = model.predict([np.array(left_gray_eye).flatten(),np.array(right_gray_eye).flatten()])

            #Verifica se a última classificação foi dois olhos fechados
            if predictAnterior == 1: 
              #Verifica se as duas predições atuais foram olhos fechados
              if prediction[0] == 1 and prediction[1] == 1:
                tempoB = dt.datetime.now()
                tempoDecorrido = tempoB - tempoA
                #Verifica se o tempo com os dois olhos fechados é maior que 1.5 segundos
                if tempoDecorrido.seconds > 1.5:
                  sonolencia = True

              #Caso as duas predições atuais não foram olhos fechados
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
                  
            #Caso a última classificação não foi dois olhos fechados
            else:
                if prediction[0] == 1 and prediction[1] == 1:
                    tempoA = dt.datetime.now()
                    predictAnterior = 1
            
            #Textos inseridos na tela
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
            
        #Verifica se foi detectada a sonolência
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

    #clicar na tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
      break
mixer.quit()
capture.release()
cv2.destroyAllWindows()