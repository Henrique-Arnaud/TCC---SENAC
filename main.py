import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# For webcam input:
cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

      left_eye = cv2.resize(image[left_eye_cord1[1]: left_eye_cord2[1], left_eye_cord2[0]: left_eye_cord1[0]], (50,50))
      right_eye = cv2.resize(image[right_eye_cord1[1]: right_eye_cord2[1], right_eye_cord1[0]: right_eye_cord2[0]], (50,50))
      cv2.imshow('teste olho esquerdo', image[left_eye_cord1[1]: left_eye_cord2[1], left_eye_cord2[0]: left_eye_cord1[0]])
      cv2.imshow('teste olho direito', image[right_eye_cord1[1]: right_eye_cord2[1], right_eye_cord1[0]: right_eye_cord2[0]])

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('webcam', cv2.flip(image, 1))
    if cv2.waitKey(1) == ord('q'):
      break
cap.release()