#IMPORTO LE LIBRERIE 
import cv2      
import time
import mediapipe as mp
import myLandmarks as ml
import numpy as np                      
import myClass 
import json
import math
import tensorflow as tf

from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from numpy.core.fromnumeric import shape
from tensorflow import keras
from numpy.core.records import array
from numpy.lib import flip
########################################################################

#INIZIALIZZO LE VARIABILI CHE UTILIZZO
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
list_data_frame=[]
file = open('list.json',"w")
frame=1
id_sbj = 1      #ID PERSONA NEL VIDEO
id_session = 1  #ID DEL VIDEO DELLA PERSONA

# VALORI CROPPED IMAGE , SERVONO PER RITAGLIARE LE IMMAGINI DEGLI OGGETTI DI STUDIO 
right_offset =  30
left_offset = 40
lips_offset = 40


valore_array_sx = []
valore_array_dx = []
colab_i = 0
########################################################################

#DEFINISCO LE CARATTERISTICHE DEI LANDMARKS
stats=myClass.Stat()
drawing_spec_silhouette = mp_drawing.DrawingSpec(color=ml.GREEN_COLOR, thickness=2, circle_radius=2)
drawing_spec_lips = mp_drawing.DrawingSpec(color=ml.YELLOW_COLOR, thickness=1, circle_radius=2)
drawing_spec_rightEye = mp_drawing.DrawingSpec(color=ml.BLUE_COLOR, thickness=1, circle_radius=2)
drawing_spec_leftEye = mp_drawing.DrawingSpec(color=ml.RED_COLOR, thickness=1, circle_radius=2)

#################################################################################

#CARICO IL MODELLO PRIMA DEL VIDEO (PARTE RETE NEURALE)
#model = keras.models.load_model('MODIFICATO/my_eye_model')

#################################################################################

# FOR STATIC IMAGES
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#############################################################################

# PRINT AND DRAW FACE MESH LANDMARKS ON THE IMAGE.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#######################################################################

# FOR WEBCAM INPUT: CARICO IL VIDEO E CI VADO A LAVORARE
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture('videoprova.mp4')

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

  count=0  
  while cap.isOpened():
    count +=1
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
# If loading a video, use 'break' instead of 'continue'.
      continue
    startTime = time.time()
  
# FLIP THE IMAGE HORIZONTALLY FOR A LATER SELFIE-VIEW DISPLAY, AND CONVERT THE BGR IMAGE TO RGB.
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

# To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
    endTime = time.time()
    
    cv2.putText(image,"cnt:" + str(int(count)), (7, 90), cv2.FONT_HERSHEY_SIMPLEX,
                   3, (0, 255, 0), 3, cv2.LINE_AA)
###########################################################################################################   
		
    # DRAW THE FACE MESH ANNOTATIONS ON THE IMAGE
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#IDENTIFICO QUALI KEYPOINTS APPARTENGONO A QUALE LANMARKS
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        rightEye = []
        leftEye = []
        lips = []
        silhouette = []
        
        re= myClass.Landmark("rightEye")
        le= myClass.Landmark("leftEye")
        li= myClass.Landmark("lips")
        time_of_frame=[]

        for l in ml.LANDMARKS:
            landmark = ml.recognizeLandmark(l)
            if(landmark == "rightEye"):
              rightEye.append(face_landmarks.landmark[l])
              re.AddCoords(face_landmarks.landmark[l].x,face_landmarks.landmark[l].y)
              
             
            if(landmark == "leftEye"):
              leftEye.append(face_landmarks.landmark[l])
              le.AddCoords(face_landmarks.landmark[l].x,face_landmarks.landmark[l].y)
              
              
            if(landmark == "lips"):
              lips.append(face_landmarks.landmark[l])
              li.AddCoords(face_landmarks.landmark[l].x,face_landmarks.landmark[l].y)
              silhouette.append(face_landmarks.landmark[l])     

############################################################################################              

#PER SEPARARE I LANDMARK DEGLI OGGETTI D'INTERESSE UTILIZZO UNA LISTA NORMALIZZATA
        landmark_subset_rightEye = landmark_pb2.NormalizedLandmarkList(landmark = rightEye)
        landmark_subset_leftEye = landmark_pb2.NormalizedLandmarkList(landmark = leftEye)
        landmark_subset_lips = landmark_pb2.NormalizedLandmarkList(landmark = lips)
        #landmark_subset_silhouette = landmark_pb2.NormalizedLandmarkList(landmark = silhouette)

#######################################################################################

        #INIZIO A LAVORARE CON LEFT EYE
        left_keypoint=[]
        x_array = np.zeros((1,16))
        y_array = np.zeros((1,16))
        
        for t,point in enumerate(landmark_subset_leftEye.landmark):
          left_keypoint.append({
            'x':point.x,
            'y':point.y,
            })
          x_array[0,t] = point.x
          y_array[0,t] = point.y
        
        left_x_max = np.amax(x_array) #x massima
        left_x_min = np.amin(x_array) #x minima
        left_y_max = np.amax(y_array) 
        left_y_min = np.amin(y_array)
        
        x1_left = int(left_x_min * (image.shape[1])) #1 larghezza
        y1_left = int(left_y_max * (image.shape[0])) #0 altezza
        x2_left = int(left_x_max * (image.shape[1]))
        y2_left = int(left_y_min * (image.shape[0]))

        print('frame:', count)

#STAMPO IMMAGINI (CON CROPPED VADO A RITAGLIARE L'IMMAGINE ASSEGNANDO DUE VALORI DI RITAGLIO ALLA X E Y )
        #reader=np.loadtxt('count.csv', delimiter=',')           
        path = 'MODIFICATO/drivers/images/left_eye/left_eye_' + str(id_sbj)+"_" + str(id_session) + "_"
        for i in range(reader.shape[0]):
          for j in range(reader.shape[1]):
            if count == reader[i][j]:
              cropped_image_left = image[(y2_left-left_offset):y2_left-left_offset+(y1_left-y2_left+(left_offset*2)),(x1_left-left_offset):x1_left-left_offset+(x2_left-x1_left+(left_offset*2))]
              if j == 0 or j == 2:
                cv2.imwrite(path + str(i)+"_"+str(j)+"_open"+".png", cropped_image_left)
              if j == 1:
                cv2.imwrite(path + str(i)+"_"+str(j)+"_close"+".png", cropped_image_left)
        frame = frame + 1

#STAMPO IMMAGINI PER COLAB, STAMPO OCCHIO CHIUSO/APERTO PER OCCHIO SINISTRO
        cropped_image_left_cb = image[(y2_left-left_offset):y2_left-left_offset+(y1_left-y2_left+(left_offset*2)),(x1_left-left_offset):x1_left-left_offset+(x2_left-x1_left+(left_offset*2))]
        resized_left = cv2.resize(cropped_image_left_cb, (96,96), interpolation=cv2.INTER_CUBIC)

        img_l = np.zeros((1,resized_left.shape[0], resized_left.shape[1], 3))  
        for i in range(3):  
          img_l[0,:,:,i] = resized_left
        #predictions = model.predict(img_l)
        
        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        cv2.putText(image,"left:" + str(int(predictions[0][0])), (7, 250), cv2.FONT_HERSHEY_SIMPLEX,
                   3, (0, 255, 0), 3, cv2.LINE_AA)
        valore_array_sx.append(int(predictions[0][0]))
    
#BOUNDING BOX
        #BOUNDIG BOX PIU ESTERNO(VERDE)
        start_point = (x1_left-10,y1_left+10)
        end_point = (x2_left+10,y2_left-10)
        color = (0,255,0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        #BOUNDING BOX PIU INTERNO NERO
        start_point = (x1_left,y1_left)
        end_point = (x2_left,y2_left)
        color = (0,0,0)
        thickness = 1
        cv2.putText(image,"ar:" + str(round((y1_left-y2_left)/(x2_left-x1_left) , 2)), (7, 400), cv2.FONT_HERSHEY_SIMPLEX,
                   3, (0, 0, 255), 3, cv2.LINE_AA)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
######################################################################################
#RIGHT EYE
        right_keypoint=[]
        x_array2 = np.zeros((1,16))
        y_array2= np.zeros((1,16)) 
        for t,point in enumerate(landmark_subset_rightEye.landmark):
          right_keypoint.append({
            'x':point.x,
            'y':point.y
          })
          x_array2[0,t] = point.x
          y_array2[0,t] = point.y
        
        right_x_max = np.amax(x_array2) #x massima
        right_x_min = np.amin(x_array2) #x minima
        right_y_max = np.amax(y_array2) 
        right_y_min = np.amin(y_array2)
        
        x1_right = int(right_x_min * (image.shape[1])) #1 larghezza
        y1_right = int(right_y_max * (image.shape[0])) #0 altezza
        x2_right = int(right_x_max * (image.shape[1]))
        y2_right = int(right_y_min * (image.shape[0]))

#STAMPO IMMAGINI     
        path_r = 'MODIFICATO/drivers/images/right_eye/right_eye' + str(id_sbj)+"_" + str(id_session) + "_"
        reader=np.loadtxt('count.csv', delimiter=',')
        for i in range(reader.shape[0]):
          for j in range(reader.shape[1]):
            if count == reader[i][j]:
              cropped_image_right = image[(y2_right-right_offset):y2_right-right_offset+(y1_right-y2_right+(right_offset*2)),(x1_right-right_offset):x1_right-right_offset+(x2_right-x1_right+(right_offset*2))]
              if j == 0 or j == 2:
                cv2.imwrite(path_r + str(i)+"_"+str(j)+"_open"+".png", cropped_image_right)
              if j == 1:
                cv2.imwrite(path_r + str(i)+"_"+str(j)+"_close"+".png", cropped_image_right)
        frame = frame + 1

#STAMPO IMMAGINI PER COLAB
        cropped_image_right_cb = image[(y2_right-right_offset):y2_right-right_offset+(y1_right-y2_right+(right_offset*2)),(x1_right-right_offset):x1_right-right_offset+(x2_right-x1_right+(right_offset*2))]
        resized_right = cv2.resize(cropped_image_right_cb, (96,96), interpolation=cv2.INTER_CUBIC)

        img_r = np.zeros((1,resized_right.shape[0], resized_right.shape[1], 3))  
        for i in range(3):  
          img_r[0,:,:,i] = resized_right
        #predictions = model.predict(img_r)

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        cv2.putText(image,"right:" + str(int(predictions[0][0])), (7, 550), cv2.FONT_HERSHEY_SIMPLEX,
                   3, (255, 0, 0), 3, cv2.LINE_AA)
        valore_array_dx.append(int(predictions[0][0]))
        print('\n')

#BOUNDING BOX
        #RETTANGOLO PIU ESTERNO 
        start_point = (x1_right-10,y1_right+10)
        end_point = (x2_right+10,y2_right-10)
        color = (0,255,0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        #RETTANGOLO PIU INTERNO
        start_point = (x1_right,y1_right)
        end_point = (x2_right,y2_right)
        color = (0,0,0)
        thickness = 1
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
    

#########################################################################################
#LIPS
        lips_keypoint=[]
        x_array3 = np.zeros((1,87))
        y_array3= np.zeros((1,87)) 
        for t,point in enumerate (landmark_subset_lips.landmark):
          lips_keypoint.append({
            'x':point.x,
            'y':point.y
          })
          x_array3[0,t] = point.x
          y_array3[0,t] = point.y
        
        lips_x_max = np.amax(x_array3) #x massima
        lips_x_min = np.amin(x_array3) #x minima
        lips_y_max = np.amax(y_array3) 
        lips_y_min = np.amin(y_array3)
        
        x1_lips = int(lips_x_min * (image.shape[1])) #1 larghezza
        y1_lips = int(lips_y_max * (image.shape[0])) #0 altezza
        x2_lips = int(lips_x_max * (image.shape[1]))
        y2_lips = int(lips_y_min * (image.shape[0]))

#STAMPO IMMAGINI         
        path_l = 'MODIFICATO/drivers/images/lips/lips' + str(id_sbj)+"_" + str(id_session) + "_"
        reader=np.loadtxt('count.csv', delimiter=',')
        for i in range(reader.shape[0]):
          for j in range(reader.shape[1]):
            if count == reader[i][j]:
              cropped_image_lips = image[(y2_lips-lips_offset):y2_lips-lips_offset+(y1_lips-y2_lips+(lips_offset*2)),(x1_lips-lips_offset):x1_lips-lips_offset+(x2_lips-x1_lips+(lips_offset*2))]
              cv2.imwrite(path_l + str(i)+"_"+str(j)+".png", cropped_image_lips)
        frame = frame + 1
#ASPECT RATIO LIPS INIZIALIZZATA A 0 SE CHIUSA , 1 SE VIENE APERTA
        ar_lips = round((y1_lips-y2_lips)/(x2_lips-x1_lips) , 2)
        if ar_lips > 0.50 :
          open_lips = 1 
        else : open_lips = 0
#BOUNDING BOX
        start_point = (x1_lips-10,y1_lips+10)
        end_point = (x2_lips+10,y2_lips-10)
        color = (0,255,0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        start_point = (x1_lips,y1_lips)
        end_point = (x2_lips,y2_lips)
        color = (0,0,0)
        thickness = 1
        cv2.putText(image,"open_lips:" + str(open_lips), (7, 700), cv2.FONT_HERSHEY_SIMPLEX,
                   3, (0, 255, 0), 3, cv2.LINE_AA)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
###################################################################################

#CREAZIONE JSON: INDICO I VARI ATTRIBUTI CHE MI INTERESSANO
        list_data_frame.append({
          'frame': count,
          'timeOfFrame':cap.get(cv2.CAP_PROP_POS_MSEC) /1000,
          'leftEye':left_keypoint,
          'rightEye': right_keypoint,
          'lips': lips_keypoint
          })
#########################################################################    
        
    cv2.imshow('MediaPipe FaceMesh', image)
    
#CLASSE MYSTATS  
    stats.AddLandmark(re)
    stats.AddLandmark(le)
    stats.AddLandmark(li)
    stats.ClearStat()
    if cv2.waitKey(1) & 0xFF == 27:
      break

#CREO/SCRIVO IL JSON 
  with open('list.json', 'w')as outline:
    json.dump(list_data_frame,outline)   
#####################################################################
#IN QUESTA SEZIONE VADO A STAMPARE DUE GRAFICI PER LA VISUALIZZAZIONE DELLA CHIUSURA E DELL'APERTURA DEGLI OCCHI, 
#ALCUNI VALORI POSSONO ESSERE DEI FALSI POSITIVI
lenght = len(valore_array_sx)

fig, ax = plt.subplots(1,2, figsize = (30 ,18))
ax[0].set_title('Left Eye')
ax[0].plot(range(lenght), valore_array_sx)                     #plotto da 0 fino alla dimensione finale 
ax[0].set_xlabel('Frame')
ax[0].set_ylabel('Apertura=1/Chiusura=0')

ax[1].set_title('Right Eye')
ax[1].plot(range(lenght), valore_array_dx)                      #plotto da 0 fino alla dimensione finale 
ax[1].set_xlabel('Frame')
ax[1].set_ylabel('Apertura=1/Chiusura=0')

fig.tight_layout(pad=10.0)                                                            #SPAZIO TRA UN GRAFICO E L ALTRO

plt.show()


