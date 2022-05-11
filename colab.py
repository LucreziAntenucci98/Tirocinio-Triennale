import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
import glob
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

#model = keras.models.load_model('MODIFICATO/my_eye_model')          #IMPORTO IL MODELLO
model = keras.models.load_model('my_eye_model') 

frame = False
list_of_dirs = glob.glob('MODIFICATO/foto_Colab/*')
count = 1
for item in list_of_dirs:
  if frame:
    frame = False
  else : frame = True
  
  img = cv2.imread(item,cv2.IMREAD_GRAYSCALE) #you can skif if you have in memory a grayscale image...
  resized = cv2.resize(img, (96,96), interpolation=cv2.INTER_CUBIC)
 
  img_r = np.zeros((1,resized.shape[0], resized.shape[1], 3))  
  for i in range(3):  
    img_r[0,:,:,i] = resized
    
  predictions = model.predict(img_r)
  
  if(frame):
    print('frame', count)
    count = count + 1
  # Apply a sigmoid since our model returns logits
  predictions = tf.nn.sigmoid(predictions)
  #print(predictions)
  predictions = tf.where(predictions < 0.5, 0, 1)
  cv2.imshow("ciao:",resized)
  print('Predictions:\n', predictions.numpy())
  if frame == False:
    print ('\n')
  