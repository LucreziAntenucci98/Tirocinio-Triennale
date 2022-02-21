import json
import matplotlib.pyplot as plt
import numpy as np
import time

from numpy.lib.arraypad import pad

#DIMESIONI CORRISPETTIVE
LEFT_EYE_DIM = 16
RIGHT_EYE_DIM = 16
LIPS_DIM = 89

#LEGGO IL JSON
with open('list.json','r') as outline:
    data = json.load(outline)

#LEGGO I VALORI X,Y DI LEFT EYE DAL JSON E INIZIALIZZO DUE ARRAY
x_array = np.zeros((len(data), LEFT_EYE_DIM))
y_array = np.zeros((len(data), LEFT_EYE_DIM))
for i in data:
    frame = i['frame']
    for t,y in enumerate(i['leftEye']):
        x_array[frame-1,t] = y['x']
        y_array[frame-1,t] = y['y']

#STESSO DISCORSO PER RIGHT EYE
x_array2 = np.zeros((len(data), RIGHT_EYE_DIM))
y_array2 = np.zeros((len(data), RIGHT_EYE_DIM))
for i in data:
    frame = i['frame']
    for t,y in enumerate(i['rightEye']):
        x_array2[frame-1,t] = y['x']
        y_array2[frame-1,t] = y['y']

#UGUALE PER LIPS
x_array3 = np.zeros((len(data), LIPS_DIM))
y_array3 = np.zeros((len(data), LIPS_DIM))
for i in data:
    frame = i['frame']
    for t,y in enumerate(i['lips']):
        x_array3[frame-1,t] = y['x']
        y_array3[frame-1,t] = y['y']

#GRAFICI DEGLI OCCHI E LABBRA
fig, ax = plt.subplots(3,2, figsize = (30,18))
ax[0,0].set_title('Left Eye X')
ax[0,0].plot(x_array[0:x_array.shape[0], 0:x_array.shape[1]]) #plotto da 0 fino alla dimensione finale 
ax[0,0].set_xlabel('Frame')
ax[0,0].set_ylabel('Position X')

ax[0,1].set_title('Left Eye Y')
ax[0,1].plot(y_array[0:x_array.shape[0], 0:x_array.shape[1]])
ax[0,1].set_ylabel('Position Y')
ax[0,1].set_xlabel('Frame')
#-----------------------------------------------------------------------

ax[1,0].set_title('Right Eye X')
ax[1,0].plot(x_array2[0:x_array2.shape[0], 0:x_array2.shape[1]]) #plotto da 0 fino alla dimensione finale 
ax[1,0].set_xlabel('Frame')
ax[1,0].set_ylabel('Position X')

ax[1,1].set_title('Right Eye Y')
ax[1,1].plot(y_array2[0:x_array2.shape[0], 0:x_array2.shape[1]])
ax[1,1].set_ylabel('Position Y')
ax[1,1].set_xlabel('Frame')
#-------------------------------------------------------------------------------

ax[2,0].set_title('Lips X')
ax[2,0].plot(x_array3[0:x_array3.shape[0], 0:x_array3.shape[1]]) #plotto da 0 fino alla dimensione finale 
ax[2,0].set_xlabel('Frame')
ax[2,0].set_ylabel('Position X')

ax[2,1].set_title('Lips Y')
ax[2,1].plot(y_array3[0:x_array3.shape[0], 0:x_array3.shape[1]])
ax[2,1].set_ylabel('Position Y')
ax[2,1].set_xlabel('Frame')

fig.tight_layout(pad=10.0)
plt.show()