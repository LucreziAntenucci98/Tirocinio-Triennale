import csv
import numpy as np
import json
import cv2
import math

from time import process_time_ns
from numpy.lib import nanfunctions
import matplotlib.pyplot as plt
#DIMENSIONI DEGLI OCCHI (NEL FILE MYLANDMARKS CONTEGGIO I VALORI D'INTERESSE)
LEFT_EYE_DIM = 16
RIGHT_EYE_DIM = 16

#INTERVALLI IN CUI GLI OCCHI INIZIANO A CHIUDERSI, ISTANTE IN CUI SONO CHIUSI , E ISTANTE IN CUI SONO APERTI (Video Prof)
data=[
    [46,48,52],
    [136,139,143],
    [232,234,238],
    [282,285,290],
    [458,460,465],
    [520,522,528],
    [544,547,553],
    [597,599,604],
    [625,627,632],
    [700,702,707],
    [755,758,763],
    [768,771,775],
    [818,821,826],
    [849,851,856],
    [873,874,877],
    [915,917,922],
    [944,947,952],
    [974,976,979],
    [1041,1043,1048],
    [1129,1131,1136],
    [1231,1233,1237],
    [1270,1272,1277],
    [1294,1296,1299],
    [1384,1387,1391],
    [1403,1406,1410],
    [1432,1434,1438],
    [1458,1460,1464],
    [1477,1479,1484],
    [1500,1503,1509],
    [1609,1611,1618],
    [1626,1628,1632],
    [1701,1704,1708],
    [1773,1776,1780]
]
######################################################
#Video Lucrezia
'''data=[
    [43,44,46],
    [82,85,97],
    [207,209,219],
    [348,350,359]
    [399,400,402]'''
##########################################
#video Matteo
'''data=[
    [26,29,32],
    [48,51,54],
    [107,111,117],
    [200,203,209],
    [275,282,287]
    ]
    '''
#creo il csv
with open('count.csv',mode='w', newline='') as file:
    file=csv.writer(file)
    file.writerows(data)

#leggo il csv
with open('count.csv', mode='r') as file:
    reader = csv.reader(file)
    list = list(reader)
    

frames = np.zeros((len(list),2))

lenght_array = 0
for i in range(0,len(list)):
    frames[i,0] = int(list[i][0]) - 10          #salvo dentro la riga del frames il valore del csv -10(10 frame prima)
    frames[i,1] = int(list[i][2]) + 10          #salvo dentro la riga del frames il valore del csv +10(10 frame dopo)
    lenght_array = lenght_array + (int(list[i][2]) - int(list[i][0]) + 21)
    


with open('list.json','r') as outline:
     data = json.load(outline)

#INIZIALIZZO LE VARIBILI 
x_array = np.zeros((len(data) , LEFT_EYE_DIM))
y_array = np.zeros((len(data) , LEFT_EYE_DIM))

x_array2 = np.zeros((len(data) , RIGHT_EYE_DIM))
y_array2 = np.zeros((len(data) , RIGHT_EYE_DIM))

velocitaY = np.zeros((len(data) , 1))
velocitaX = np.zeros((len(data) , 1))

EAR_array1 = np.zeros((len(data) , 1))
EAR_array2 = np.zeros((len(data) , 1))
EAR_array3 = np.zeros((len(data) , 1))

asp_ratio1 = np.zeros((len(data) , 1))
asp_ratio2 = np.zeros((len(data) , 1))
#############################################################################################
c = 0
space = 0
space_prec = 0
time_prec = 0
time = 0
for i in data:
    frame = i['frame']
    time = i['timeOfFrame']
    if frame >= int(frames[c,0]) and frame <= int(frames[c,1]):
        for t,z in enumerate(i['leftEye']):
            x_array[frame-1,t] = z['x']
            y_array[frame-1,t] = z['y']
            
            if t == 5 or t==4 :                                 #prendo il landmarks n 4
                spaceY =  z['y'] - space_prec 
                spaceX =  z['x'] - space_prec                   #spazio-spazio prec
                time = i['timeOfFrame'] - time_prec
                if spaceY < 0 :
                    space = -space
                velocitaY[frame-1,0] = spaceY/time   
                velocitaX[frame-1,0] = spaceX/time              #salvo dentro a un vettore 
                space_precY = z['y']   
                space_precX = z['x']                            #spazio precedente aggiornato0
        if  frame==frames[c,1]:
            c = c + 1
            time = 0
    else :                                                      #questo else serve per tenere il passo della y
        for t,z in enumerate(i['leftEye']):
            if t == 5 or t==4 :
                space_precY = z['y']
                space_precX=z['x']
    time_prec = i['timeOfFrame']


for i in range (0,x_array.shape[0]):
    for y in range (0,x_array.shape[1]):
        if x_array[i,y] == 0:
            x_array[i,y] = np.nan

for i in range (0,y_array.shape[0]):
    for y in range (0,y_array.shape[1]):
        if y_array[i,y] == 0:
            y_array[i,y] = np.nan
####################################################################################
c = 0
for i in data:
    frame = i['frame']
    if frame >= int(frames[c,0]) and frame <= int(frames[c,1]):
        for t,y in enumerate(i['rightEye']):
            x_array2[frame-1,t] = y['x']
            y_array2[frame-1,t] = y['y']
        if frame==frames[c,1]:
            c = c + 1
        if t==4 or t==5 :
            spaceY=z['y'] - space_prec 
            spaceX=z['x'] - space_prec 
            time = i['timeOfFrame'] - time_prec
            velocitaY[frame-1,0] = spaceY/time   
            velocitaX[frame-1,0] = spaceX/time                  #salvo dentro a un vettore 
            space_precY = y['y']   
            space_precX = y['x']                                #spazio precedente aggiornato
        if  frame==frames[c,1]:
            c = c + 1
            time = 0
    else :                                                      #questo else serve per tenere il passo della y
        for t,y in enumerate(i['rightEye']):
            if t==4 or t==5 :
                space_precY = y['y']
                space_precX=y['x']
    time_prec = i['timeOfFrame']


#CALCOLO LA VELOCITA
for i in range (0,velocitaY.shape[0]):
    for y in range (0,velocitaY.shape[1]):
        if velocitaY[i,y] == 0:
            velocitaY[i,y] = np.nan

for i in range (0,velocitaX.shape[0]):
    for y in range (0,velocitaX.shape[1]):
        if velocitaX[i,y] == 0:
            velocitaX[i,y] = np.nan
########################################
for i in range (0,x_array2.shape[0]):
    for y in range (0,x_array2.shape[1]):
        if x_array2[i,y] == 0:
            x_array2[i,y] = np.nan

for i in range (0,y_array2.shape[0]):
    for y in range (0,y_array2.shape[1]):
        if y_array2[i,y] == 0:
            y_array2[i,y] = np.nan

#EYE BLINKING  RIGHT_EYE
for i in data:
    frame = i['frame']
    for t,z in enumerate(i['rightEye']):
        if t == 7:
            xr1 = z['x']
            yr1 = z['y']
        if t == 2:
            xr2 = z['x']
            yr2 = z['y']
        if t == 4:
            xr3 = z['x']
            yr3 = z['y']
        if t == 15:
            xr4 = z['x']
            yr4 = z['y']
        if t == 12:
            xr5 = z['x']
            yr5 = z['y']
        if t == 10:
            xr6 = z['x']
            yr6 = z['y']
#DISTANZA EUCLIDEA
    vert1 = math.sqrt(pow((xr2-xr6),2)+pow((yr2-yr6),2))
    vert2 = math.sqrt(pow((xr3-xr5),2)+pow((yr3-yr5),2))
    oriz = math.sqrt(pow((xr1-xr4),2)+pow((yr1-yr4),2))
    right_EAR = (vert1+vert2)/(2*oriz)
    EAR_array1[frame-1,0] = right_EAR

##################################################
#EYE BLINKING  LEFT_EYE
for i in data:
    frame = i['frame']
    for t,z in enumerate(i['leftEye']):
        if t == 7:
            xl1 = z['x']
            yl1 = z['y']
        if t == 2:
            xl2 = z['x']
            yl2 = z['y']
        if t == 4:

            xl3 = z['x']
            yl3 = z['y']
        if t == 15:
            xl4 = z['x']
            yl4 = z['y']
        if t == 12:
            xl5 = z['x']
            yl5 = z['y']
        if t == 10:
            xl6 = z['x']
            yl6 = z['y']
#DISTANZA EUCLIDEA
    vert1 = math.sqrt(pow((xl2-xl6),2)+pow((yl2-yl6),2))
    vert2 = math.sqrt(pow((xl3-xl5),2)+pow((yl3-yl5),2))
    oriz = math.sqrt(pow((xl1-xl4),2)+pow((yl1-yl4),2))
    left_EAR = (vert1+vert2)/(2*oriz)
    EAR_array2[frame-1,0] = left_EAR
#############################################################################à
for i in data:
    frame = i['frame']
    for t,z in enumerate(i['lips']):
        if t == 21:
            xr1 = z['x']
            yr1 = z['y']
        if t == 24:
            xr2 = z['x']
            yr2 = z['y']
        if t == 28:
            xr3 = z['x']
            yr3 = z['y']
        if t == 31:
            xr4 = z['x']
            yr4 = z['y']
        if t == 39:
            xr5 = z['x']
            yr5 = z['y']
        if t == 35:
            xr6 = z['x']
            yr6 = z['y']
#DISTANZA EUCLIDEA
    vert1 = math.sqrt(pow((xr2-xr6),2)+pow((yr2-yr6),2))
    vert2 = math.sqrt(pow((xr3-xr5),2)+pow((yr3-yr5),2))
    oriz = math.sqrt(pow((xr1-xr4),2)+pow((yr1-yr4),2))
    lips_EAR = (vert1+vert2)/(2*oriz)
    EAR_array3[frame-1,0] = lips_EAR

#CALCOLO L'ASPECT RATIO DEI DUE OCCHI
for i in data:
    frame = i['frame']
    for t,z in enumerate(i['rightEye']):
        if t == 15:
            xmin = z['x']
        if t == 7:
            xmax = z['x']
        if t == 11:
            ymin = z['y']
        if t == 3:
            ymax = z['y']

    ar = (ymax - ymin)/(xmax - xmin)
    asp_ratio1[frame-1,0] = ar

for i in data:
    frame = i['frame']
    for t,z in enumerate(i['leftEye']):
        if t == 7:
            xmin = z['x']
        if t == 15:
            xmax = z['x']
        if t == 11:
            ymin = z['y']
        if t == 3:
            ymax = z['y']
    ar = (ymax - ymin)/(xmax - xmin)
    asp_ratio2[frame-1,0] = ar
#####################################################################

#GRAFICI OCCHI
fig, ax = plt.subplots(7,2, figsize = (30 ,18))
ax[0,0].set_title('Left Eye X')
ax[0,0].plot(x_array[0:x_array.shape[0], 0:x_array.shape[1]])               #plotto da 0 fino alla dimensione finale 
ax[0,0].set_xlabel('Frame')
ax[0,0].set_ylabel('Position X')
ax[0,1].set_title('Left Eye Y')
ax[0,1].plot(y_array[0:x_array.shape[0], 0:x_array.shape[1]])
ax[0,1].set_ylabel('Position Y')
ax[0,1].set_xlabel('Frame')

ax[1,0].set_title('Right Eye X')
ax[1,0].plot(x_array2[0:x_array2.shape[0], 0:x_array2.shape[1]])            #plotto da 0 fino alla dimensione finale 
ax[1,0].set_xlabel('Frame')
ax[1,0].set_ylabel('Position X')
ax[1,1].set_title('Right Eye Y')
ax[1,1].plot(y_array2[0:x_array2.shape[0], 0:x_array2.shape[1]])
ax[1,1].set_ylabel('Position Y')
ax[1,1].set_xlabel('Frame')
#####################################################################################

#GRAFICI VELOCITA'

ax[2,0].set_title('Left Eye X')
ax[2,0].plot(velocitaX[0:velocitaX.shape[0], 0:velocitaX.shape[1]])                 #plotto da 0 fino alla dimensione finale 
ax[2,0].set_xlabel('Frame')
ax[2,0].set_ylabel('Speed X')

ax[2,1].set_title('Left Eye Y')
ax[2,1].plot(velocitaY[0:velocitaY.shape[0], 0:velocitaY.shape[1]])                 #plotto da 0 fino alla dimensione finale 
ax[2,1].set_xlabel('Frame')
ax[2,1].set_ylabel('Speed Y')


ax[3,0].set_title('Right Eye X')
ax[3,0].plot(velocitaX[0:velocitaX.shape[0], 0:velocitaX.shape[1]])                 #plotto da 0 fino alla dimensione finale 
ax[3,0].set_xlabel('Frame')
ax[3,0].set_ylabel('Speed X')

ax[3,1].set_title('Right Eye Y')
ax[3,1].plot(velocitaY[0:velocitaY.shape[0], 0:velocitaY.shape[1]])                 #plotto da 0 fino alla dimensione finale 
ax[3,1].set_xlabel('Frame')
ax[3,1].set_ylabel('Speed Y')

###############################################################################################

#GRAFICI BLINKING

ax[4,0].set_title('Right eye blink detection')
ax[4,0].plot(EAR_array1[0:EAR_array1.shape[0], 0:EAR_array1.shape[1]])              #plotto da 0 fino alla dimensione finale 
ax[4,0].set_xlabel('Frame')
ax[4,0].set_ylabel('EAR right eye')

ax[4,1].set_title('Left eye blink detection')
ax[4,1].plot(EAR_array2[0:EAR_array2.shape[0], 0:EAR_array2.shape[1]])               #plotto da 0 fino alla dimensione finale 
ax[4,1].set_xlabel('Frame')
ax[4,1].set_ylabel('EAR left eye')


ax[5,0].set_title('Lips blink detection')
ax[5,0].plot(EAR_array3[0:EAR_array3.shape[0], 0:EAR_array3.shape[1]])              #plotto da 0 fino alla dimensione finale 
ax[5,0].set_xlabel('Frame')
ax[5,0].set_ylabel('EAR lips')

#GRAFICI ASPECT RATIO
ax[5,1].set_title('Right eye aspect ratio')
ax[5,1].plot(asp_ratio1[0:asp_ratio1.shape[0], 0:asp_ratio1.shape[1]])              #plotto da 0 fino alla dimensione finale 
ax[5,1].set_xlabel('Frame')
ax[5,1].set_ylabel('aspect ratio right eye')

ax[6,0].set_title('Left eye aspect ratio')
ax[6,0].plot(asp_ratio2[0:asp_ratio2.shape[0], 0:asp_ratio2.shape[1]])              #plotto da 0 fino alla dimensione finale 
ax[6,0].set_xlabel('Frame')
ax[6,0].set_ylabel('aspect ratio left eye')

fig.tight_layout(pad=10.0)                                                          #SPAZIO TRA UN GRAFICO E L ALTRO

plt.show()