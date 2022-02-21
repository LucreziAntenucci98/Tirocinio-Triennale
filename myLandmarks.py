
#face landmarks

#lips
lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291] 
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291] 
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308] 
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308] 
lipsup1 = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306]
lipslow1 = [76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306]
lipsup2 = [62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292]
lipslow2 = [62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292]
LIPS = lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner + lipsup1 + lipslow1 + lipsup2 + lipslow2

#right eye
rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
R_EYE = rightEyeUpper0 + rightEyeLower0  

#left eye
leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
L_EYE = leftEyeUpper0 + leftEyeLower0  

midwayBetweenEyes = [168]
noseTip = [1]
noseBottom = [2]
noseRightCorner = [98]
noseLeftCorner = [327]
rightCheek = [205]
leftCheek = [425]

#all landmarks
TOT_LANDMARKS = []
for i in range(468):
        TOT_LANDMARKS.append(i)

KEYPOINTS = L_EYE + R_EYE + LIPS
SILHOUETTE = [x for x in TOT_LANDMARKS if x not in KEYPOINTS]

#all landmarks
LANDMARKS = L_EYE+R_EYE+LIPS+SILHOUETTE

LIPS_CONNECTIONS = frozenset([
(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),(314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
(40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
(87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
(312, 311), (311, 310), (310, 415), (415, 308) ])
    
LEFTEYE_CONNECTIONS = frozenset([# Left eye.
(263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), (263, 466), (466, 388), (388, 387),
(387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
# Left eyebrow.
(276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336) ])

RIGHTEYE_CONNECTIONS = frozenset([# Right eye.
(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
(159, 158), (158, 157), (157, 173), (173, 133),
# Right eyebrow.
(46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)])

FACE_CONNECTIONS = frozenset([# Face oval.
(10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288),
(288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150),
(150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67),
(67, 109), (109, 10) ])

def recognizeLandmark(values):
    for x in L_EYE:
        if(x == values):
             return "leftEye"
    for x in R_EYE:
        if(x == values):
            return "rightEye"
    for x in LIPS:
        if(x == values):
            return "lips"
    for x in SILHOUETTE:
        if(x == values):
            return "silhouette"         

            
#colours
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (0, 255, 255)
BLUE_COLOR = (255, 0, 0)

