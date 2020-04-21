# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:37:44 2019

@author: Jayita
"""

import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

#webcam initiated
webcam = cv2.VideoCapture("rush.mp4")
#width=webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
#height=webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)

#output = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (int(width),int(height)))

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        break

    # apply object detection
    bbox, label, conf = cv.detect_common_objects(frame, confidence=.50, model='yolov3-tiny')
    #print(bbox, label, conf)
    # draw bounding box over detected objects
    out = draw_bbox(frame, bbox, label, conf, write_conf=True)
    
    # display output
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(out,'Detected Vehicles: ' + str(label.count('car')),(10, 35), font ,0.8,(0, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("Real-time object detection", out)
    #output.write(out)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
