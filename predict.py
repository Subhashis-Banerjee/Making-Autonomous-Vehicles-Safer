#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:16:39 2019

@author: subhashis
"""


from imageai.Detection import ObjectDetection
import pandas as pd
import os
import numpy as np
import cv2
import random

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# reading test images
df = pd.read_excel('Test_set.xlsx', sheet_name=0)
test_images = []
for index, row in df.iterrows():
    test_images.append(row['Image_name'])

##### Models
moels_weight_path = 'model_weights'
data_path = 'IDD_Detection/JPEGImages'
out_path = 'predict_RetinaNet'
out_path1 = 'predict_refined_RetinaNet'

detector = ObjectDetection()
#detector.setModelTypeAsYOLOv3()
#detector.setModelTypeAsTinyYOLOv3()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(moels_weight_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
out_id=1
for img in test_images:
    detections = detector.detectObjectsFromImage(input_image=os.path.join(
            data_path , img), output_image_path=os.path.join(
                    out_path , str(out_id)+'.jpg'), minimum_percentage_probability=30)

#    detections = refine(detections)
    for index in range(len(detections)):
        if index >= len(detections):
            break
        ob1 = detections[index]
        cmp = [i for i, n in enumerate(range(len(detections))) if n != index]
        
        for k in cmp:
            ob2 = detections[k]
            if ((ob1["name"] == 'motorcycle' and ob2["name"] == 'person') or 
            (ob1["name"] == 'person' and ob2["name"] == 'motorcycle')):
                [ax1, ay1, ax2, ay2] = ob1["box_points"]
                [bx1, by1, bx2, by2] = ob2["box_points"]
                
                bb1 = {'x1':ax1, 'x2':ax2, 'y1':ay1, 'y2':ay2}
                bb2 = {'x1':bx1, 'x2':bx2, 'y1':by1, 'y2':by2}
                
                overlap = get_iou(bb1, bb2)
                
                if overlap > 0.2:
                    union = np.array([min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)])
                    detections.pop(k)
                    
                    detections[index]['box_points'] = union
                    detections[index]['name'] = 'rider'
                    break
#                if (ax1 > bx1) or ()
#                overlap = calculate_overlap = (ob1["box_points"], ob2["box_points"])
                
    image = cv2.imread(os.path.join(data_path, img))
    for eachObject in detections:
        disp1 = random.randint(0,50)
        disp2 = random.randint(0,50)
        clr1 = random.randint(0,255)
        clr2 = random.randint(0,255)
        clr3 = random.randint(0,255)
        p = eachObject["box_points"]
        cv2.rectangle(image,(p[0], p[1]),(p[2], p[3]),(clr1,clr2,clr3),3)
        cv2.putText(image,eachObject["name"],((p[0]+p[2]+disp1)//2, (p[1]+p[3]+disp2)//2), 4, 1, (0,255,0), 2)
        cv2.imwrite(os.path.join(out_path1 , str(out_id)+'.jpg'), image)
        
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
        
    out_id = out_id +1
    
    
    
    
    
    
    
    