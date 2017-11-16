# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import cv2
import math

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x = x - x.mean()
    x = x/(x.std()+1e-5)
    x = x*0.1
    
    # clip to [0, 1]
    x = x + 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def upsample_image(x, new_size):
    #used Bilinear Interpolation    
    #x = x[:,:,0]
    x = np.expand_dims(x, axis = 2)
    resized_image = tf.Session().run(tf.image.resize_images( tf.cast(x, tf.float32), new_size))
    resized_image = resized_image[:,:,0]
    return resized_image

def empty_dump_folder(folder_path):
    
    for the_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
            
def get_gabor_image(img):
    kSize = 7
    sigma_value = 2
    theta_value = 0
    lambda_value = 8
    gamma_value = 2    
    '''
    kernel = cv2.getGaborKernel( (kSize, kSize), sigma_value, 0, lambda_value, gamma_value)
    gabor_image_0 = cv2.filter2D(img,cv2.CV_8UC3,kernel)    

    kernel = cv2.getGaborKernel( (kSize, kSize), sigma_value, 90, lambda_value, gamma_value)
    gabor_image_90 = cv2.filter2D(img,cv2.CV_8UC3,kernel)
    '''
    gabor_image = []
    for lambda_value in range(5, 50, 10):
        for theta_value in range(0, 90, 45):
            kernel = cv2.getGaborKernel( (kSize, kSize), sigma_value, 90, lambda_value, gamma_value)
            gabor_image_temp = cv2.filter2D(img,cv2.CV_8UC3,kernel)    
            gabor_image.append(gabor_image_temp)
        
    #gabor_image = [gabor_image_0, gabor_image_45, gabor_image_90]
    return gabor_image

def get_block_level_feature(block):
    #print ("[INFO] get_block_level_feature()" )
    feature_block_width = (int)(float(np.shape(block[0])[1])/6)
    feature_block_height = (int)(float(np.shape(block[0])[0])/6)
    
    feature_list = []
    for h in range(0, 6):
        for w in range(0, 6):
            for i in range(0, np.shape(block)[0]):
            #print ( str(h) + "_" + str(w))
                temp_image1 = block[i]
                temp = temp_image1[h*feature_block_height : (h+1)*feature_block_height, w*feature_block_width : (w+1)*feature_block_width]
                feature_list.append(np.mean(temp))
                feature_list.append(np.std(temp))
    
    return feature_list

def get_gabor_feature(img):
    gabor_image = get_gabor_image(img)
    feature_list = get_block_level_feature(gabor_image)
    
    return feature_list

def extract_roi(gt_file):
    gtFile = open(gt_file, 'r')
    left = 0
    top = 0
    right = 0
    bottom = 0
    count = 0
    listOfRois = []
    for line in gtFile:
        #print (line)
        count = 0
        for word in line.split():
            #print (word)
            if(count==0):
                left = int(word)
            elif(count==1):
                top = int(word)
            elif(count==2):
                right = int(word)
            elif(count==3):
                bottom = int(word)        
            count = count + 1
            
        #print (left)
        print (str(left) + "__" + str(top) + "__" + str(right) + "__" + str(bottom))
        roi = [left, top, right, bottom]
        listOfRois.append(roi)
    gtFile.close()   
    return listOfRois 

def cluster_rois(list_of_rois, img_height, img_width):
    nsize = np.shape(list_of_rois)[0]
    
    roi_count = 0
    list_of_clusters = []
    while (nsize>0):
        #roi_count = roi_count + 1
        roi = list_of_rois[0]
        roi_centroid_x = int((roi[0] + roi[2])/2)
        roi_centroid_y = int((roi[1] + roi[3])/2)
        
        r_count = -1
        minDist = img_width
        minIndex = 0
        for r in list_of_rois: 
            r_count = r_count + 1
            if(r_count == roi_count):
                continue
            else:
                r_centroid_x = int((r[0] + r[2])/2)
                r_centroid_y = int((r[1] + r[3])/2)
                
                dist = ((r_centroid_x - roi_centroid_x)*(r_centroid_x - roi_centroid_x)
                        + (r_centroid_y - roi_centroid_y)*(r_centroid_y - roi_centroid_y))
                dist = math.sqrt(dist)
                
                if (dist<minDist):
                    minDist = dist
                    minIndex = r_count
        
        if (minDist< (0.25*img_width)):
            list_of_rois[0][0] = min(list_of_rois[0][0], list_of_rois[minIndex][0])
            list_of_rois[0][1] = min(list_of_rois[0][1], list_of_rois[minIndex][1])
            list_of_rois[0][2] = max(list_of_rois[0][2], list_of_rois[minIndex][2])
            list_of_rois[0][3] = max(list_of_rois[0][3], list_of_rois[minIndex][3])
            
            if(nsize==1):
                temp = [list_of_rois[0][0], list_of_rois[0][1], list_of_rois[0][2], list_of_rois[0][3]]
                list_of_clusters.append(temp)
            
            del list_of_rois[minIndex]
            #roi_count = roi_count + 1
        else:
            temp = [list_of_rois[0][0], list_of_rois[0][1], list_of_rois[0][2], list_of_rois[0][3]]
            list_of_clusters.append(temp)
            del list_of_rois[0]
        nsize = np.shape(list_of_rois)[0]
            
    return list_of_clusters
            
            
            

def extract_roi_testdis(gtFilePath):
    gtFile = open(gtFilePath, 'r')
    count = 0
    listOfRois = []

    for line in gtFile:
        #print (line)
        listOfX = []
        listOfY = []
        count = 0
        for word in line.split():
            #print (word)
            if(count%2==0 and count<8):
                listOfX.append(word)
            elif(count<8):
                listOfY.append(word)     
            count = count + 1
            
        #print (left)
        #print (str(left) + "__" + str(top) + "__" + str(right) + "__" + str(bottom))
        listOfX = np.array(listOfX, dtype=np.int64)
        listOfY = np.array(listOfY, dtype=np.int64)
        left = np.amin(listOfX)
        top = np.amin(listOfY)
        right = np.amax(listOfX)
        bottom = np.amax(listOfY)
        roi = [ left, top, right, bottom]
        listOfRois.append(roi)
    gtFile.close()   
    return listOfRois


    
    
