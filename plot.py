import os
import glob
import cv2
import dlib
import numpy as np
from PIL import Image
import tensorflow as tf

def loadImages(fileDirectory):
    # Load from a file
    files = glob.glob(fileDirectory+"*.ppm")
    n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.ppm')[0], files)]
    files = [x for (y, x) in sorted(zip(n, files))]
    return files

def faceDetectCrop(imageFile, size = 112, padding = 0.25):
    # Now process all the images
    print("Processing file: {}".format(imageFile))
    im_cv = cv2.imread(imageFile)
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)
        # Let's generate the aligned image using get_face_chip
        face_chip = dlib.get_face_chip(img, shape, size=size, padding=padding)
        face_chip = cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR)
    return face_chip   

def faceFeatureExtract(faceImage):
    feature = []
    return feature

def compareSimilarity(featureFoo, featureBar):
    similarity = 0.0
    return similarity

#load models
detector = dlib.get_frontal_face_detector() #dlib FD model
sp = dlib.shape_predictor("geo_vision_5_face_landmarks.dat") #dlib LM model

#load image list
imageFileList = loadImages("pnas/")
win = dlib.image_window()

#get face detected aligned crops
faceCrops = []
for f in imageFileList:
    faceCrops.append(faceDetectCrop(f))
    cv2.imshow("faceCrop", faceCrops[-1])
    cv2.waitKey()




