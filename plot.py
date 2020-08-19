import os
import glob
import cv2
import dlib
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cbook import boxplot_stats

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

def load_graph(frozen_graph_path):
    graph = tf.Graph()
    with tf.compat.v2.io.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it 
    with graph.as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

def faceFeatureExtract(graph,imageBatch,dim):
    img = np.ndarray(imageBatch.shape,dtype = np.float32)
    img[:,:,:,0] = (imageBatch[:,:,:,0] - 255.0/2) / (255.0/2)
    img[:,:,:,1] = (imageBatch[:,:,:,1] - 255.0/2) / (255.0/2)
    img[:,:,:,2] = (imageBatch[:,:,:,2] - 255.0/2) / (255.0/2)
    with graph.as_default():
        x = graph.get_tensor_by_name('input:0')
        embeddings = graph.get_tensor_by_name('embedding:0')
        cut_interval = 20
        with tf.compat.v1.Session(graph = graph) as sess:
            total_num = img.shape[0]
            emb_np = np.ndarray(shape=[total_num,dim], dtype=np.float32)
            cut_ind = np.arange(0,total_num,cut_interval)
            if cut_ind[-1] != total_num:
                cut_ind = np.append(cut_ind,total_num)
            for i in range (cut_ind.shape[0]-1):
                start = cut_ind[i]
                end = cut_ind[i+1]
                temp = sess.run(embeddings,feed_dict = {x:img[start:end]})
                # print ("temp.shape: {}".format(temp.shape))
                emb_np[start:end] = temp
            return emb_np
    
def compareSimilarity(featureFoo, featureBar):
    similarity = 1.00 - ((np.linalg.norm(featureFoo-featureBar))*0.50 - 0.20)
    return similarity

def plotGIBoxScatter(matchScore):
    #init variables
    genuineScore = []
    imposterScore = []
    genuineX = []
    imposterX = []
    X=[]
    Y=[]
    knownToUnknown = 0
    unknownToKnown = 0
    Known = 0
    Unknown = 0
    #label for same id(Genuine) and different id(Imposter)
    #pnas dataset
    for i, score in enumerate(matchScore):
        if i < 12: #Genuine
            genuineScore.append(score)
        else:      #Imposter
            imposterScore.append(score)
    #plot box and scatter pairs
    sns.set(style="whitegrid")
    dfG = pd.DataFrame(genuineScore,columns=['Score'])
    dfG['Labels'] = 'Genuine'
    dfI = pd.DataFrame(imposterScore,columns=['Score'])
    dfI['Labels'] = 'Imposter'
    frames = [dfG, dfI]
    result = pd.concat(frames)
    result.Score = pd.to_numeric(result.Score, errors='coerce')
    print(result)
    ax = sns.boxplot(x="Labels", y="Score", data=result, showfliers = False)
    ax = sns.swarmplot(x="Labels", y="Score", data=result, color=".25")
    plt.yticks(np.arange(0, 1, 0.2))
    plt.savefig('pnasGIboxPlot.png')
    plt.show()
    #print box plot status
    dfG.Score = pd.to_numeric(dfG.Score, errors='coerce')
    statsG = boxplot_stats(dfG['Score'])
    print('Genuine: ',statsG,'\n')
    print(statsG[0]['whislo'])
    dfI.Score = pd.to_numeric(dfI.Score, errors='coerce')
    statsI = boxplot_stats(dfI['Score'])
    print('Imposter: ',statsI)
    return statsG, statsI

#load models
detector = dlib.get_frontal_face_detector() #dlib FD model
sp = dlib.shape_predictor("geo_vision_5_face_landmarks.dat") #dlib LM model
g2 = load_graph("09-02_02-45.pb") #tensorflow FR resnet_v1_50 model

#load image list
imageFileList = loadImages("pnas/")
win = dlib.image_window()

#get face detected aligned crops
faceCrops = []
for i, f in enumerate(imageFileList):
    faceCrops.append(faceDetectCrop(f))
    cv2.imshow("faceCrop", faceCrops[-1])
    cropName = "crop/{}.jpg".format(i)
    cv2.imwrite(cropName,faceCrops[-1])
    cv2.waitKey(delay=1)

#inference FR
emb_dim = 512
print(np.shape(faceCrops))
arrFaceCropBatch = np.array(faceCrops)
embedding = faceFeatureExtract(g2,arrFaceCropBatch, emb_dim)
print(np.shape(embedding))

#save features to txt 512-D per person a row
with open('pnas.txt', 'wb') as f:
    np.savetxt(f, np.row_stack(embedding), fmt='%f')

#match pairs similarity score
similarityScores = []
print(len(embedding))
for i in range(int(len(embedding)*0.5)):
    similarityScores.append(compareSimilarity(embedding[2*i], embedding[2*i+1]))
print(similarityScores)
plotGIBoxScatter(similarityScores)