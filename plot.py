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

def loadDataframe(fileDirectory):
    # Load from a file
    enrolldir = fileDirectory + "/enroll.txt"
    verifdir = fileDirectory + "/verif.txt"
    matchdir = fileDirectory + "/match.txt"
    dfEnroll = pd.read_csv(enrolldir, delimiter= '\s+', header = None, names=["id", "path", "dataset"])
    dfVerif = pd.read_csv(verifdir, delimiter= '\s+', header = None, names=["id", "path", "dataset"]) 
    dfMatch = pd.read_csv(matchdir, delimiter= '\s+', header = None, names=["enroll", "verif"])
    # Assign headers
    dfEnroll['path'] = dfEnroll['path'].str.strip("../")
    dfVerif['path'] = dfVerif['path'].str.strip("../")
    dfMatch['enroll'] = dfMatch['enroll'].str[:-9]
    dfMatch['verif'] = dfMatch['verif'].str[:-9]
    # Get UUID
    imgEnrollName = ([p.strip().split('/')[2] for p in dfEnroll['path']])
    noExtEnrollName = [n.strip(".ppm") for n in imgEnrollName]
    enrollUUID = ([n.strip().split('-')[0] for n in noExtEnrollName]) 
    dfEnroll['UUID'] = enrollUUID
    imgVerifName = ([p.strip().split('/')[2] for p in dfVerif['path']])
    noExtVerifName = [n.strip(".ppm") for n in imgVerifName]
    verifUUID = ([n.strip().split('-')[0] for n in noExtVerifName]) 
    dfVerif['UUID'] = verifUUID
    # Assign result columns
    dfEnroll['features'] = np.nan
    dfVerif['features'] = np.nan
    dfMatch['score'] = np.nan
    dfMatch['GIlabel'] = np.nan
    dfEnroll['features'] = dfEnroll['features'].astype('object')
    dfVerif['features'] = dfVerif['features'].astype('object')
    dfMatch['score'] = dfMatch['score'].astype('object')
    dfMatch['GIlabel'] = dfMatch['GIlabel'].astype('object')
    # Unable to detect a face in the image
    dfEnroll['FaceDetectionError'] = False
    dfVerif['FaceDetectionError'] = False
    # Either or both of the input templates were result of failed feature extraction
    dfMatch['VerifTemplateError'] = np.nan 
    dfMatch['VerifTemplateError'] = dfMatch['VerifTemplateError'].astype('object')
    print(dfEnroll.head())
    print(dfVerif.head())
    print(dfMatch.head())
    return dfEnroll,dfVerif,dfMatch

def faceDetectCrop(inputDataframe, size = 112, padding = 0.25):
    # Now process the image
    print("Processing file: {}".format(inputDataframe['path']))
    im_cv = cv2.imread(inputDataframe['path'].item())
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    # win.clear_overlay()
    # win.set_image(img)

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
        # win.clear_overlay()
        # win.add_overlay(d)
        # win.add_overlay(shape)
        # Let's generate the aligned image using get_face_chip
        face_chip = dlib.get_face_chip(img, shape, size=size, padding=padding)
        face_chip = cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR)
    if len(dets) == 0:
        bFDSuccess = False
        face_chip = np.zeros((size,size,3), np.uint8)
    else:
        bFDSuccess = True
    return face_chip, bFDSuccess

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

def plotGIBoxScatter(dfMatch):
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
    # ax = sns.boxplot(x="GIlabel", y="score", data=dfMatch, showfliers = False)
    ax = sns.swarmplot(x="GIlabel", y="score", data=dfMatch, hue="VerifTemplateError")
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid()
    plt.savefig('GIboxPlot.png')
    plt.show()
    #print box plot status
    # dfG.Score = pd.to_numeric(dfG.Score, errors='coerce')
    statsG = boxplot_stats(dfMatch['score'])
    print('Genuine: ',statsG,'\n')
    print(statsG[0]['whislo'])
    # dfI.Score = pd.to_numeric(dfI.Score, errors='coerce')
    statsI = boxplot_stats(dfMatch['score'])
    print('Imposter: ',statsI)
    return statsG, statsI

#load models
detector = dlib.get_frontal_face_detector() #dlib FD model
sp = dlib.shape_predictor("geo_vision_5_face_landmarks.dat") #dlib LM model
g2 = load_graph("09-02_02-45.pb") #tensorflow FR resnet_v1_50 model

#load image list
dfEnroll, dfVerif, dfMatch = loadDataframe('mugshotInput')
# dfEnroll, dfVerif, dfMatch = loadDataframe('pnasInput')
# win = dlib.image_window()

#for debug use
# dfEnroll = dfEnroll[:10]
# dfVerif = dfVerif[:10]
# dfMatch = dfMatch[:10]

#get face detected aligned crops
enrollCrops = []
verifCrops = []


for i in range(len(dfVerif)):
    cropEnroll, bEnrollFDSuccess = faceDetectCrop(dfEnroll.iloc[[i]])
    if bEnrollFDSuccess:
        dfEnroll.at[i,'FaceDetectionError']= False
    else:
        dfEnroll.at[i,'FaceDetectionError']= True
    enrollCrops.append(cropEnroll)
    # cv2.imshow("faceCrop", enrollCrops[-1])
    enrollName = "{}_crop/{}_enroll.jpg".format(dfEnroll.iloc[[i]].dataset.item(), i)
    cv2.imwrite(enrollName,enrollCrops[-1])
    # cv2.waitKey(delay=1)
    cropVerif, bVerifFDSuccess = faceDetectCrop(dfVerif.iloc[[i]])
    if bVerifFDSuccess:
        dfVerif.at[i,'FaceDetectionError']= False
    else:
        dfVerif.at[i,'FaceDetectionError']= True
    verifCrops.append(cropVerif)
    # cv2.imshow("faceCrop", verifCrops[-1])
    verifName = "{}_crop/{}_verif.jpg".format(dfVerif.iloc[[i]].dataset.item(), i)
    cv2.imwrite(verifName,verifCrops[-1])
    # cv2.waitKey(delay=1)

#inference FR
emb_dim = 512
embedding = []

arrEnrollCropBatch = np.array(enrollCrops)
enrollEmbedding = faceFeatureExtract(g2,arrEnrollCropBatch, emb_dim)

arrVerifCropBatch = np.array(verifCrops)
verifEmbedding = faceFeatureExtract(g2,arrVerifCropBatch, emb_dim)
 
for i in range(len(dfVerif)):
    if dfEnroll.at[i,'FaceDetectionError'] == False:
        dfEnroll.at[i, 'features'] = enrollEmbedding[i]
    if dfVerif.at[i,'FaceDetectionError'] == False:
        dfVerif.at[i, 'features'] = verifEmbedding[i]

#save features to txt 512-D per person a row
# txtFileName = dataset + '.txt'
# with open(txtFileName, 'wb') as f:
#     np.savetxt(f, np.row_stack(embedding), fmt='%f')

#match pairs similarity score
for i in range(len(dfMatch)):
    enrollId = dfMatch.at[i, 'enroll']
    verifId = dfMatch.at[i, 'verif']
    filterEnroll = dfEnroll[dfEnroll['id'] == int(enrollId)]
    filterVerif = dfVerif[dfVerif['id'] == int(verifId)]
    featureEnroll = filterEnroll['features'].item()
    featureVerif = filterVerif['features'].item()
    if dfEnroll.at[i, 'FaceDetectionError'] == True or dfVerif.at[i, 'FaceDetectionError'] == True:
        dfMatch.at[i, 'score'] = 0
        dfMatch.at[i, 'VerifTemplateError'] = 'FaceDetectionError'
    else:
        dfMatch.at[i,'score'] = compareSimilarity(featureEnroll,featureVerif)
        dfMatch.at[i, 'VerifTemplateError'] = 'MatchSuccess'
    if dfEnroll.at[i, 'UUID'] == dfVerif.at[i, 'UUID']:
        dfMatch.at[i, 'GIlabel'] = 'Genuine'
    else:
        dfMatch.at[i, 'GIlabel'] = 'Imposter'
plotGIBoxScatter(dfMatch)