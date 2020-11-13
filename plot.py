import os
import glob
import cv2
import dlib
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm
import torch
from torchvision import transforms
import backbone.model_irse_org
from backbone.model_irse import IR_SE_101, l2_norm
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
    # uncertain_score
    dfEnroll['uncertain_score'] = np.nan
    dfVerif['uncertain_score'] = np.nan
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

def faceFeatureExtract(image, data_transform, device, backbone_model, dul_model, flip=False):
    img_tensor = read_img(image, data_transform, flip)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        x, _ = backbone_model(img_tensor)
        x = l2_norm(x)
        mu, sigma = dul_model(img_tensor)
        mu = l2_norm(mu)
        sigma = torch.exp(sigma)
        sigma = sigma.cpu().data.numpy().reshape(-1)

        return x, mu, sigma

    
def compareSimilarity(featureFoo, featureBar):
    # similarity = 1.00 - ((np.linalg.norm(featureFoo-featureBar))*0.50 - 0.20)
    similarity = np.dot(featureFoo, featureBar) / (np.linalg.norm(featureFoo) * np.linalg.norm(featureBar))
    return similarity


def compareUncertainty(mu1, var1, mu2, var2):
    mu_diff = np.sqrt(np.sum((mu1 - mu2) ** 2))
    # print(np.sqrt(mu1 ** 2 - mu2 ** 2))
    var_sum = var1 + var2
    score = 0.5 * (mu_diff / var_sum + np.log(var_sum))
    return score


def read_img(img, transform, flip=False):
    pil_image = Image.fromarray(img)
    if flip:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    image_tensor = transform(pil_image)
    return image_tensor


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
    ax = sns.boxplot(x="GIlabel", y="score", data=dfMatch, showfliers = False)
    ax = sns.swarmplot(x="GIlabel", y="score", data=dfMatch, color='.2', hue="VerifTemplateError")
    # ax = sns.boxplot(x="GIlabel", y="uncertainty", data=dfMatch, showfliers=False)
    # ax = sns.swarmplot(x="GIlabel", y="uncertainty", data=dfMatch, color='.2', hue="VerifTemplateError")
    # plt.yticks(np.arange(0, 1, 0.05))
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


def plotGIUncertain(dfEnroll, dfVerif, dfMatch):
    dfEnroll_ = dfEnroll[:len(dfVerif)]

    Genuine = dfEnroll_[dfMatch['GIlabel'] == 'Genuine']['uncertain_score'].append(dfVerif[dfMatch['GIlabel'] == 'Genuine']['uncertain_score'], ignore_index=True)
    Imposter = dfEnroll_[dfMatch['GIlabel'] == 'Imposter']['uncertain_score'].append(dfVerif[dfMatch['GIlabel'] == 'Imposter']['uncertain_score'], ignore_index=True)
    sns.distplot(Genuine, label='Genuine', hist=False)
    sns.distplot(Imposter, label='Imposter', hist=False)
    #label for same id(Genuine) and different id(Imposter)
    # ax = sns.boxplot(x="GIlabel", y="score", data=dfMatch, showfliers = False)
    # ax = sns.swarmplot(x="GIlabel", y="score", data=dfMatch, color='.2', hue="VerifTemplateError")
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.grid()
    plt.savefig('plotGIUncertain.png')
    plt.show()
    # #print box plot status
    # # dfG.Score = pd.to_numeric(dfG.Score, errors='coerce')
    # statsG = boxplot_stats(dfMatch['score'])
    # print('Genuine: ',statsG,'\n')
    # print(statsG[0]['whislo'])
    # # dfI.Score = pd.to_numeric(dfI.Score, errors='coerce')
    # statsI = boxplot_stats(dfMatch['score'])
    # print('Imposter: ',statsI)
    # return statsG, statsI


def main(dataset, crop_dir, load=False):
    # load image list
    dfEnroll, dfVerif, dfMatch = loadDataframe(dataset)

    # for debug use
    # dfEnroll = dfEnroll[:10]
    # dfVerif = dfVerif[:10]
    # dfMatch = dfMatch[:10]
    # mugshotInput protocol has bug , should be follow up len(dfVerif)
    # print(len(dfEnroll))
    # print(len(dfVerif))
    #
    # assert len(dfEnroll) == len(dfVerif)

    # get face detected aligned crops
    enrollCrops = []
    verifCrops = []

    for i in range(len(dfVerif)):
        file_name = dfEnroll.iloc[[i]]['path'].item().split('/')[-1].replace('ppm', 'jpg')
        # print(file_name)

        crop_img_path = os.path.join(crop_dir, file_name)
        cropEnroll = None
        if os.path.isfile(crop_img_path):
            dfEnroll.at[i, 'FaceDetectionError'] = False
            cropEnroll = cv2.imread(crop_img_path)
        else:
            dfEnroll.at[i, 'FaceDetectionError'] = True

        enrollCrops.append(cropEnroll)

        file_name = dfVerif.iloc[[i]]['path'].item().split('/')[-1].replace('ppm', 'jpg')
        # print(file_name)

        crop_img_path = os.path.join(crop_dir, file_name)
        cropVerif = None
        if os.path.isfile(crop_img_path):
            dfVerif.at[i, 'FaceDetectionError'] = False
            cropVerif = cv2.imread(crop_img_path)
        else:
            dfVerif.at[i, 'FaceDetectionError'] = True

        verifCrops.append(cropVerif)

    if os.path.isfile('Enroll_embeddings_org_list_' + dataset + '.npy'):
        Enroll_embeddings_org_list = np.load('Enroll_embeddings_org_list_' + dataset + '.npy')
        Enroll_embeddings_mu_list = np.load('Enroll_embeddings_mu_list_' + dataset + '.npy')
        Enroll_embeddings_sigma_list = np.load('Enroll_embeddings_sigma_list_' + dataset + '.npy')
        Verif_embeddings_org_list = np.load('Verif_embeddings_org_list_' + dataset + '.npy')
        Verif_embeddings_mu_list = np.load('Verif_embeddings_mu_list_' + dataset + '.npy')
        Verif_embeddings_sigma_list = np.load('Verif_embeddings_sigma_list_' + dataset + '.npy')
    else:
        #inference FR
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = 1
        INPUT_SIZE = [112, 112]
        backbone_model_path = './Backbone_IR_SE_101_Epoch_24_Time_2020-11-08-12-25_checkpoint.pth'
        dul_model_path = './checkpoints/20201109_NPCFace_dul_reg/sota.pth'
        backbone_model = backbone.model_irse_org.IR_SE_101(INPUT_SIZE).to(device)
        dul_model = IR_SE_101(INPUT_SIZE).to(device)

        checkpoint = torch.load(backbone_model_path, map_location=lambda storage, loc: storage)
        backbone_model.load_state_dict(checkpoint)

        checkpoint = torch.load(dul_model_path, map_location=lambda storage, loc: storage)
        dul_model.load_state_dict(checkpoint['backbone'])

        backbone_model.eval()
        dul_model.eval()

        data_transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE[0], INPUT_SIZE[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        arrEnrollCropBatch = np.array(enrollCrops)
        Enroll_embeddings_org_list = []
        Enroll_embeddings_mu_list = []
        Enroll_embeddings_sigma_list = []

        for i, (image) in tqdm.tqdm(enumerate((arrEnrollCropBatch))):
            x, mu, sigma = faceFeatureExtract(image, data_transform, device, backbone_model, dul_model)
            x_flip, mu_flip, sigma_flip = faceFeatureExtract(image, data_transform, device, backbone_model, dul_model, flip=True)

            x = l2_norm(x + x_flip)
            mu = l2_norm(mu + mu_flip)
            x = x.cpu().data.numpy()
            mu = mu.cpu().data.numpy()
            sigma = np.concatenate((sigma, sigma_flip), axis=0)

            sigma_harmonic = 0.0
            for v in sigma:
                sigma_harmonic += 1.0 / v
            sigma_harmonic = sigma.shape[0] / sigma_harmonic

            Enroll_embeddings_org_list.append(x.reshape(-1))
            Enroll_embeddings_mu_list.append(mu.reshape(-1))
            Enroll_embeddings_sigma_list.append(sigma_harmonic)

        arrVerifCropBatch = np.array(verifCrops)
        Verif_embeddings_org_list = []
        Verif_embeddings_mu_list = []
        Verif_embeddings_sigma_list = []

        for i, (image) in tqdm.tqdm(enumerate((arrVerifCropBatch))):
            x, mu, sigma = faceFeatureExtract(image, data_transform, device, backbone_model, dul_model)
            x_flip, mu_flip, sigma_flip = faceFeatureExtract(image, data_transform, device, backbone_model, dul_model, flip=True)

            x = l2_norm(x + x_flip)
            mu = l2_norm(mu + mu_flip)
            x = x.cpu().data.numpy()
            mu = mu.cpu().data.numpy()
            sigma = np.concatenate((sigma, sigma_flip), axis=0)

            sigma_harmonic = 0.0
            for v in sigma:
                sigma_harmonic += 1.0 / v
            sigma_harmonic = sigma.shape[0] / sigma_harmonic

            Verif_embeddings_org_list.append(x.reshape(-1))
            Verif_embeddings_mu_list.append(mu.reshape(-1))
            Verif_embeddings_sigma_list.append(sigma_harmonic)

        np.save('Enroll_embeddings_org_list_' + dataset, np.array(Enroll_embeddings_org_list))
        np.save('Enroll_embeddings_mu_list_' + dataset, np.array(Enroll_embeddings_mu_list))
        np.save('Enroll_embeddings_sigma_list_' + dataset, np.array(Enroll_embeddings_sigma_list))
        np.save('Verif_embeddings_org_list_' + dataset, np.array(Verif_embeddings_org_list))
        np.save('Verif_embeddings_mu_list_' + dataset, np.array(Verif_embeddings_mu_list))
        np.save('Verif_embeddings_sigma_list_' + dataset, np.array(Verif_embeddings_sigma_list))

    for i in range(len(dfVerif)):
        if dfEnroll.at[i,'FaceDetectionError'] == False:
            # dfEnroll.at[i, 'features'] = Enroll_embeddings_org_list[i]
            dfEnroll.at[i, 'features'] = Enroll_embeddings_mu_list[i]
            dfEnroll.at[i, 'uncertain_score'] = Enroll_embeddings_sigma_list[i]
        if dfVerif.at[i,'FaceDetectionError'] == False:
            # dfVerif.at[i, 'features'] = Verif_embeddings_org_list[i]
            dfVerif.at[i, 'features'] = Verif_embeddings_mu_list[i]
            dfVerif.at[i, 'uncertain_score'] = Verif_embeddings_sigma_list[i]

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
        sigmaEnroll = filterEnroll['uncertain_score'].item()
        sigmaVerif = filterVerif['uncertain_score'].item()
        if dfEnroll.at[i, 'FaceDetectionError'] == True or dfVerif.at[i, 'FaceDetectionError'] == True:
            dfMatch.at[i, 'score'] = 0
            dfMatch.at[i, 'VerifTemplateError'] = 'FaceDetectionError'
        else:
            dfMatch.at[i, 'score'] = compareSimilarity(featureEnroll,featureVerif)
            dfMatch.at[i, 'uncertainty'] = compareUncertainty(featureEnroll, sigmaEnroll, featureVerif, sigmaVerif)
            dfMatch.at[i, 'VerifTemplateError'] = 'MatchSuccess'
        if dfEnroll.at[i, 'UUID'] == dfVerif.at[i, 'UUID']:
            dfMatch.at[i, 'GIlabel'] = 'Genuine'
        else:
            dfMatch.at[i, 'GIlabel'] = 'Imposter'

    plotGIBoxScatter(dfMatch)
    plotGIUncertain(dfEnroll, dfVerif, dfMatch)


if __name__ == '__main__':
    crop_dir = 'pnas_crop'
    main('pnasInput', crop_dir)
    crop_dir = 'mugshot_crop'
    main('mugshotInput', crop_dir)
