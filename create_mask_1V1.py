import os
import numpy as np
import pandas as pd


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
    # print(dfEnroll.head())
    # print(dfVerif.head())
    # print(dfMatch.head())
    return dfEnroll,dfVerif,dfMatch


def generate_file_list(data_dir, file_list):
    for dir, dirs, files in os.walk(data_dir):
        for index, file in enumerate(files):
            filepath = os.path.join(dir, file)
            if filepath.endswith(".jpg") or filepath.endswith(".JPG") or filepath.endswith(".png"):
                file_list.append(filepath)


# protocol_folder = "mugshotInput"
# dataset_folder = "../face_dataset/NIST_SD32_MEDS_II_face_crop"
# masked_dataset_folder = "../face_dataset/NIST_SD32_MEDS_II_face_crop_mask"
# file_name = "MEDS_mask_pairs.txt"

protocol_folder = "mugshotInput"
dataset_folder = "../face_dataset/NIST_SD32_MEDS_II_face_RetinaFaceCoV_crop"
masked_dataset_folder = "../face_dataset/NIST_SD32_MEDS_II_face_FMA_mask_RetinaFaceCoV_crop"
file_name = "MEDS_mask_RetinaFaceCoV_crop_pairs.txt"

dfEnroll, dfVerif, dfMatch = loadDataframe(protocol_folder)

data_list = []
masked_data_list = []

generate_file_list(dataset_folder, data_list)
generate_file_list(masked_dataset_folder, masked_data_list)

meds_file = open(file_name, 'w')

for i in range(len(dfMatch)):
    enrollId = dfMatch.at[i, 'enroll']
    verifId = dfMatch.at[i, 'verif']
    filterEnroll = dfEnroll[dfEnroll['id'] == int(enrollId)]
    filterVerif = dfVerif[dfVerif['id'] == int(verifId)]

    Genuine_flag = "1"
    if dfEnroll.at[i, 'UUID'] != dfVerif.at[i, 'UUID']:
        Genuine_flag = "0"

    enroll_file_name = filterEnroll['path'].item().split('/')[-1].replace('ppm', 'jpg')
    verif_file_name = filterVerif['path'].item().split('/')[-1].replace('ppm', 'jpg')

    new_enroll_file_list = []
    new_verif_file_list = []
    # unmasked
    for file_path in data_list:
        if file_path.find(enroll_file_name) >= 0:
            new_enroll_file_list.append(file_path)

    # masked
    for file_path in masked_data_list:
        if file_path.find(verif_file_name.split('.jpg')[0]) >= 0:
            new_verif_file_list.append(file_path)

    if len(new_enroll_file_list) > 0 and len(new_verif_file_list) > 0:
        for enroll_name in new_enroll_file_list:
            for verif_name in new_verif_file_list:
                write_line = enroll_name + " " + verif_name + " " + Genuine_flag + "\n"
                print(write_line)
                meds_file.write(write_line)

    new_enroll_file_list = []
    new_verif_file_list = []
    # unmasked
    for file_path in data_list:
        if file_path.find(verif_file_name) >= 0:
            new_enroll_file_list.append(file_path)

    # masked
    for file_path in masked_data_list:
        if file_path.find(enroll_file_name.split('.jpg')[0]) >= 0:
            new_verif_file_list.append(file_path)

    if len(new_enroll_file_list) > 0 and len(new_verif_file_list) > 0:
        for enroll_name in new_enroll_file_list:
            for verif_name in new_verif_file_list:
                write_line = enroll_name + " " + verif_name + " " + Genuine_flag + "\n"
                print(write_line)
                meds_file.write(write_line)