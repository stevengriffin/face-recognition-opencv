# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import glob
import torch
from load import build_dataloaders
from model import build_mlp

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--testdir", default="test_dataset/",
            help="path to test dataset")
    ap.add_argument("-r", "--recognizer", default="output/recognizer.pt",
            help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", default="output/le.pickle",
            help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.0,
            help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load the actual face recognition model along with the label encoder
    dataloaders, attrib_dict = build_dataloaders()
    device = torch.device('cpu')
    recognizer = build_mlp(dataloaders, attrib_dict, device)
    try:
        recognizer.load_state_dict(torch.load(args['recognizer'], map_location=device))
    except Exception as e:
        print("Could not load MLP. " + str(e))
        sys.exit()
    recognizer.eval()
    le = pickle.loads(open(args["le"], "rb").read())

    test_dir = args["testdir"]
    if not test_dir.endswith("/"):
        test_dir += "/"
    print(test_dir)

    running_total = 0
    running_corrects = 0
    for step, (inputs, labels) in enumerate(dataloaders['val']):
        outputs = recognizer.forward(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        running_total += len(preds)
    acc = running_corrects.double().item() / attrib_dict['val'].num_examples
    print("Acc: " + str(acc))

def recognize(image_file, le, recognizer, detector, embedder, min_confidence):

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(image_file)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    # only first for now
    for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            names = []

            # filter out weak detections
            if confidence > min_confidence:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = image #[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                            continue


                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                            (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    inputs = torch.tensor(vec, dtype=torch.float)
                    outputs = recognizer.forward(inputs)
                    _, preds = torch.max(outputs, 1)

                    # perform classification to recognize the face
                    #preds = recognizer.predict_proba(vec)[0]
                    #j = np.argmax(preds)
                    #proba = preds[j]
                    name = le.classes_[preds]
                    #print("name " + str(name))
                    #print("j " + str(j))
                    names.append(name)

            return names


'''
                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                    cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
'''

if __name__ == '__main__':
    main()
