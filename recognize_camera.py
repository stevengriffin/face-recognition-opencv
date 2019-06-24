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
    ap.add_argument("-d", "--detector", default="face_detection_model",
            help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", default="nn4.small2.v1.t7",
            help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", default="output/recognizer.pt",
            help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", default="output/le.pickle",
            help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.0,
            help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

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



def recognize(image, le, recognizer, detector, embedder, min_confidence):

    # resize the image to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    names = []

    # loop over the detections
    # only first for now
    for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]


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
                    name = le.classes_[preds]
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

def process_stream():
    # loop over frames from the video file stream
    while True:
            # grab the frame from the threaded video stream
            frame = vs.read()

            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(rgb,
                    model=args["detection_method"])
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            # loop over the facial embeddings
            for encoding in encodings:
                    # attempt to match each face in the input image to our known
                    # encodings
                    matches = face_recognition.compare_faces(data["encodings"],
                            encoding)
                    name = "Unknown"

                    # check to see if we have found a match
                    if True in matches:
                            # find the indexes of all matched faces then initialize a
                            # dictionary to count the total number of times each face
                            # was matched
                            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                            counts = {}

                            # loop over the matched indexes and maintain a count for
                            # each recognized face face
                            for i in matchedIdxs:
                                    name = data["names"][i]
                                    counts[name] = counts.get(name, 0) + 1

                            # determine the recognized face with the largest number
                            # of votes (note: in the event of an unlikely tie Python
                            # will select first entry in the dictionary)
                            name = max(counts, key=counts.get)

                    # update the list of names
                    names.append(name)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                    # rescale the face coordinates
                    top = int(top * r)
                    right = int(right * r)
                    bottom = int(bottom * r)
                    left = int(left * r)

                    # draw the predicted face name on the image
                    cv2.rectangle(frame, (left, top), (right, bottom),
                            (0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            # if the video writer is None *AND* we are supposed to write
            # the output video to disk initialize the writer
            if writer is None and args["output"] is not None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(args["output"], fourcc, 20,
                            (frame.shape[1], frame.shape[0]), True)

            # if the writer is not None, write the frame with recognized
            # faces t odisk
            if writer is not None:
                    writer.write(frame)

            # check to see if we are supposed to display the output frame to
            # the screen
            if args["display"] > 0:
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                            break

    # do a bit of cleanup
    cv2.destroyAllWindows()

    # check to see if the video writer point needs to be released
    if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()

