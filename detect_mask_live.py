from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import imutils
import numpy as np
import time
import os
import cv2 as cv

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector from disk
maskNet = load_model("mask_detector.model")


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # Initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask_network

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))

            # Extract face Region, convert it into RGB from BGR format
            # ordering resizing it to 224x224 and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of face locations and their corresponding
        # locations
    return (locs, preds)


# Initializing the video Stream
print("Starting Video Stream......")
vs = VideoStream(src='/dev/video2').start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1200)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (x1, y1, x2, y2) = box
        (mask, without_mask) = pred

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Including the probability in the label
        label = "{}:{:.2f}%".format(label, max(mask, without_mask)*100)

        cv.putText(frame, label, (x1, y1-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv.imshow("Live Mask Detector", frame)
        key = cv.waitKey(1)
        if key == 27:
            break

vs.stop()
cv.destroyAllWindows()
