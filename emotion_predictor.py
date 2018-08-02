import math
import pickle
from dlib import get_frontal_face_detector, shape_predictor
import cv2
import numpy as np

data = {}
prediction_data = []
prediction_labels = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = get_frontal_face_detector()
predictor = shape_predictor("/home/deeplearning/Desktop/shape_predictor_68_face_landmarks.dat")
clf = pickle.load(open("/home/deeplearning/PycharmProjects/project2/model.xml", "rb"))
emotions = ["happy", "anger", "fear", "sadness", "disgust", "surprise", "neutral"]


def get_landmarks(image):
    detected = detector(image, 1)
    for k, d in enumerate(detected):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1,68):
            xlist.append(shape.part(i).x)
            ylist.append(shape.part(i).y)
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        vectorized_landmarks = []
        for x,y,w,z in zip(xcentral, ycentral, xlist, ylist):
            vectorized_landmarks.append(w)
            vectorized_landmarks.append(z)
            mean_array = np.asarray((ymean,xmean))
            coordinates_array = np.asarray((z,w))
            normalized_distance = np.linalg.norm(coordinates_array - mean_array)
            vectorized_landmarks.append(normalized_distance)
            vectorized_landmarks.append((math.atan2(y, x)*360)/(2*math.pi))
    if len(detected) < 1:
        data["landmark_vectorization"] = "error"
    data["landmark_vectorization"] = vectorized_landmarks


prediction_image = cv2.imread("/home/deeplearning/Desktop/face_recognition examples/biden.jpg") #input your image
gray = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2GRAY)
clahe_image = clahe.apply(gray)
get_landmarks(clahe_image)
if data["landmark_vectorization"] == "error":
    print("no faces detected")
else:
    prediction_data.append(data["landmark_vectorization"])
nparray = np.array(prediction_data)
print(clf.predict_proba(nparray))
