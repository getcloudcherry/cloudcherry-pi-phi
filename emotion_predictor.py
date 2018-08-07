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
clf = pickle.load(open("/home/deeplearning/PycharmProjects/project2/model.sav", "rb"))
emotions = ["happy", "anger", "fear", "sadness", "disgust", "surprise", "neutral"]


def get_landmarks(image):
    detected = detector(image, 1)
    for k, d in enumerate(detected):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1,68):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
            xlist.append(shape.part(i).x)
            ylist.append(shape.part(i).y)
        cv2.imshow("got landmarks", image)
        cv2.waitKey(3000)
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        vectorized_landmarks = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
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


prediction_image = cv2.imread("/home/deeplearning/Desktop/face_recognition examples/obama-720p.jpg")
gray = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("initial image", gray)
cv2.waitKey(3000)
faceDet = cv2.CascadeClassifier("/home/deeplearning/PycharmProjects/project2/haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("/home/deeplearning/PycharmProjects/project2/haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("/home/deeplearning/PycharmProjects/project2/haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("/home/deeplearning/PycharmProjects/project2/haarcascade_frontalface_alt_tree.xml")
face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

if len(face) == 1:
    facefeatures = face
elif len(face2) == 1:
    facefeatures = face2
elif len(face3) == 1:
    facefeatures = face3
elif len(face4) == 1:
    facefeatures = ""
for (x, y, w, h) in facefeatures:  # Get coordinates and size of rectangle containing face
        gray = gray[y:y+h, x:x+w]  # Cut the frame to size
        try:
            out = cv2.resize(gray, (350, 350))
            clahe_image = clahe.apply(out)
            cv2.imshow("clahe image", clahe_image)
            cv2.waitKey(4000)
            get_landmarks(clahe_image)
        except:
            pass


if data["landmark_vectorization"] == "error":
    print("no faces detected")
else:
    prediction_data.append(data["landmark_vectorization"])
nparray = np.array(prediction_data)
final_results = clf.predict_proba(nparray)[0]
font = cv2.FONT_HERSHEY_SIMPLEX
initial_point = 100
for a,b in zip(emotions, final_results):
    confidence_text = str(str(a) + " : " + str(b * 100) + " %")
    cv2.putText(prediction_image, confidence_text, (30, initial_point), font, 1, (255,0,0), 1, cv2.LINE_AA)
    initial_point = initial_point + 50
final_text = str("Emotion : " + str(emotions[int(np.argmax(final_results))]))
cv2.putText(prediction_image, final_text, (30, initial_point), font, 1, (255,0,0), 1, cv2.LINE_AA)
cv2.imshow("final output", prediction_image)
cv2.waitKey(6000)


