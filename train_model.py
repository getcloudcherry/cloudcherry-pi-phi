import cv2
from dlib import get_frontal_face_detector, shape_predictor
import numpy as np
import math
import glob
import random
import six
from sklearn.svm import SVC
import pickle
import matplotlib.image as mpimg


emotions = ["happy", "sadness", "anger", "fear", "neutral", "disgust", "surprise" ]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = get_frontal_face_detector()
predictor = shape_predictor("/home/deeplearning/Desktop/shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-3)
data = {}


def get_files(emotion):
    files = glob.glob("/home/deeplearning/PycharmProjects/project2/sorted_set/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction


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
            mean_array = np.asarray((ymean, xmean))
            coordinates_array = np.asarray((z, w))
            normalized_distance = np.linalg.norm(coordinates_array - mean_array)
            vectorized_landmarks.append(normalized_distance)
            vectorized_landmarks.append((math.atan2(y, x)*360)/(2*math.pi))
    if len(detected) < 1:
        data["landmark_vectorization"] = "error"
    data["landmark_vectorization"] = vectorized_landmarks


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    prediction_images = []
    for emotion in emotions:
        print("Currently training %s emotion" % emotion)
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data["landmark_vectorization"] == "error":
                print("no faces detected")
            else:
                training_data.append(data["landmark_vectorization"])
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            prediction_images.append(item)
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data["landmark_vectorization"] == "error":
                print("no faces detected")
            else:
                prediction_data.append(data["landmark_vectorization"])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels, prediction_images


accuracy_linear = [0]
for i in range(0, 10):
    print("Making sets %s" % i)
    training_data, training_labels, prediction_data, prediction_labels, prediction_images = make_sets()
    nparray_train = np.array(training_data)
    nparray_train_labels = np.array(training_labels)
    print("Testing SVM Linear test %s" % i)
    clf.fit(nparray_train, nparray_train_labels)
    print("Getting accuracy %s" % i)
    nparray_prediction = np.array(prediction_data)
    prediction_linear = clf.score(nparray_prediction, prediction_labels)
    probabilities = clf.predict_proba(nparray_prediction)
    #for x,y in zip(probabilities, prediction_images):  #loop to show output for each image
    #    print("prediction is " + str(x) + " image is " + str(y))
    print("Linear: " + str(prediction_linear))
    if prediction_linear > np.max(accuracy_linear):
        pickle._dump(clf, open("/home/deeplearning/PycharmProjects/project2/model.xml", 'wb'))
    accuracy_linear.append(prediction_linear)
accuracy_linear.pop(0)
print("Mean accuracy: " + str(np.mean(accuracy_linear)))
print("Max accuracy: " + str(np.max(accuracy_linear)))
