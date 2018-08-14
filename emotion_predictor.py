import math
import pickle
from dlib import get_frontal_face_detector, shape_predictor
import cv2
import numpy as np
import time
from picamera import PiCamera
from picamera import PiCamera

data = {}
prediction_labels = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #create histogram equalization parameters
detector = get_frontal_face_detector()
predictor = shape_predictor("/home/pi/Downloads/shape_predictor_68_face_landmarks.dat") #file to get 68 landmarks 
clf = pickle.load(open("/home/pi/Downloads/model.sav", "rb"))
emotions = ["happy", "anger", "fear", "sadness", "disgust", "surprise", "neutral"]
prediction_image = []


def get_face(image):   #function to crop face from image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceDet = cv2.CascadeClassifier("/home/pi/Downloads/haarcascade_frontalface_default.xml") #Haar classifiers to detect face
    faceDet2 = cv2.CascadeClassifier("/home/pi/Downloads/haarcascade_frontalface_alt2.xml")
    faceDet3 = cv2.CascadeClassifier("/home/pi/Downloads/haarcascade_frontalface_alt.xml")
    faceDet4 = cv2.CascadeClassifier("/home/pi/Downloads/haarcascade_frontalface_alt_tree.xml")
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
        facefeatures = face4
    for (x, y, w, h) in facefeatures:  # Get coordinates and size of rectangle containing face
            gray = gray[y:y+h, x:x+w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #resize images to standard size
                clahe_image = clahe.apply(out) #applying histogram equalization
            except:
                pass
    return clahe_image


def get_landmarks(image): #function to produce features from facial landmarks
    detected = detector(image, 1)
    for k, d in enumerate(detected):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1,68):
            #cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
            xlist.append(shape.part(i).x) #appending x and y coordinates of facial landmarks
            ylist.append(shape.part(i).y)
        #cv2.imshow("got landmarks", image)
        #cv2.waitKey(3000)
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
            normalized_distance = np.linalg.norm(coordinates_array - mean_array) #distance from center to other landmarks
            vectorized_landmarks.append(normalized_distance)
            vectorized_landmarks.append((math.atan2(y, x)*360)/(2*math.pi))  #calculating face offset 
    if len(detected) < 1:
        data["landmark_vectorization"] = "error"
    data["landmark_vectorization"] = vectorized_landmarks #appending coordinates(model features) to dictionary
    return data



def get_emotion(data, count): #function to use facial features and run it through the ML model
    prediction_data = []
    if data["landmark_vectorization"] == "error":
        print("no faces detected")
    else:
        prediction_data.append(data["landmark_vectorization"])
    nparray = np.array(prediction_data)
    final_results = clf.predict_proba(nparray)[0] #getting probabilities of each emotion in the image
    print(final_results)
    font = cv2.FONT_HERSHEY_SIMPLEX
    initial_point = 100
    for a,b in zip(emotions, final_results):
        confidence_text = str(str(a) + " : " + str(b * 100) + " %")
        cv2.putText(prediction_image, confidence_text, (30, initial_point), font, 1, (255,0,0), 1, cv2.LINE_AA) #Writing results on the image
        initial_point = initial_point + 50
    final_text = str("Emotion : " + str(emotions[int(np.argmax(final_results))])) 
    cv2.putText(prediction_image, final_text, (30, initial_point), font, 1, (255,0,0), 1, cv2.LINE_AA) #Writing final emotion prediction to image
    cv2.imwrite(str("/home/pi/Desktop/results/image_emotion" + str(count) + ".jpg"), prediction_image) #saving image with results locally



count = 1
camera = PiCamera()
while True:
    camera.start_preview()
    time.sleep(10) #sleep for 10 seconds before picture is taken
    image_path = str("/home/pi/Desktop/results/image" + str(count) + ".jpg")
    camera.capture(image_path)
    camera.stop_preview()
    prediction_image = cv2.imread(image_path)
    clahe_image = get_face(prediction_image)
    data = get_landmarks(clahe_image)  #getting features for image
    get_emotion(data, count)
    count = count + 1
    if count == 11:
        break
    else:
        time.sleep(10) #wait for 10 seconds to capture next emotion image
        continue
print("images and emotion results saved to local machine.")
