import cv2
import time
import numpy as np

print("OpenCV version:", cv2.__version__)
#initialize the webcam with face detector (Haar Cascade), set it to frontal and default
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#start the webcam
#set it to 0 which is the default camera device

#utilize pre trained models (downloaded from openCV github) Caffe models are generally best 
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')


#define age lists and gender lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['MALE', 'FEMALE']

#from the caffe model, we need to define the mean pixel intensity of the training data 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Error, Camera could not be opened.")
    exit()

time.sleep(3)
print("Webcam started successfully.")

print("Press 'q' to quit the webcam.")
#loop to control frame by frame analysis


while True:
    #read frames from cam
    #ret determines if the frame was read, frame is the actual image
    ret, frame = cam.read()
    if not ret:
        print("Error could not read frame.")
        break

    #convert the frame to grayscale
    #cvtColor converts (frame, to a given color space)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #create a var to hold the detected faces
    #scale factor reduces image size to seepd up detection at the cost of accuracy
    #minNeighbors define how many neighbors each candidate rectangle should have to retain it higher means few detects but better accuracy
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

    #pos and rectangle size do for every face
    for (x,y,w,h) in faces:

        #create an input blob from the face region to use for the models 
        blob = cv2.dnn.blobFromImage(frame[y:y+h, x:x+w], 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        #set up the models and their inputs
        gender_net.setInput(blob)
        gender_preds = gender_net.forward() #one forward pass for the predict
        #argmax to get index of pred
        gender = GENDER_LIST[gender_preds[0].argmax()]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        
        label = f"{gender}, {age}"

        #frame, pos, size, color, thickness
        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0,0), 2)
        #frame, label, pos, font, scale, color, thickness
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 2)

    cv2.imshow("Face Detection with Age and Gender", frame)

    #wait for 1ms and check if q was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting webcam.")
        break

cam.release()
cv2.destroyAllWindows()
