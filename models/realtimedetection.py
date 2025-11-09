import cv2
import time
from keras.models import model_from_json
import numpy as np
# from keras_preprocessing.image import load_img
json_file = open("models/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("models/facialemotionmodel.h5") 
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

im = cv2.imread("models/images/train/disgust/388.jpg")
# print(type(im))
def return_emotion(image):
    labels = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happy', 4 : 'Neutral', 5 : 'Sad', 6 : 'Surprise'}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(image,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(image,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            print(prediction_label)
            return prediction_label
    except cv2.error:
        return "No face detected"

emotion_label = return_emotion(im)

# print(emotion_label)
# webcam=cv2.imread("/images/siris.jpg")
# emotion recognition from image code here
# labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
# current_emotion_label = None
# start_time = None
# im = cv2.imread("images/siris.jpg")
# gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# faces=face_cascade.detectMultiScale(im,1.3,5)
# emotion_captured = False
# try: 
#     for (p,q,r,s) in faces:
#         image = gray[q:q+s,p:p+r]
#         cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
#         image = cv2.resize(image,(48,48))
#         img = extract_features(image)
#         pred = model.predict(img)
#         prediction_label = labels[pred.argmax()]
        
#         cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
#         # if prediction_label == current_emotion_label:
#         #     if start_time is None:
#         #         start_time = time.time()
#         #     elif time.time() - start_time >=2:
#         #         print("Same emotion " + current_emotion_label + " for 2 seconds. Breaking the loop.")
#         #         emotion_captured = True
#         #         break
#         # else:
#         #     start_time = None
        
#         current_emotion_label = prediction_label
#     # cv2.imshow("Output",im)
#     cv2.waitKey(27)
#     print("Emotion captured: " + current_emotion_label)
# except cv2.error:
    

# emotion recognition from webcam code here
# while True:
#     i,im=webcam.read()
#     gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     faces=face_cascade.detectMultiScale(im,1.3,5)
#     emotion_captured = False
#     try: 
#         for (p,q,r,s) in faces:
#             image = gray[q:q+s,p:p+r]
#             cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
#             image = cv2.resize(image,(48,48))
#             img = extract_features(image)
#             pred = model.predict(img)
#             prediction_label = labels[pred.argmax()]
            
#             cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
#             # if prediction_label == current_emotion_label:
#             #     if start_time is None:
#             #         start_time = time.time()
#             #     elif time.time() - start_time >=2:
#             #         print("Same emotion " + current_emotion_label + " for 2 seconds. Breaking the loop.")
#             #         emotion_captured = True
#             #         break
#             # else:
#             #     start_time = None
            
#             current_emotion_label = prediction_label
#         cv2.imshow("Output",im)
#         cv2.waitKey(27)
#         if emotion_captured:
#             break
#     except cv2.error:
#         pass
    