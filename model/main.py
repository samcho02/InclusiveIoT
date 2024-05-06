# Author of this file: Sungmin Cho and Sujin Shin
# Adapted from the sign language recognition project described in the article:
# https://towardsdatascience.com/sign-language-recognition-with-advanced-computer-vision-7b74f20f3442
# Credit and acknowledgment to the original author for their valuable contributions.

# modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

# Load pre-trained model
model = load_model('saved_model2.h5')

# Initialize MediaPipe Hands module
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define gesture labels
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Open video capture device
cap = cv2.VideoCapture(0)  
_, frame = cap.read()
h, w, c = frame.shape

# Main loop
print("RECORDING!!!!!") # recording
while True:
    _, frame = cap.read()
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        print("Escape hit, closing...")  
        break
    
    elif k % 256 == 32:
        analysisframe = frame
        showframe = analysisframe
        cv2.imshow("Frame", showframe)
        
        # Hand analysis
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks

        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                # set dimensions
                x_max, y_max, x_min, y_min = 0, 0, w, h
                for lmanalysis in handLMsanalysis.landmark:
                    # get window to analyze
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    x_max = max(x, x_max)
                    x_min = min(x, x_min)
                    y_max = max(y, y_max)
                    y_min = min(y, y_min)
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20 


        analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
        analysisframe = analysisframe[y_min:y_max, x_min:x_max]
        analysisframe = cv2.resize(analysisframe, (28, 28))

        # Preprocess data for prediction
        pixeldata = analysisframe.reshape(-1, 28, 28, 1) / 255

        # Make prediction
        prediction = model.predict(pixeldata)[0]

        # Get top 3 predicted characters and confidence scores
        top_predictions_indices = prediction.argsort()[-3:][::-1]
        top_predictions = [(letterpred[i], prediction[i]) for i in top_predictions_indices]

        # Print predictions
        for i, (letter, confidence) in enumerate(top_predictions, start=1):
            print(f"Predicted Character {i}: {letter}, Confidence: {100 * confidence:.2f}%")
        
        # Show predictions on frame (optional)

    else:
        # Hand detection and tracking
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks

        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max, y_max, x_min, y_min = 0, 0, w, h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x, x_max)
                    x_min = min(x, x_min)
                    y_max = max(y, y_max)
                    y_min = min(y, y_min)
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
        cv2.imshow("Frame", frame)

# destroy sequence
time.sleep(5)
cap.release()
cv2.destroyAllWindows()  