import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time

def load_hand_model(model_path):
    return load_model(model_path)

def initialize_mediapipe_hands():
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing

def detect_hands(frame, hands):
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(framergb)

def extract_hand_bbox(handLMs, w, h):
    x_max, y_max, x_min, y_min = w, h, 0, 0
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
    return x_min, y_min, x_max, y_max

def preprocess_frame(frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    analysis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    analysis_frame = analysis_frame[y_min:y_max, x_min:x_max]
    analysis_frame = cv2.resize(analysis_frame, (28, 28))
    return analysis_frame

def predict_gesture(model, analysis_frame, letterpred):
    pixeldata = analysis_frame.reshape(-1, 28, 28, 1) / 255
    prediction = model.predict(pixeldata)[0]
    top_predictions_indices = prediction.argsort()[-3:][::-1]
    top_predictions = [(letterpred[i], prediction[i]) for i in top_predictions_indices]
    return top_predictions

def main():
    # Load pre-trained model
    model_path = 'saved_model2.h5'
    model = load_hand_model(model_path)

    # Initialize MediaPipe Hands module
    hands, mp_drawing = initialize_mediapipe_hands()

    # Define gesture labels
    letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    # Open video capture device
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape

    print("RECORDING!!!!!")  # Indicate recording
    while True:
        _, frame = cap.read()
        k = cv2.waitKey(1)

        if k % 256 == 27:
            print("Escape hit, closing...")  # Indicate closing
            break
        elif k % 256 == 32:
            analysis_frame = frame.copy()
            show_frame = analysis_frame
            cv2.imshow("Frame", show_frame)

            # Hand analysis
            result_analysis = detect_hands(analysis_frame, hands)
            hand_landmarks_analysis = result_analysis.multi_hand_landmarks

            if hand_landmarks_analysis:
                for handLMsanalysis in hand_landmarks_analysis:
                    bbox = extract_hand_bbox(handLMsanalysis, w, h)
                    analysis_frame = preprocess_frame(analysis_frame, bbox)
                    top_predictions = predict_gesture(model, analysis_frame, letterpred)
                    # Print predictions
                    for i, (letter, confidence) in enumerate(top_predictions, start=1):
                        print(f"Predicted Character {i}: {letter}, Confidence: {100 * confidence:.2f}%")

                    # Show predictions on frame (optional)

        else:
            # Hand detection and tracking
            result = detect_hands(frame, hands)
            hand_landmarks = result.multi_hand_landmarks

            if hand_landmarks:
                for handLMs in hand_landmarks:
                    bbox = extract_hand_bbox(handLMs, w, h)
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
            cv2.imshow("Frame", frame)

    time.sleep(5)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
