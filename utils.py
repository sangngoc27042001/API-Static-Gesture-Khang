import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
CLASSES = ['fist', 'like', 'ok', 'one', 'palm', 'peace', 'rock', 'stop']

class HandGestureStaticClassifier:
    def __init__(self, file_weight):
        self.file_weight = file_weight
        self.model_name, self.file_weight_type = file_weight.split('.')
        self.model = pickle.load(open(file_weight,'rb')) if self.file_weight_type == 'pkl' else tf.keras.models.load_model(file_weight)
    
    def predict(self, landmarks, min_conf=0.5):

        landmarks_std_scaled = StandardScaler().fit_transform(landmarks)
        landmarks_Chebyshev = pdist(landmarks_std_scaled, 'chebyshev')
        input = np.concatenate([landmarks_std_scaled.reshape(-1), landmarks_Chebyshev], axis=0).reshape(1,-1)

        if self.model_name == 'SVM':
            return self.model.predict(input)[0]
        
        proba =  self.model.predict_proba(input)[0] if self.file_weight_type == 'pkl'else self.model.predict(input)[0]

        return proba.argmax() if proba.max()>=min_conf else None
                