# agents/recognition_agent.py
import numpy as np
from tensorflow.keras.models import load_model

class RecognitionAgent:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    def classify(self, landmarks):
        # Preprocess landmarks to model input
        input_data = np.array(landmarks).reshape(1, -1)
        prediction = self.model.predict(input_data)
        return np.argmax(prediction)