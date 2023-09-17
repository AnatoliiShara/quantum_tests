from abc import ABC, abstractmethod

class DigitClassificationInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass

import tensorflow as tf

class CNNModel(DigitClassificationInterface):
    def __init__(self, model_path):
        # Load the pre-trained CNN model from model_path using TensorFlow/Keras
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image):
        # Preprocess the image (resize, normalize, etc.)
        # Make predictions using the CNN model
        predictions = self.model.predict(image)
        return predictions.argmax()

from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(DigitClassificationInterface):
    def __init__(self, model_path):
        # Load the pre-trained Random Forest model from model_path using scikit-learn
        self.model = joblib.load(model_path)

    def predict(self, image):
        # Flatten the image to a 1D array (784 pixels)
        image = image.reshape(1, -1)
        # Make predictions using the Random Forest model
        prediction = self.model.predict(image)
        return prediction[0]



from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(DigitClassificationInterface):
    def __init__(self, model_path):
        # Load the pre-trained Random Forest model from model_path using scikit-learn
        self.model = joblib.load(model_path)

    def predict(self, image):
        # Flatten the image to a 1D array (784 pixels)
        image = image.reshape(1, -1)
        # Make predictions using the Random Forest model
        prediction = self.model.predict(image)
        return prediction[0]


import random

class RandomModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image):
        # Generate a random digit prediction (0-9)
        return random.randint(0, 9)



class DigitClassifier:
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.model = CNNModel(model_path='path_to_cnn_model')
        elif algorithm == 'rf':
            self.model = RandomForestModel(model_path='path_to_rf_model')
        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError("Invalid algorithm name")

    def predict(self, image):
        return self.model.predict(image)
