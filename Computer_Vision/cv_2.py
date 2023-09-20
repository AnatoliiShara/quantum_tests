import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random

class DigitClassificationInterface:
    def __init__(self):
        pass

    def predict(self, image):
        raise NotImplementedError("Subclasses must implement this method")

class ConvolutionalNeuralNetwork(DigitClassificationInterface):
    def __init__(self):
        # Load and initialize CNN model using TensorFlow
        # Define model architecture
        self.model = self.build_model()
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def predict(self, image):
        # Perform CNN-based prediction
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = np.argmax(self.model.predict(image))
        return prediction

class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        # Load and initialize Random Forest model using sklearn package
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data.astype('float32')
        y = mnist.target.astype('int')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train a Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, image):
        # Flatten the input image to match the MNIST format (784 pixels)
        image = image.reshape(1, -1).astype('float32')
        # Make predictions using the trained Random Forest model
        prediction = self.model.predict(image)
        return prediction[0]

class RandomModel(DigitClassificationInterface):
    def __init__(self):
        # No need to load any model for this random model
        pass

    def predict(self, image):
        # Generate a random integer as a result
        prediction = random.randint(0, 9)
        return prediction

class DigitClassifier:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        if self.algorithm == 'cnn':
            self.model = ConvolutionalNeuralNetwork()
        elif self.algorithm == 'rf':
            self.model = RandomForestModel()
        elif self.algorithm == 'rand':
            self.model = RandomModel()

    def predict(self, image):
        # Perform prediction using the selected model
        return self.model.predict(image)

# Example usage:
if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_data = x_test[0]  
    algorithm = "cnn"  # Change this to "rf" or "rand" for different models

    model = DigitClassifier(algorithm)
    prediction = model.predict(image_data)
    print(f"Predicted digit: {prediction}")
