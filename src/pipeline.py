import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from src.data import train_images_split, train_labels_split, val_images, val_labels, test_images, test_labels

class Pipeline:
    def __init__(self):
        self._models = []

    def add_model(self, creation_matrix: list):
        ### Shape of Pictures
        layers = [tf.keras.layers.Flatten(input_shape=(28,28))]

        ### Implementing configuration of neural layers
        for neuron_count, activation_type in creation_matrix:
            layers.append(tf.keras.layers.Dense(neuron_count, activation=activation_type))
    
        ### Output layer with 10 categories
        layers.append(tf.keras.layers.Dense(10, activation="softmax"))

        ### Creation of model
        model = tf.keras.Sequential(layers)

        self._models.append(model)

    def run_pipeline(self, compile_matrix: list):
        for i in range(len(self._models)):
            model = self._models[i]
            
            model.compile(**compile_matrix[i])

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            history = model.fit(
                train_images_split,
                train_labels_split,
                epochs=30,
                validation_data=(val_images, val_labels),
                callbacks=[early_stopping]
            )

            test_loss, test_accuracy = model.evaluate(
                test_images,
                test_labels
            )

            labels = [
                "T-Shirt/Top",
                "Hose",
                "Pullover",
                "Kleid",
                "Mantel",
                "Sandale",
                "Hemd",
                "Sneaker",
                "Tasche",
                "Stiefelette"
            ]

            predictions = model.predict(test_images, verbose=0)
            predicted_labels = predictions.argmax(axis=1)

            val_losses = history.history["val_loss"]
            best_epoch = np.argmin(val_losses) + 1

            print(f"Model Nr: {i}")
            print("final Test Loss: ", test_loss)
            print("final Test Accuracy: ", test_accuracy)
            print("--- --- ---")
            print(f"Beste Epoche [loss]: {best_epoch}")
            print(f"Validation Loss: {val_losses[best_epoch - 1]:.4f}")
            print("###################################")