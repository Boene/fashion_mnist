import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

train_images = train_images / 255.0
test_images = test_images / 255.0

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

train_images_split, val_images, train_labels_split, val_labels = (
    train_test_split(
        train_images,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels
    )
)