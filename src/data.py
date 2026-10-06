import tensorflow as tf
import matplotlib.pyplot as plt

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

print(train_images.min())
print(train_images.max())

""" plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[i], cmap="gray")
    plt.title(labels[train_labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show() """